import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange, repeat
from math import ceil
import math
import time

# feedforward and attention

class GhostNorm(nn.Module):
    def __init__(self, inner_norm, virtual_batch_size= 16):
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.inner_norm = inner_norm

    def forward(self, x):
        # If the input is three-dimensional, it is necessary to adjust its shape.
        original_shape = x.shape
        if len(original_shape) == 3:  # (batch_size, seq_length, feature_dim)
            x = x.view(-1, original_shape[-1])  # reshape (batch_size * seq_length, feature_dim)
        
        # Divide into virtual batches
        chunk_size = int(ceil(x.shape[0] / self.virtual_batch_size))
        chunk_norm = [self.inner_norm(chunk) for chunk in x.chunk(chunk_size, dim=0)]
        
        # Concatenate the normalized results
        x = torch.cat(chunk_norm, dim=0)
        
        # Restore the original shape
        if len(original_shape) == 3:
            x = x.view(original_shape)
        
        return x

class LeakyGate(nn.Module):
    """
    Performs an element-wise linear transformation followed by activation.
    Supports both 2D and 3D inputs.
    """

    def __init__(
        self,
        dim,
        activation = nn.ReLU,
    ):
        """
        Parameters
        ----------
        input_size : int
            Size of the last dimension (feature size).
        bias : bool, optional
            Whether to include a bias term; default is True.
        activation : torch.nn.Module, optional
            Activation function; default is nn.LeakyReLU.
        device : str or torch.device, optional
            Device for computation; default is "cpu".
        """
        super().__init__()
        self.linear = nn.Linear(dim, dim) 
        self.activation = activation()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (num_rows, num_fields) or (num_rows, num_fields, embedding_size).

        Returns
        -------
        torch.Tensor
            Transformed tensor with the same shape as input.
        """
        # Apply linear transformation
        out = self.linear(X)
        
        # Apply activation
        out = self.activation(out)
        
        return out

def MLP(dim, dropout = 0.1):
    return nn.Sequential(
        nn.Linear(dim, dim * 2),
        GhostNorm(inner_norm = nn.LayerNorm(dim * 2)),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 2, dim * 4),
        GhostNorm(inner_norm = nn.LayerNorm(dim * 4)),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 4, dim * 2),
        GhostNorm(inner_norm = nn.LayerNorm(dim * 2)),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 2, dim),
    )

class MLPP(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.LeakyGate = LeakyGate(dim)
        self.mlp = MLP(dim)
        self.linear = nn.Linear(dim , dim)
    def forward(self, x):
        x1 = self.LeakyGate(x)
        x2 = self.mlp(x1)
        x1 = self.linear(x1)
        return (x2 +x1) / 2




def expand_to_10_columns(vector, bins):
    """
    Expand the single-column tensor to 10 columns and fill it according to the specified rules.
    Parameters:

    vector: Input 3D torch tensor with dimensions (b, n, 1)
    bins: List of boundaries used for partitioning

    Returns:
    result: Expanded 3D tensor with dimensions (b, n, 10)
    """
    b, n, _ = vector.size()
    bins = torch.tensor(bins, device=vector.device)
    n_cols = len(bins) - 1  # Number of intervals = number of boundaries - 1

    # Compute the interval index for each value
    value = vector.squeeze(-1)
    indices = torch.bucketize(value, bins) - 1
    indices = indices.clamp(0, n_cols - 1)  # Ensure indices are within valid range

    # Initialize the result tensor
    result = torch.zeros((b, n, n_cols), device=vector.device)

    # Calculate normalized values: (value - min) / (max - min)
    min_vals = bins[indices]  # Minimum value of the current interval
    max_vals = bins[indices + 1]  # Maximum value of the current interval
    normalized_values = (value - min_vals) / (max_vals - min_vals + 1e-8)  # Avoid division by zero

    # Fill the result tensor
    row_indices = torch.arange(n_cols, device=vector.device)
    mask = row_indices.unsqueeze(0).unsqueeze(0) < indices.unsqueeze(-1)
    result[mask] = 1.0  # Fill previous columns with 1.0
    result.scatter_(2, indices.unsqueeze(-1), normalized_values.unsqueeze(-1))  # Fill the corresponding interval with normalized values

    return result


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types, device=None):
        super().__init__()
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        self.ran = [
                    -10, -2.15387469, -1.86273187, -1.67593972, -1.53412054, -1.41779714,
                    -1.3180109, -1.22985876, -1.15034938, -1.07751557, -1.00999017, -0.94678176,
                    -0.88714656, -0.83051088, -0.77642176, -0.72451438, -0.67448975, -0.62609901,
                    -0.57913216, -0.53340971, -0.48877641, -0.44509652, -0.40225007, -0.36012989,
                    -0.31863936, -0.27769044, -0.23720211, -0.19709908, -0.15731068, -0.11776987,
                    -0.07841241, -0.03917609, 0.0, 0.03917609, 0.07841241, 0.11776987,
                    0.15731068, 0.19709908, 0.23720211, 0.27769044, 0.31863936, 0.36012989,
                    0.40225007, 0.44509652, 0.48877641, 0.53340971, 0.57913216, 0.62609901,
                    0.67448975, 0.72451438, 0.77642176, 0.83051088, 0.88714656, 0.94678176,
                    1.00999017, 1.07751557, 1.15034938, 1.22985876, 1.3180109, 1.41779714,
                    1.53412054, 1.67593972, 1.86273187, 2.15387469, 10
                    ]
        self.linears = nn.ModuleList([
            nn.Linear(64, dim) for _ in range(num_numerical_types)
        ])
    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        x = expand_to_10_columns(x, self.ran)
        x = x.to(self.device)
        b, n, _ = x.size()
        outputs = []
        for i in range(n):
            # Use the corresponding linear layer for each sequence position
            outputs.append(self.linears[i](x[:, i, :]))
        # Concatenate the output of each position back into a single tensor
        return F.relu(torch.stack(outputs, dim=1))  


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.relu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.batch_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mult * 2)
        self.geglu1 = GEGLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * mult, dim * mult * 2)
        self.geglu2 = GEGLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(dim * mult, dim)

    def forward(self, x):

        x = self.batch_norm(x)

        x = self.fc1(x)
        x = self.geglu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.geglu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn


class Attention_final(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.q_norm = nn.LayerNorm(dim_head)

        self.q = nn.Parameter(torch.randn(1, heads, 1, dim_head))  # 初始化时扩展维度
        self.to_kv = nn.Linear(dim, inner_dim * 2)
        self.to_out = nn.Linear(inner_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        q = self.q_norm(self.q)
        h = self.heads
        
        x = self.norm(x)

        kv = self.to_kv(x)  # 分割 key 和 value
        k, v = kv.chunk(2, dim=-1)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (k, v))
        q = q * self.scale
        q = q.expand(b, -1, -1, -1) 

        sim = torch.einsum('b h t d, b h s d -> b h t s', q, k)
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)
        out = torch.einsum('b h t s, b h s d -> b h t d', dropped_attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.to_out(out)
        
        return out, attn

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))
        self.Attention_final = Attention_final(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)
        self.FeedForward_final = FeedForward(dim, dropout = ff_dropout)
    def forward(self, x, return_attn = False):
        post_softmax_attns = []
        x_org_1 = x
        for attn, ff in self.layers:
            # x_org_1 = x
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x

            x = ff(x) + x
        x = torch.cat((x,x_org_1),dim = 1) # RAT
        x, att = self.Attention_final(x)
        x = self.FeedForward_final(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# numerical embedder

class DivideBySqrtDk(nn.Module):
    def __init__(self, dim):
        super(DivideBySqrtDk, self).__init__()
        self.scale = math.sqrt(dim)  # compute d_k

    def forward(self, x):

        return x / self.scale  # Perform the operation of dividing by the square root of d_k

# main class

class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        batch_size = 64,
        dim_head = 32,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        try:
            assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
            # categories related calculations

            self.num_categories = len(categories)
            self.num_unique_categories = sum(categories)
            self.batch_size = batch_size
            self.num_tokens = len(categories) + num_continuous
            # create category embeddings table
            self.dim_out = dim_out
            self.num_special_tokens = num_special_tokens
            total_tokens = self.num_unique_categories + num_special_tokens

            # for automatically offsetting unique category ids to the correct position in the categories embedding table

            if self.num_unique_categories > 0:
                categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
                categories_offset = categories_offset.cumsum(dim=-1)[:-1]
                self.register_buffer('categories_offset', categories_offset)

                # categorical embedding

                self.categorical_embeds = nn.Embedding(total_tokens, dim)
        except AssertionError as e:
            # print(f"Assertion failed: {e}. Applying fallback...")
            # 执行你的操作，例如处理错误数据
            self.num_unique_categories = 0
        try:
            assert len(categories) + num_continuous > 0, 'input shape must not be null'

            # continuous

            self.num_continuous = num_continuous

            if self.num_continuous > 0:
                self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)
        except AssertionError as e:
            self.num_continuous = 0


        # transformer

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
        )



        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out),
            DivideBySqrtDk(dim_out),
        )

        # to HGCN
        self.LeakyGate = LeakyGate(dim=dim)
        self.mlpp = MLPP(dim=dim)
        self.mlpp_1 = MLPP(dim=dim)
        self.mlpp_2 = MLPP(dim=dim)
        self.mlpp_3 = MLPP(dim=dim)
    def forward(self, x_numer = None, x_categ = None, return_attn = False):

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical
        
        x = torch.cat(xs, dim = 1)
        

        
        x, attns = self.transformer(x, return_attn = True)

        x = x.mean(dim=1)

        x = self.mlpp(x)

        logits = self.to_logits(x)
           
        if not return_attn:
            return logits

        return logits, attns
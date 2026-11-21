from typing import Union, Final
from torch_geometric.nn import GATConv
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
import typing
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, NoneType, OptTensor, PairTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, is_torch_sparse_tensor, remove_self_loops, softmax
from torch_geometric.utils.sparse import set_sparse_value
if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload
from torch_geometric.nn.models.basic_gnn import BasicGNN
from utils import set_topk_values

def jackson_density_correction(dist):
    dist = torch.where(torch.isinf(dist), torch.zeros_like(dist), dist)
    return 1 / torch.sum(dist, dim=1, keepdim=True)

def jackson_density_correction_pre(pos1):
    dist_1 = torch.norm(pos1.unsqueeze(0) - pos1.unsqueeze(1), dim=-1)
    dist_1_sum = torch.sum(dist_1, dim=0, keepdim=True)
    return 1 / dist_1_sum

def interpolate_from_graphs(pos1, features1, pos2, dist_thres=torch.inf, topk=None, density_correction='jackson', interp_kernel=None):
    pos1 = pos1.unsqueeze(0)
    pos2 = pos2.unsqueeze(1)
    if interp_kernel is None:
        dist = torch.norm(pos1 - pos2, dim=-1)
        dist[dist > dist_thres] = torch.inf
    else:
        assert dist_thres == torch.inf
        dist_pre = pos1 - pos2
        dist_pre_d = torch.norm(dist_pre, dim=-1).unsqueeze(-1)
        dist_pre = torch.abs(dist_pre)
        dist_pre = torch.cat([dist_pre, dist_pre_d], dim=-1)
        dist = interp_kernel(dist_pre).squeeze(-1)
    if topk is not None:
        assert dist_thres == torch.inf
        dist = set_topk_values(dist, topk)
    if density_correction == 'jackson':
        density = jackson_density_correction(dist if interp_kernel is None else dist_pre_d.squeeze(-1))
    elif density_correction == 'jackson_pre':
        density = jackson_density_correction_pre(pos1.squeeze(0))
    elif density_correction == 'uniform':
        density = 1
    else:
        raise ValueError(f'Unknown density correction method: {density_correction}')
    density = density / torch.mean(density)
    dist = dist * density
    dist_exp = torch.exp(-dist)
    weights = dist_exp / (dist_exp.sum(dim=-1, keepdim=True) + 1e-05)
    return torch.matmul(weights, features1)

def back_gridding(vals, coords, im_size=(64, 64)):
    device = vals.device
    N, C = vals.shape
    pos2 = torch.meshgrid(torch.arange(im_size[0]), torch.arange(im_size[1]), indexing='ij')
    pos2 = torch.stack(pos2, axis=-1).reshape(-1, 2)
    grid_values = interpolate_from_graphs(coords.to(dtype=torch.float32), vals, pos2.to(dtype=torch.float32, device=device))
    recon = grid_values.reshape(im_size[0], im_size[1], C).permute(2, 0, 1).contiguous()
    return recon


class GATv2Conv(MessagePassing):
    """The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
    static attention problem of the standard
    :class:`~torch_geometric.conv.GATConv` layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in :class:`GATv2`, every node can attend to any other node.

    .. math::
        \\mathbf{x}^{\\prime}_i = \\sum_{j \\in \\mathcal{N}(i) \\cup \\{ i \\}}
        \\alpha_{i,j}\\mathbf{\\Theta}_{t}\\mathbf{x}_{j},

    where the attention coefficients :math:`\\alpha_{i,j}` are computed as

    .. math::
        \\alpha_{i,j} =
        \\frac{
        \\exp\\left(\\mathbf{a}^{\\top}\\mathrm{LeakyReLU}\\left(
        \\mathbf{\\Theta}_{s} \\mathbf{x}_i + \\mathbf{\\Theta}_{t} \\mathbf{x}_j
        \\right)\\right)}
        {\\sum_{k \\in \\mathcal{N}(i) \\cup \\{ i \\}}
        \\exp\\left(\\mathbf{a}^{\\top}\\mathrm{LeakyReLU}\\left(
        \\mathbf{\\Theta}_{s} \\mathbf{x}_i + \\mathbf{\\Theta}_{t} \\mathbf{x}_k
        \\right)\\right)}.

    If the graph has multi-dimensional edge features :math:`\\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\\alpha_{i,j}` are computed as

    .. math::
        \\alpha_{i,j} =
        \\frac{
        \\exp\\left(\\mathbf{a}^{\\top}\\mathrm{LeakyReLU}\\left(
        \\mathbf{\\Theta}_{s} \\mathbf{x}_i
        + \\mathbf{\\Theta}_{t} \\mathbf{x}_j
        + \\mathbf{\\Theta}_{e} \\mathbf{e}_{i,j}
        \\right)\\right)}
        {\\sum_{k \\in \\mathcal{N}(i) \\cup \\{ i \\}}
        \\exp\\left(\\mathbf{a}^{\\top}\\mathrm{LeakyReLU}\\left(
        \\mathbf{\\Theta}_{s} \\mathbf{x}_i
        + \\mathbf{\\Theta}_{t} \\mathbf{x}_k
        + \\mathbf{\\Theta}_{e} \\mathbf{e}_{i,k}]
        \\right)\\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities in case of a bipartite graph.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge,
            *i.e.* :math:`\\mathbf{\\Theta}_{s} = \\mathbf{\\Theta}_{t}`.
            (default: :obj:`False`)
        residual (bool, optional): If set to :obj:`True`, the layer will add
            a learnable skip-connection. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\\mathcal{V}|, F_{in})` or
          :math:`((|\\mathcal{V_s}|, F_{s}), (|\\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\\mathcal{E}|)`,
          edge features :math:`(|\\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\\mathcal{V}|, H * F_{out})` or
          :math:`((|\\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\\mathcal{V}|, H * F_{out}),
          ((2, |\\mathcal{E}|), (|\\mathcal{E}|, H)))`
          or :math:`((|\\mathcal{V_t}|, H * F_{out}), ((2, |\\mathcal{E}|),
          (|\\mathcal{E}|, H)))` if bipartite
    """

    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int=1, concat: bool=True, negative_slope: float=0.2, dropout: float=0.0, add_self_loops: bool=True, edge_dim: Optional[int]=None, fill_value: Union[float, Tensor, str]='mean', bias: bool=True, share_weights: bool=False, residual: bool=False, **kwargs):
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights
        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels, bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels, bias=bias, weight_initializer='glorot')
        self.att = Parameter(torch.empty(1, heads, out_channels))
        if edge_dim is None:
            edge_dim = 1
            Warning("Warning: 'edge_dim' was not explicitly set. Assuming to be 1")
        self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
        total_out_channels = out_channels * (heads if concat else 1)
        if residual:
            self.res = Linear(in_channels if isinstance(in_channels, int) else in_channels[1], total_out_channels, bias=False, weight_initializer='glorot')
        else:
            self.register_parameter('res', None)
        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    @overload
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor=None, return_attention_weights: NoneType=None) -> Tensor:
        pass

    @overload
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Tensor, edge_attr: OptTensor=None, return_attention_weights: bool=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(self, x: Union[Tensor, PairTensor], edge_index: SparseTensor, edge_attr: OptTensor=None, return_attention_weights: bool=None) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor=None, return_attention_weights: Optional[bool]=None) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]], Tuple[Tensor, SparseTensor]]:
        """Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = (self.heads, self.out_channels)
        res: Optional[Tensor] = None
        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            if self.res is not None:
                res = self.res(x)
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = (x[0], x[1])
            assert x[0].dim() == 2
            if x_r is not None and self.res is not None:
                res = self.res(x_r)
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
        assert x_l is not None
        assert x_r is not None
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=self.fill_value, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError("The usage of 'edge_attr' and 'add_self_loops' simultaneously is currently not yet supported for 'edge_index' in a 'SparseTensor' form")
        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if res is not None:
            out = out + res
        if self.bias is not None:
            out = out + self.bias
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return (out, (adj, alpha))
                else:
                    return (out, (edge_index, alpha))
            elif isinstance(edge_index, SparseTensor):
                return (out, edge_index.set_value(alpha, layout='coo'))
        else:
            return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor, index: Tensor, ptr: OptTensor, dim_size: Optional[int]) -> Tensor:
        x = x_i + x_j
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})'

class GAT(BasicGNN):
    """The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs) -> MessagePassing:
        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)
        if getattr(self, '_is_conv_to_out', False):
            concat = False
        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of 'GATConv' (got '{out_channels}') is divisible by the number of heads (got '{heads}')")
        if concat:
            out_channels = out_channels // heads
        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat, dropout=self.dropout.p, **kwargs)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATWithResNorm(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=8, dropout=0.6, add_residual=True, add_norm=True):
        """Initialize GATWithResNorm."""
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.add_residual = add_residual
        self.add_norm = add_norm
        assert num_layers >= 1, 'num_layers must be at least 1'
        if isinstance(heads, int):
            heads = [heads] * (num_layers - 1) + [1]
        else:
            assert len(heads) == num_layers, 'heads must have num_layers entries'
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if add_norm else None
        self.residual_projections = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * heads[i - 1]
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            head = heads[i]
            concat = True if i < num_layers - 1 else False
            self.convs.append(GATConv(in_dim, out_dim, heads=head, concat=concat, dropout=dropout))
            if add_residual:
                residual_in_dim = in_dim
                current_out_dim = out_dim * head if concat else out_dim
                if residual_in_dim != current_out_dim:
                    self.residual_projections.append(nn.Linear(residual_in_dim, current_out_dim))
                else:
                    self.residual_projections.append(nn.Identity())
            if add_norm and i < num_layers - 1:
                norm_dim = out_dim * head if concat else out_dim
                self.norms.append(nn.LayerNorm(norm_dim))

    def forward(self, x, edge_index, edge_attr=None):
        z = x
        for i in range(self.num_layers):
            conv = self.convs[i]
            residual = z
            z = conv(z, edge_index.to(dtype=torch.int64), edge_attr=edge_attr)
            if self.add_residual and i < self.num_layers - 1:
                residual_transformed = self.residual_projections[i](residual)
                if residual_transformed.shape == z.shape:
                    z = z + residual_transformed
                else:
                    raise ValueError(f'Residual dimensions do not match at layer {i}')
            if i < self.num_layers - 1:
                if self.add_norm:
                    norm = self.norms[i]
                    z = norm(z)
                z = F.elu(z)
                z = F.dropout(z, p=self.dropout, training=self.training)
        return z


class TwoScaleGATInterp(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, v2, heads, feature_up=True, feature_down=False, dist_thres=torch.inf, topk=None, density_correction='uniform', learn_kernel=None, twod=False, residual=False, norm=False):
        super().__init__()
        self.dist_thres = dist_thres
        self.feature_up = feature_up
        self.feature_down = feature_down
        self.topk = topk
        self.density_correction = density_correction
        self.learn_kernel = learn_kernel
        if not residual and (not norm):
            self.gat1 = GAT(in_channels, hidden_channels, num_layers, hidden_channels, v2=v2, heads=heads)
            self.gat2 = GAT(in_channels, hidden_channels, num_layers, hidden_channels, v2=v2, heads=heads)
            self.gat3 = GAT(hidden_channels * 2 if feature_up else hidden_channels, hidden_channels, num_layers, out_channels, v2=v2, heads=heads)
            self.gat4 = GAT(hidden_channels * 2 if feature_down else hidden_channels, hidden_channels, num_layers, out_channels, v2=v2, heads=heads)
        else:
            self.gat1 = GATWithResNorm(in_channels, hidden_channels, hidden_channels, num_layers=num_layers, heads=heads, dropout=0.0, add_residual=residual, add_norm=norm)
            self.gat2 = GATWithResNorm(in_channels, hidden_channels, hidden_channels, num_layers=num_layers, heads=heads, dropout=0.0, add_residual=residual, add_norm=norm)
            self.gat3 = GATWithResNorm(hidden_channels * 2 if feature_up else hidden_channels, hidden_channels, out_channels, num_layers=num_layers, heads=heads, dropout=0.0, add_residual=residual, add_norm=norm)
            self.gat4 = GATWithResNorm(hidden_channels * 2 if feature_down else hidden_channels, hidden_channels, out_channels, num_layers=num_layers, heads=heads, dropout=0.0, add_residual=residual, add_norm=norm)
        if self.learn_kernel is not None:
            interp_kernel_lst = []
            linear_n_channels = 3 if twod else 4
            for _ in range(learn_kernel - 1):
                interp_kernel_lst.append(nn.Linear(linear_n_channels, linear_n_channels))
                interp_kernel_lst.append(nn.ReLU())
            interp_kernel_lst.append(nn.Linear(linear_n_channels, 1))
            self.interp_kernel = nn.Sequential(*interp_kernel_lst)
            for p in self.interp_kernel.parameters():
                nn.init.uniform_(p, 0.8, 1.2)

    def forward(self, feature1, edge_index1, edge_attr1, pos1, feature2, edge_index2, edge_attr2, pos2):
        device_this = feature1.device
        dtype_this = feature1.dtype
        edge_index1 = edge_index1.to(device_this)
        edge_index2 = edge_index2.to(device_this)
        edge_attr1 = edge_attr1.to(device_this, dtype=dtype_this) if edge_attr1 is not None else None
        edge_attr2 = edge_attr2.to(device_this, dtype=dtype_this) if edge_attr2 is not None else None
        if pos1 is not None:
            pos1 = pos1.to(device_this, dtype=dtype_this)
        if pos2 is not None:
            pos2 = pos2.to(device_this, dtype=dtype_this)
        mid1 = self.gat1(feature1, edge_index1, edge_attr1)
        mid2 = self.gat2(feature2, edge_index2, edge_attr2)
        if self.feature_up:
            interp1 = interpolate_from_graphs(pos2, mid2, pos1, dist_thres=self.dist_thres, topk=self.topk, density_correction=self.density_correction, interp_kernel=self.interp_kernel if self.learn_kernel is not None else None)
            mid10 = torch.clone(mid1)
            mid1 = torch.cat([mid1, interp1], dim=-1)
        if self.feature_down:
            mid1_this = mid10 if self.feature_up else mid1
            interp2 = interpolate_from_graphs(pos1, mid1_this, pos2, dist_thres=self.dist_thres, topk=self.topk, density_correction=self.density_correction, interp_kernel=self.interp_kernel if self.learn_kernel is not None else None)
            mid2 = torch.cat([mid2, interp2], dim=-1)
        out1 = self.gat3(mid1, edge_index1, edge_attr1)
        out2 = self.gat4(mid2, edge_index2, edge_attr2)
        return (out1, out2)


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple




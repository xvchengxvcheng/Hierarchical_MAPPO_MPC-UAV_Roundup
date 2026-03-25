import torch.nn as nn
from .util import init, get_clones

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU, hidden_sizes=None):
        super(MLPLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        # 若提供了自定义层维度，优先使用
        if hidden_sizes is not None and len(hidden_sizes) > 0:
            dims = [input_dim] + list(hidden_sizes)
            layers = []
            for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                layers.append(init_(nn.Linear(in_dim, out_dim)))
                layers.append(active_func)
                layers.append(nn.LayerNorm(out_dim))
            self.net = nn.Sequential(*layers)
            self.output_dim = dims[-1]
            self._layer_N = len(hidden_sizes)
        else:
            self._layer_N = layer_N
            self.fc1 = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
            self.fc_h = nn.Sequential(init_(
                nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
            self.fc2 = get_clones(self.fc_h, self._layer_N)
            self.output_dim = hidden_size

    def forward(self, x):
        if hasattr(self, "net"):
            return self.net(x)

        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N

        custom_hidden_sizes = getattr(args, "mlp_hidden_sizes", None)
        if custom_hidden_sizes is not None and len(custom_hidden_sizes) > 0:
            self.hidden_size = custom_hidden_sizes[-1]
        else:
            self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(
            obs_dim,
            self.hidden_size,
            self._layer_N,
            self._use_orthogonal,
            self._use_ReLU,
            hidden_sizes=custom_hidden_sizes,
        )

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x
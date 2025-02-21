import argparse
from torch import nn
import torch
import torch.nn.functional as F


class SkipConnection(nn.Module):
    """Basic skip connection module with dimension matching"""

    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.activate = activation()
        self.fc = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, inputs):
        x, new_x = inputs
        return self.activate(self.fc(x) + new_x)


class GatedSkipConnection(nn.Module):
    """Gated skip connection with learnable gate coefficients"""

    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.gate_x = nn.Linear(output_dim, output_dim)
        self.gate_newx = nn.Linear(output_dim, output_dim)

    def forward(self, inputs):
        x, new_x = inputs
        x = self.fc(x)
        gate = torch.sigmoid(self.gate_x(x) + self.gate_newx(new_x))
        return gate * new_x + (1 - gate) * x


class GraphConv(nn.Module):
    """Graph convolutional layer with optional skip connections"""

    def __init__(self, input_dim, hidden_dim, activation, connection_type):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

        if connection_type == 'sc':
            self.connection = SkipConnection(input_dim, hidden_dim, activation)
        elif connection_type == 'gsc':
            self.connection = GatedSkipConnection(input_dim, hidden_dim, activation)
        else:
            self.connection = None

        self.activation = activation()

    def forward(self, inputs):
        x, adj = inputs
        x_new = self.fc(torch.bmm(adj, x))

        if self.connection:
            x = self.connection((x, x_new))
        else:
            x = self.activation(x_new)

        return (x, adj)


class Readout(nn.Module):
    """Graph feature aggregation module"""

    def __init__(self, input_dim, hidden_dim, activation, dropout_rate):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activate = activation()

    def forward(self, x):
        x = self.dropout(self.fc(x))
        x = torch.sum(x, dim=1)
        return self.dropout(self.activate(self.bn(x)))


class GCN(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args

        # 统一使用与测试代码一致的层命名
        self.conv_layers = self._build_conv_layers(input_dim)

        # 其他层保持与测试代码一致的命名
        self.readout = Readout(args.hidden_size, args.hidden_size2, nn.ReLU, args.dp_rate)
        self.fc1 = nn.Linear(args.hidden_size2, args.hidden_size2)
        self.fc2 = nn.Linear(args.hidden_size2, args.hidden_size2)
        self.predictor = nn.Linear(args.hidden_size2, 1)
        self.bn = nn.BatchNorm1d(args.hidden_size2)
        self.dropout = nn.Dropout(args.dp_rate)

    def _build_conv_layers(self, input_dim):
        """构建与测试代码一致的卷积层结构"""
        layers = []
        # 第一层
        layers.append(
            GraphConv(input_dim, self.args.hidden_size, nn.ReLU, self.args.using_sc)
        )
        # 后续层
        for _ in range(self.args.num_layers - 1):
            layers.append(
                GraphConv(self.args.hidden_size, self.args.hidden_size, nn.ReLU, self.args.using_sc)
            )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        # 保持与测试代码一致的forward逻辑
        x, adj = inputs
        for conv in self.conv_layers:
            x, adj = conv((x, adj))
        x = self.readout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.bn(self.fc2(x))
        return torch.sigmoid(self.predictor(x))


def get_args():
    parser = argparse.ArgumentParser(description='GCN Training')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-steps', nargs='+', type=int, default=[30],
                        help='Epoch steps for learning rate decay')
    parser.add_argument('-dp_rate', type=float, default=0.3, help='dropout rate')


    # 模型参数
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of GCN layers')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--hidden-size2', type=int, default=256,
                        help='Second hidden dimension')
    parser.add_argument('--using-sc', type=str, default='sc',
                        choices=['sc', 'gsc', 'no'],
                        help='Skip connection type')

    # 系统参数
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Enable CUDA acceleration')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log interval in batches')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

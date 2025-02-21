#!/usr/bin/env python3
import sys
import warnings
import torch
from typing import List, Tuple
from model import GCN, get_args

warnings.filterwarnings("ignore")


def load_data(file_path: str) -> Tuple[List[str], List[float], List[str]]:
    """Load and process input data"""
    from utils import read_csv  # 延迟导入减少依赖
    return read_csv(file_path, mode='test')


def prepare_device(use_cuda: bool) -> torch.device:
    """Configure compute device"""
    return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


def load_model(model_path):
    """直接加载兼容的检查点"""
    checkpoint = torch.load(model_path, map_location='cpu')
    args = get_args()
    # 从检查点中加载配置参数并覆盖默认参数
    config = checkpoint['config']
    args.num_layers = config['num_layers']
    args.hidden_size = config['hidden_size']
    args.hidden_size2 = config['hidden_size2']
    args.using_sc = config['using_sc']
    args.dp_rate = config['dp_rate']
    # 初始化模型时使用检查点的input_dim
    model = GCN(args, input_dim=config['input_dim'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


def write_results(results: List[Tuple[float, str]], output_path: str) -> None:
    """Write formatted results to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for score, cid in results:
            f.write(f"{score:.4f}\t{cid}\n")


def main():
    # 配置参数并确保与训练时一致
    args = get_args()
    # 强制覆盖可能被命令行修改的关键参数（可选）
    # args.num_layers = 6
    # args.hidden_size = 64
    # args.hidden_size2 = 256
    # args.using_sc = 'sc'
    # args.dp_rate = 0.3

    device = prepare_device(args.cuda)

    # 加载数据
    data_path = 'tcm-screen.csv'
    smiles, labels, cids = load_data(data_path)

    # 转换图数据
    from utils import convert_to_graph  # 延迟导入
    X, A = convert_to_graph(smiles)
    X_tensor = torch.tensor(X).float().to(device)
    A_tensor = torch.tensor(A).float().to(device)

    # 初始化模型
    # 修正参数传递错误，只传递model_path
    model = load_model('best_model_best.pth')  # 确保路径正确
    model = model.to(device).eval()

    # 执行预测
    with torch.no_grad():
        # 移除多余的sigmoid，因为模型输出已经是概率值
        predictions = model((X_tensor, A_tensor)).cpu().numpy()

    # 格式化输出
    results = list(zip(predictions.flatten().tolist(), cids))
    write_results(results, 'predictions.tsv')


if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from typing import List, Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore")

# 原子特征常量
ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
    'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
    'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
    'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb'
]
MAX_ATOMS = 80
FEATURE_DIM = 58


def read_csv(file_path: str, mode: str) -> Tuple[List[str], List[Union[int, float]], List[str]]:
    """读取并预处理CSV数据

    Args:
        file_path: 数据文件路径
        mode: 数据模式 ('train' 或 'test')

    Returns:
        valid_smiles: 有效的SMILES列表
        labels: 对应的标签列表
        split_info: 分割信息（训练时为标签，测试时为CID）
    """
    df = pd.read_csv(file_path)
    smiles = df['SMILES'].str.strip().tolist()
    labels = df['label'].values.tolist()
    split_col = 'label' if mode == 'train' else 'CID'
    split_info = df[split_col].values.tolist()

    valid_smiles, valid_labels, valid_split = [], [], []

    for smi, label, spl in zip(smiles, labels, split_info):
        mol = Chem.MolFromSmiles(smi)
        if not mol or not is_valid_molecule(mol):
            continue

        valid_smiles.append(smi)
        valid_labels.append(label)
        valid_split.append(spl)

    print(f"Loaded {len(valid_smiles)} valid molecules")
    return valid_smiles, valid_labels, valid_split


def is_valid_molecule(mol: Chem.Mol) -> bool:
    """验证分子是否符合要求"""
    for atom in mol.GetAtoms():
        if atom.GetDegree() > 5:
            return False
    return mol.GetNumAtoms() <= MAX_ATOMS


def convert_to_graph(smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """将SMILES列表转换为图结构表示

    Args:
        smiles_list: SMILES字符串列表

    Returns:
        features: 原子特征矩阵 [batch_size, MAX_ATOMS, FEATURE_DIM]
        adj_matrices: 邻接矩阵 [batch_size, MAX_ATOMS, MAX_ATOMS]
    """
    features = []
    adj_matrices = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi.strip())
        if not mol:
            continue

        # 生成邻接矩阵
        adj = rdmolops.GetAdjacencyMatrix(mol)
        if adj.shape[0] > MAX_ATOMS:
            continue

        # 生成原子特征
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]

        # 填充到最大长度
        padded_features = np.zeros((MAX_ATOMS, FEATURE_DIM))
        padded_features[:len(atom_features)] = atom_features

        padded_adj = np.zeros((MAX_ATOMS, MAX_ATOMS))
        padded_adj[:adj.shape[0], :adj.shape[1]] = adj + np.eye(adj.shape[0])

        features.append(padded_features)
        adj_matrices.append(padded_adj)

    return np.array(features), np.array(adj_matrices)


def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """生成原子特征向量"""
    return np.concatenate([
        one_hot_encode(atom.GetSymbol(), ATOM_SYMBOLS),
        one_hot_encode(atom.GetDegree(), [0, 1, 2, 3, 4, 5]),
        one_hot_encode(atom.GetTotalNumHs(), [0, 1, 2, 3, 4], unknown_handle=True),
        one_hot_encode(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5], unknown_handle=True),
        [float(atom.GetIsAromatic())]
    ])


def one_hot_encode(
        value: Union[str, int],
        allowed: List,
        unknown_handle: bool = False
) -> np.ndarray:
    """通用one-hot编码函数

    Args:
        value: 需要编码的值
        allowed: 允许的值列表
        unknown_handle: 是否处理未知值（使用最后一个位置）

    Returns:
        one-hot编码向量
    """
    if unknown_handle and (value not in allowed):
        value = allowed[-1]
    return np.array([int(value == x) for x in allowed])


def create_mini_batch(
        features: np.ndarray,
        labels: np.ndarray,
        start_idx: int,
        batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, int]:
    """创建mini-batch数据

    Args:
        features: 特征矩阵
        labels: 标签数组
        start_idx: 起始索引
        batch_size: batch大小

    Returns:
        batch_features: 批特征
        batch_labels: 批标签
        new_start_idx: 新的起始索引
    """
    data_size = len(labels)

    if start_idx + batch_size <= data_size:
        end_idx = start_idx + batch_size
        return features[start_idx:end_idx], labels[start_idx:end_idx], end_idx

    # 处理循环情况
    wrap_size = (start_idx + batch_size) % data_size
    batch_features = np.concatenate([features[start_idx:], features[:wrap_size]], axis=0)
    batch_labels = np.concatenate([labels[start_idx:], labels[:wrap_size]], axis=0)
    return batch_features, batch_labels, wrap_size
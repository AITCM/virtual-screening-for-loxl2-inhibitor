
# LOXL2 Inhibitor Prediction Model

## 📌 项目概述
这是一个基于图卷积神经网络（GCN）的LOXL2抑制剂活性预测模型，实现了以下核心功能：
- **实时单样本预测**：通过 Gradio 界面上传化合物 SMILES 文件进行毒性概率预测
- **批量预测**：使用 `testing.py` 脚本处理多组测试数据
- **环境一致性**：严格复现训练环境的模型结构和超参数配置

## 🛠️ 环境准备

### 硬件要求
- GPU 支持（推荐 NVIDIA CUDA 11.2+）
- 至少 8GB 内存

### 软件依赖
```bash
# 核心依赖
python=3.6
conda env create -f environment.yml
conda activate loxl2_env
pip install  torch==1.8.1+cu102 torchvision==0.9.1+cu102 \
    torchtext==0.9.1 torch-geometric==2.0.3 dgl-cu111 numpy==1.19.5 \
    pandas==1.1.5 scikit-learn==0.24.2

# 开发调试依赖
pip install argparse pytest
```

> ⚠️ 注意：需额外安装 [RDKit](https://www.rdkit.org/) 处理分子结构文件

## ⚙️ 快速上手

批量预测脚本
```bash
python test.py --data_path tcm-screen.csv --output predictions.tsv
```
#### 参数说明
| 参数            | 类型     | 默认值         | 说明                     |
|-----------------|----------|----------------|--------------------------|
| `data_path`     | str      | `tcm-screen.csv` | 输入数据文件路径          |
| `output`        | str      | `predictions.tsv` | 输出结果文件路径          |

## 🧩 模型技术细节

### 网络架构
- **图卷积层**：6 层 GCN 网络
- **节点维度**：58 维分子表征
- **激活函数**：ReLU + Sigmoid 输出层
- **正则化**：Dropout(0.3) + BatchNorm

### 训练参数
| 参数             | 值                  |
|------------------|---------------------|
| 学习率           | 0.001               |
| 批量大小         | 128                 |
| 训练轮数         | 50                  |
| Dropout 率       | 0.3                 |

## 📊 数据说明
- **输入文件格式**：CSV 文件需包含 `smiles` 和 `label` 两列
- **示例数据**：`tcm-screen.csv`

## 🛑 注意事项
1. **GPU 加速**：使用 `CUDA_VISIBLE_DEVICES=0` 指定 GPU 设备
2. **内存限制**：批量预测时建议设置 `max_norm=5.0` 防止显存溢出
3. **模型恢复**：若遇到 `KeyError: 'config'`，请检查 `best_model_best.pth` 是否完整
4. 模型权重可参考：链接：https://pan.baidu.com/s/1Zeer3ezQl3ncqzH1La-7dw 提取码：tnty 

## 📜 许可证
MIT License  


> 💡 提示：推荐优先使用conda 环境 environment.yml
> 也可执行 `pip install -r requirements.txt` 安装依赖

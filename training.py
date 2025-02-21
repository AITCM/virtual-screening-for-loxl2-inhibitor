#!/usr/bin/env python3
import sys
import logging
import numpy as np
import torch
from typing import Tuple, List
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from model import GCN, get_args
from utils import convert_to_graph

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])

class Trainer:
    """GCN模型训练器"""

    def __init__(self, args):
        self.args = args
        self.device = self._prepare_device()
        self.model, self.optimizer, self.criterion = self._setup_components()
        self.best_auc = 0.0

    def _prepare_device(self) -> torch.device:
        """配置计算设备"""
        return torch.device(
            "cuda" if self.args.cuda and torch.cuda.is_available() else "cpu"
        )

    def _setup_components(self) -> Tuple[GCN, torch.optim.Optimizer, torch.nn.Module]:
        """初始化模型组件"""
        model = GCN(self.args, input_dim=58).to(self.device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr
        )
        criterion = torch.nn.BCELoss()
        return model, optimizer, criterion

    def _adjust_learning_rate(self, epoch: int) -> float:
        """调整学习率"""
        lr = self.args.lr * (0.95 ** np.sum(epoch >= np.array(self.args.lr_steps)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _train_epoch(self, train_data: Tuple, epoch: int) -> float:
        """单epoch训练"""
        self.model.train()
        total_loss = 0.0
        smiles, labels = train_data
        num_batches = len(labels) // self.args.batch_size

        for batch_idx in range(num_batches):
            # 获取批次数据
            start = batch_idx * self.args.batch_size
            end = start + self.args.batch_size
            batch_smiles = smiles[start:end]
            batch_labels = labels[start:end]

            # 转换图数据
            X, A = convert_to_graph(batch_smiles)
            X_tensor = torch.tensor(X).float().to(self.device)
            A_tensor = torch.tensor(A).float().to(self.device)
            y_tensor = torch.tensor(batch_labels).float().unsqueeze(1).to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model((X_tensor, A_tensor))
            loss = self.criterion(outputs, y_tensor)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                max_norm=5.0
            )
            self.optimizer.step()

            # 记录指标
            total_loss += loss.item()
            if batch_idx % self.args.log_interval == 0:
                auc = roc_auc_score(y_tensor.cpu().detach().numpy(), outputs.detach().cpu().numpy())
                logging.info(
                    f"Epoch[{epoch}] Batch[{batch_idx}/{num_batches}] "
                    f"Loss: {loss.item():.4f} AUC: {auc:.4f}"
                )

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, val_data: Tuple) -> Tuple[float, float]:
        """模型验证"""
        self.model.eval()
        smiles, labels = val_data
        X, A = convert_to_graph(smiles)

        X_tensor = torch.tensor(X).float().to(self.device)
        A_tensor = torch.tensor(A).float().to(self.device)
        y_tensor = torch.tensor(labels).float().unsqueeze(1).to(self.device)

        outputs = self.model((X_tensor, A_tensor))
        loss = self.criterion(outputs, y_tensor)
        auc = roc_auc_score(y_tensor.cpu().numpy(), outputs.cpu().numpy())
        return loss.item(), auc

    def save_checkpoint(self, path: str, is_best: bool = False):
        """保存与测试代码兼容的检查点"""
        state = {
            'state_dict': self.model.state_dict(),
            'config': {
                'input_dim': 58,
                'hidden_size': self.args.hidden_size,
                'hidden_size2': self.args.hidden_size2,
                'num_layers': self.args.num_layers,
                'using_sc': self.args.using_sc,
                'dp_rate': self.args.dp_rate
            }
        }
        torch.save(state, path)
        if is_best:
            torch.save(state, path.replace('.pth', '_best.pth'))

    def train(self, train_data: Tuple, val_data: Tuple):
        """完整训练流程"""
        for epoch in range(1, self.args.epochs + 1):
            # 调整学习率
            current_lr = self._adjust_learning_rate(epoch)
            logging.info(f"Epoch {epoch} started, learning rate: {current_lr:.6f}")

            # 训练阶段
            train_loss = self._train_epoch(train_data, epoch)

            # 验证阶段
            val_loss, val_auc = self.evaluate(val_data)
            logging.info(
                f"Epoch[{epoch}] Validation - Loss: {val_loss:.4f} AUC: {val_auc:.4f}"
            )

            # 保存最佳模型
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.save_checkpoint("best_model.pth", is_best=True)
                logging.info(f"New best model saved with AUC: {val_auc:.4f}")


def load_data(path: str) -> Tuple:
    """加载并预处理数据"""
    from utils import read_csv  # 延迟导入
    smiles, labels, split = read_csv(path, mode='train')
    return train_test_split(
        smiles, labels,
        test_size=0.4,
        stratify=split,
        random_state=42
    )


def main():
    args = get_args()

    # 加载数据
    train_smiles, val_smiles, train_labels, val_labels = load_data("loxl_data.csv")
    logging.info(f"Training samples: {len(train_labels)}, Validation samples: {len(val_labels)}")

    # 初始化训练器
    trainer = Trainer(args)

    # 开始训练
    trainer.train(
        train_data=(train_smiles, train_labels),
        val_data=(val_smiles, val_labels)
    )


if __name__ == "__main__":
    main()
"""
회귀 모델 학습 트레이너
"""
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np


class RegressionTrainer:
    """
    회귀 모델 학습 트레이너
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        save_dir: str = './checkpoints',
        weight_mean: float = 0.0,
        weight_std: float = 1.0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.weight_mean = weight_mean
        self.weight_std = weight_std

        # 손실 함수 (MSE + Huber 조합)
        self.mse_criterion = nn.MSELoss()
        self.huber_criterion = nn.SmoothL1Loss()

        # 옵티마이저
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 학습 기록
        self.history = {
            'train_loss': [],
            'train_mae': [],
            'train_rmse': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_mae_real': [],  # 원래 스케일의 MAE (그램)
            'lr': []
        }
        self.best_val_mae = float('inf')

    def _denormalize(self, value):
        """정규화된 값을 원래 스케일로 변환"""
        return value * self.weight_std + self.weight_mean

    def train_epoch(self) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, weights in pbar:
            images = images.to(self.device)
            weights = weights.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Huber Loss (MSE보다 이상치에 강건함)
            loss = self.huber_criterion(outputs, weights)

            # Backward
            loss.backward()
            self.optimizer.step()

            # 통계
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_mae += torch.abs(outputs - weights).sum().item()
            total_mse += ((outputs - weights) ** 2).sum().item()
            total += batch_size

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mae': f"{total_mae / total:.4f}"
            })

        return {
            'loss': total_loss / total,
            'mae': total_mae / total,
            'rmse': np.sqrt(total_mse / total)
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        total_mae_real = 0.0  # 원래 스케일
        total = 0

        pbar = tqdm(self.val_loader, desc="Validation")
        for images, weights in pbar:
            images = images.to(self.device)
            weights = weights.to(self.device)

            outputs = self.model(images)
            loss = self.huber_criterion(outputs, weights)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_mae += torch.abs(outputs - weights).sum().item()
            total_mse += ((outputs - weights) ** 2).sum().item()

            # 원래 스케일로 MAE 계산
            outputs_real = self._denormalize(outputs.cpu().numpy())
            weights_real = self._denormalize(weights.cpu().numpy())
            total_mae_real += np.abs(outputs_real - weights_real).sum()

            total += batch_size

        rmse = np.sqrt(total_mse / total)
        rmse_real = rmse * self.weight_std

        print(f"\n  Real Scale Metrics:")
        print(f"    MAE: {total_mae_real / total:.2f}g")
        print(f"    RMSE: {rmse_real:.2f}g")

        return {
            'loss': total_loss / total,
            'mae': total_mae / total,
            'rmse': rmse,
            'mae_real': total_mae_real / total
        }

    def train(
        self,
        epochs: int = 50,
        early_stopping_patience: int = 10
    ):
        """전체 학습 루프"""
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"Starting Regression Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Weight normalization: mean={self.weight_mean:.2f}g, std={self.weight_std:.2f}g")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)

            # 학습
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])

            # 검증
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_mae_real'].append(val_metrics['mae_real'])

            # 학습률 기록
            current_lr = scheduler.get_last_lr()[0]
            self.history['lr'].append(current_lr)
            scheduler.step()

            print(f"\n  Train Loss: {train_metrics['loss']:.4f}, Train MAE: {train_metrics['mae']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # 베스트 모델 저장 (MAE 기준)
            if val_metrics['mae'] < self.best_val_mae:
                self.best_val_mae = val_metrics['mae']
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                print(f"  ★ New best model saved! (MAE: {val_metrics['mae_real']:.2f}g)")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # 주기적 체크포인트 저장
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_metrics)

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation MAE: {self._denormalize(self.best_val_mae) - self.weight_mean + self.best_val_mae * self.weight_std:.2f}g")
        print(f"{'='*60}")

        return self.history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_mae': self.best_val_mae,
            'metrics': metrics,
            'history': self.history,
            'weight_mean': self.weight_mean,
            'weight_std': self.weight_std
        }
        torch.save(checkpoint, self.save_dir / filename)

    def save_history(self, filename: str = 'training_history.json'):
        """학습 기록 저장"""
        import json
        history_serializable = {}
        for key, value in self.history.items():
            history_serializable[key] = [float(v) for v in value]
        history_serializable['weight_mean'] = float(self.weight_mean)
        history_serializable['weight_std'] = float(self.weight_std)

        with open(self.save_dir / filename, 'w') as f:
            json.dump(history_serializable, f, indent=2)

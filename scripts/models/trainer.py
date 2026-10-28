"""
EfficientNet 모델 학습 트레이너
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


class Trainer:
    """
    모델 학습 트레이너
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        save_dir: str = './checkpoints',
        class_names: Optional[list] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        # 손실 함수
        self.criterion = nn.CrossEntropyLoss()

        # 옵티마이저
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 학습 기록
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        self.best_val_acc = 0.0

    def train_epoch(self) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # 통계
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })

        return {
            'loss': total_loss / total,
            'acc': correct / total
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # 클래스별 정확도 계산용
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        pbar = tqdm(self.val_loader, desc="Validation")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 클래스별 통계
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1

        # 클래스별 정확도 출력
        print("\n  Class-wise Accuracy:")
        for i in range(self.num_classes):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                print(f"    {self.class_names[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

        return {
            'loss': total_loss / total,
            'acc': correct / total
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
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)

            # 학습
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])

            # 검증
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])

            # 학습률 기록
            current_lr = scheduler.get_last_lr()[0]
            self.history['lr'].append(current_lr)
            scheduler.step()

            print(f"\n  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']*100:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']*100:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")

            # 베스트 모델 저장
            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                print(f"  ★ New best model saved! (Acc: {self.best_val_acc*100:.2f}%)")
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
        print(f"Best Validation Accuracy: {self.best_val_acc*100:.2f}%")
        print(f"{'='*60}")

        return self.history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'metrics': metrics,
            'history': self.history
        }
        torch.save(checkpoint, self.save_dir / filename)

    def save_history(self, filename: str = 'training_history.json'):
        """학습 기록 저장"""
        import json
        with open(self.save_dir / filename, 'w') as f:
            json.dump(self.history, f, indent=2)

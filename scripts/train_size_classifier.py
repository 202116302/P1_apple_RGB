"""
사과 크기 분류 모델 학습 스크립트
클래스: L (Large), M (Medium), S (Small)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
from models.efficientnet_classifier import create_model
from models.qc_dataset import QCSizeDataset, get_qc_transforms
from models.trainer import Trainer
from torch.utils.data import DataLoader


def main():
    # ===== 설정 =====
    DATA_DIR = r"C:\code\P1_apple_RGB\data\qc_images"
    SAVE_DIR = r"C:\code\P1_apple_RGB\models\size_classifier"

    NUM_CLASSES = 3
    CLASS_NAMES = ['L (Large)', 'M (Medium)', 'S (Small)']

    # 하이퍼파라미터
    BATCH_SIZE = 32
    IMG_SIZE = 224
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"\n*** 사과 크기 분류 모델 학습 ***")
    print(f"데이터 경로: {DATA_DIR}")
    print(f"에폭: {EPOCHS}")

    # ===== 데이터 로드 (전처리 ON - White Balance + CLAHE) =====
    print("\n[1/3] Loading datasets with preprocessing...")

    train_transform = get_qc_transforms(IMG_SIZE, is_training=True, apply_preprocessing=True)
    val_transform = get_qc_transforms(IMG_SIZE, is_training=False, apply_preprocessing=True)

    train_dataset = QCSizeDataset(
        data_dir=DATA_DIR,
        split='train',
        transform=train_transform
    )

    val_dataset = QCSizeDataset(
        data_dir=DATA_DIR,
        split='val',
        transform=val_transform
    )

    if len(train_dataset) == 0:
        print("Error: No training data found!")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # ===== 모델 생성 =====
    print("\n[2/3] Creating model...")
    model = create_model(num_classes=NUM_CLASSES, pretrained=True, device=device)

    # ===== 학습 =====
    print("\n[3/3] Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=NUM_CLASSES,
        device=device,
        learning_rate=LEARNING_RATE,
        save_dir=SAVE_DIR,
        class_names=CLASS_NAMES
    )

    trainer.train(epochs=EPOCHS, early_stopping_patience=7)
    trainer.save_history()

    print(f"\nModel saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()

"""
사과 무게 예측 모델 학습 스크립트 (회귀)
무게: 그램 단위
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
from models.efficientnet_regressor import create_regressor
from models.qc_dataset import QCWeightDataset, get_qc_transforms
from models.regression_trainer import RegressionTrainer
from torch.utils.data import DataLoader


def main():
    # ===== 설정 =====
    DATA_DIR = r"C:\code\P1_apple_RGB\data\qc_images"
    SAVE_DIR = r"C:\code\P1_apple_RGB\models\weight_regressor"

    # 하이퍼파라미터
    BATCH_SIZE = 32
    IMG_SIZE = 224
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"\n*** 사과 무게 예측 모델 학습 ***")
    print(f"데이터 경로: {DATA_DIR}")
    print(f"에폭: {EPOCHS}")

    # ===== 데이터 로드 (전처리 ON - White Balance + CLAHE) =====
    print("\n[1/3] Loading datasets with preprocessing...")

    train_transform = get_qc_transforms(IMG_SIZE, is_training=True, apply_preprocessing=True)
    val_transform = get_qc_transforms(IMG_SIZE, is_training=False, apply_preprocessing=True)

    # 먼저 학습 데이터셋 생성하여 정규화 파라미터 계산
    train_dataset = QCWeightDataset(
        data_dir=DATA_DIR,
        split='train',
        transform=train_transform,
        normalize_weight=True
    )

    if len(train_dataset) == 0:
        print("Error: No training data found!")
        return

    # 검증 데이터셋은 학습 데이터의 정규화 파라미터 사용
    val_dataset = QCWeightDataset(
        data_dir=DATA_DIR,
        split='val',
        transform=val_transform,
        normalize_weight=True,
        weight_mean=train_dataset.weight_mean,
        weight_std=train_dataset.weight_std
    )

    print(f"\nWeight normalization parameters:")
    print(f"  Mean: {train_dataset.weight_mean:.2f}g")
    print(f"  Std: {train_dataset.weight_std:.2f}g")

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
    model = create_regressor(pretrained=True, device=device)

    # ===== 학습 =====
    print("\n[3/3] Starting training...")
    trainer = RegressionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        save_dir=SAVE_DIR,
        weight_mean=train_dataset.weight_mean,
        weight_std=train_dataset.weight_std
    )

    trainer.train(epochs=EPOCHS, early_stopping_patience=7)
    trainer.save_history()

    print(f"\nModel saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()

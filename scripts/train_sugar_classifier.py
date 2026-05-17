"""
숙도(당도 등급) 분류 모델 - 빠른 학습 버전
캐싱된 전처리 이미지 사용 + 20 에폭
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
from models.efficientnet_classifier import create_model
from models.dataset import CachedSugarDataset, get_transforms
from models.trainer import Trainer
from torch.utils.data import DataLoader


def main():
    # ===== 설정 =====
    DATA_DIR = r"C:\code\P1_apple_RGB\cached_data\sugar"  # 캐싱된 이미지
    LABEL_DIR = r"C:\code\P1_apple_RGB\data\sugar_data"  # 원본 라벨
    SAVE_DIR = r"C:\code\P1_apple_RGB\models\sugar_classifier_preprocessed"

    NUM_CLASSES = 3
    CLASS_NAMES = ['A등급', 'B등급', 'C등급']

    BATCH_SIZE = 32
    IMG_SIZE = 224
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"\n*** 숙도(당도) 분류 - 빠른 학습 ***")

    # ===== 데이터 로드 =====
    print("\n[1/3] Loading cached datasets...")

    train_transform = get_transforms(IMG_SIZE, is_training=True, apply_preprocessing=False)
    val_transform = get_transforms(IMG_SIZE, is_training=False, apply_preprocessing=False)

    train_dataset = CachedSugarDataset(
        data_dir=DATA_DIR,
        label_dir=LABEL_DIR,
        split='train',
        transform=train_transform,
        crop_to_bbox=True
    )

    val_dataset = CachedSugarDataset(
        data_dir=DATA_DIR,
        label_dir=LABEL_DIR,
        split='val',
        transform=val_transform,
        crop_to_bbox=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ===== 모델 =====
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

"""
과실 병 감염 분류 모델 - 빠른 학습 버전
캐싱된 전처리 이미지 사용 + 20 에폭
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
from models.efficientnet_classifier import create_model
from models.dataset import CachedDiseaseDataset, get_transforms
from models.trainer import Trainer
from torch.utils.data import DataLoader


def main():
    # ===== 설정 =====
    # 캐싱된 데이터 사용
    DATA_DIR = r"C:\code\P1_apple_RGB\cached_data\disease_fruit"
    LABEL_DIR = r"C:\code\P1_apple_RGB\data\disease_data"  # 라벨은 원본 경로
    SAVE_DIR = r"C:\code\P1_apple_RGB\models\disease_fruit_classifier_preprocessed"

    NUM_CLASSES = 2
    CLASS_NAMES = ['정상', '질병']
    AREA_FILTER = [1]

    # 하이퍼파라미터 (에폭 줄임)
    BATCH_SIZE = 32
    IMG_SIZE = 224
    EPOCHS = 20  # 50 → 20
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"\n*** 과실(Area=1) 병 감염 분류 - 빠른 학습 ***")
    print(f"캐싱된 데이터 사용: {DATA_DIR}")
    print(f"에폭: {EPOCHS}")

    # ===== 데이터 로드 (전처리 OFF - 이미 캐싱됨) =====
    print("\n[1/3] Loading cached datasets...")

    train_transform = get_transforms(IMG_SIZE, is_training=True, apply_preprocessing=False)
    val_transform = get_transforms(IMG_SIZE, is_training=False, apply_preprocessing=False)

    train_dataset = CachedDiseaseDataset(
        data_dir=DATA_DIR,
        split='train',
        transform=train_transform
    )

    val_dataset = CachedDiseaseDataset(
        data_dir=DATA_DIR,
        split='val',
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

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

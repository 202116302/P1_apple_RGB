"""
YOLO 모델 경량 학습 스크립트 (빠른 학습용)
10,000장 샘플링 + 30 에폭으로 빠르게 학습
"""
from ultralytics import YOLO
import torch
import os
import random
import shutil
from pathlib import Path


def sample_dataset(source_path: str, output_path: str, max_samples: int = 10000):
    """
    데이터셋에서 일부만 샘플링
    """
    source = Path(source_path)
    output = Path(output_path)

    # 출력 폴더 초기화
    if output.exists():
        shutil.rmtree(output)

    for split in ['train', 'val']:
        (output / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output / 'labels' / split).mkdir(parents=True, exist_ok=True)

        img_dir = source / 'images' / split
        label_dir = source / 'labels' / split

        if not img_dir.exists():
            continue

        # 이미지 파일 목록
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

        # 샘플링
        if split == 'train':
            sample_size = min(len(images), max_samples)
        else:
            sample_size = min(len(images), max_samples // 5)  # val은 20%

        sampled = random.sample(images, sample_size)

        print(f"[{split}] Sampling {sample_size}/{len(images)} images")

        for img_path in sampled:
            # 이미지 복사
            shutil.copy2(img_path, output / 'images' / split / img_path.name)

            # 라벨 복사
            label_path = label_dir / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, output / 'labels' / split / label_path.name)

    # data.yaml 생성
    yaml_content = f"""# YOLO Apple Segmentation Dataset (Sampled)
path: {output.absolute()}
train: images/train
val: images/val

names:
  0: apple

nc: 1
"""
    with open(output / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"\nSampled dataset saved to: {output}")
    return str(output / 'data.yaml')


def train_yolo_light(
    data_yaml_path: str,
    epochs: int = 30,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "models",
    name: str = "yolo_apple_seg"
):
    """
    YOLOv8 경량 학습
    """
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # YOLOv8n-seg 모델 로드
    model = YOLO("yolov8n-seg.pt")

    print(f"\n=== Light Training Configuration ===")
    print(f"Data: {data_yaml_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")

    # 학습
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        patience=10,
        save=True,
        cache=False,
        workers=4,
        # 증강 설정
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        verbose=True,
        seed=42,
    )

    print(f"\n=== Training Complete ===")
    print(f"Best model: {project}/{name}/weights/best.pt")

    return results


if __name__ == "__main__":
    # 경로 설정
    FULL_DATASET = r"C:\code\P1_apple_RGB\datasets\yolo_apple_seg"
    SAMPLED_DATASET = r"C:\code\P1_apple_RGB\datasets\yolo_apple_seg_sampled"
    PROJECT_DIR = r"C:\code\P1_apple_RGB\models"

    # 1. 데이터 샘플링
    print("=== Step 1: Sampling Dataset ===")
    data_yaml = sample_dataset(
        source_path=FULL_DATASET,
        output_path=SAMPLED_DATASET,
        max_samples=10000
    )

    # 2. 학습
    print("\n=== Step 2: Training YOLO ===")
    train_yolo_light(
        data_yaml_path=data_yaml,
        epochs=30,
        batch_size=16,
        img_size=640,
        device="0",
        project=PROJECT_DIR,
        name="yolo_apple_seg"
    )

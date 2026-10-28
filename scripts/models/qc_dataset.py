"""
QC 이미지 데이터셋 클래스
- 크기 분류 (L/M/S)
- 무게 예측 (회귀)
"""
import os
import json
from pathlib import Path
from typing import Tuple, List, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class QCSizeDataset(Dataset):
    """
    사과 크기 분류 데이터셋
    클래스: 0=L (Large), 1=M (Medium), 2=S (Small)
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',  # 'train' or 'val'
        transform: Optional[Callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []  # (img_path, label)

        self.size_to_label = {'L': 0, 'M': 1, 'S': 2}
        self.label_to_size = {0: 'L', 1: 'M', 2: 'S'}

        self._load_samples()

    def _load_samples(self):
        """데이터 샘플 로드"""
        if self.split == 'train':
            img_base = self.data_dir / "01.데이터" / "1.Training" / "원천데이터_230921_add"
        else:
            img_base = self.data_dir / "01.데이터" / "2.Validation" / "원천데이터_230921_add"

        for size_class in ['L', 'M', 'S']:
            folder_name = f"Apple_fuji_{size_class}"
            img_dir = img_base / folder_name

            if not img_dir.exists():
                print(f"Warning: {img_dir} does not exist")
                continue

            label = self.size_to_label[size_class]

            # 이미지 파일 수집
            for img_path in list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.PNG")) + list(img_dir.glob("*.JPG")):
                self.samples.append((str(img_path), label))

        print(f"[{self.split}] Loaded {len(self.samples)} samples")
        for size, label in self.size_to_label.items():
            count = sum(1 for _, l in self.samples if l == label)
            print(f"  - Size {size}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class QCDiameterDataset(Dataset):
    """
    사과 횡경(크기) 예측 데이터셋 (회귀)
    횡경: mm 단위 (정규화하여 사용)
    기존 분류(L/M/S)보다 회귀로 더 정확한 크기 예측
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        normalize: bool = True,
        diameter_mean: float = None,
        diameter_std: float = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.samples: List[Tuple[str, float]] = []  # (img_path, diameter_mm)

        self._load_samples()

        # 횡경 정규화 파라미터
        if normalize:
            if diameter_mean is not None and diameter_std is not None:
                self.diameter_mean = diameter_mean
                self.diameter_std = diameter_std
            else:
                diameters = [d for _, d in self.samples]
                self.diameter_mean = np.mean(diameters)
                self.diameter_std = np.std(diameters)
        else:
            self.diameter_mean = 0
            self.diameter_std = 1

    def _load_samples(self):
        """데이터 샘플 로드"""
        if self.split == 'train':
            img_base = self.data_dir / "01.데이터" / "1.Training" / "원천데이터_230921_add"
            label_base = self.data_dir / "01.데이터" / "1.Training" / "라벨링데이터_230921_add"
        else:
            img_base = self.data_dir / "01.데이터" / "2.Validation" / "원천데이터_230921_add"
            label_base = self.data_dir / "01.데이터" / "2.Validation" / "라벨링데이터_230921_add"

        for size_class in ['L', 'M', 'S']:
            folder_name = f"Apple_fuji_{size_class}"
            img_dir = img_base / folder_name
            label_dir = label_base / folder_name

            if not img_dir.exists() or not label_dir.exists():
                print(f"Warning: {img_dir} or {label_dir} does not exist")
                continue

            for json_path in label_dir.glob("*.json"):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 횡경 (width) 추출 - cm를 mm로 변환
                    width_cm = float(data.get('width', 0))
                    if width_cm <= 0:
                        continue
                    diameter_mm = width_cm * 10  # cm -> mm

                    # 이미지 파일 찾기
                    img_name = json_path.stem + ".png"
                    img_path = img_dir / img_name
                    if not img_path.exists():
                        img_name = json_path.stem + ".jpg"
                        img_path = img_dir / img_name

                    if img_path.exists():
                        self.samples.append((str(img_path), diameter_mm))

                except Exception as e:
                    continue

        print(f"[{self.split}] Loaded {len(self.samples)} samples with diameter labels")
        if self.samples:
            diameters = [d for _, d in self.samples]
            print(f"  - Diameter range: {min(diameters):.1f}mm ~ {max(diameters):.1f}mm")
            print(f"  - Diameter mean: {np.mean(diameters):.1f}mm, std: {np.std(diameters):.1f}mm")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, diameter = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.normalize:
            diameter = (diameter - self.diameter_mean) / (self.diameter_std + 1e-6)

        return image, torch.tensor(diameter, dtype=torch.float32)

    def denormalize(self, normalized_diameter):
        """정규화된 횡경을 원래 값으로 복원"""
        return normalized_diameter * self.diameter_std + self.diameter_mean

    @staticmethod
    def diameter_to_grade(diameter_mm: float) -> str:
        """횡경(mm)을 등급으로 변환"""
        if diameter_mm >= 85:  # 8.5cm 이상
            return 'L'
        elif diameter_mm >= 75:  # 7.5cm 이상
            return 'M'
        else:
            return 'S'


class QCWeightDataset(Dataset):
    """
    사과 무게 예측 데이터셋 (회귀)
    무게: 그램 단위 (정규화하여 사용)
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',  # 'train' or 'val'
        transform: Optional[Callable] = None,
        normalize_weight: bool = True,
        weight_mean: float = None,
        weight_std: float = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.normalize_weight = normalize_weight
        self.samples: List[Tuple[str, float]] = []  # (img_path, weight)

        self._load_samples()

        # 무게 정규화 파라미터
        if normalize_weight:
            if weight_mean is not None and weight_std is not None:
                self.weight_mean = weight_mean
                self.weight_std = weight_std
            else:
                weights = [w for _, w in self.samples]
                self.weight_mean = np.mean(weights)
                self.weight_std = np.std(weights)
        else:
            self.weight_mean = 0
            self.weight_std = 1

    def _load_samples(self):
        """데이터 샘플 로드"""
        if self.split == 'train':
            img_base = self.data_dir / "01.데이터" / "1.Training" / "원천데이터_230921_add"
            label_base = self.data_dir / "01.데이터" / "1.Training" / "라벨링데이터_230921_add"
        else:
            img_base = self.data_dir / "01.데이터" / "2.Validation" / "원천데이터_230921_add"
            label_base = self.data_dir / "01.데이터" / "2.Validation" / "라벨링데이터_230921_add"

        for size_class in ['L', 'M', 'S']:
            folder_name = f"Apple_fuji_{size_class}"
            img_dir = img_base / folder_name
            label_dir = label_base / folder_name

            if not img_dir.exists() or not label_dir.exists():
                print(f"Warning: {img_dir} or {label_dir} does not exist")
                continue

            # JSON 파일 순회
            for json_path in label_dir.glob("*.json"):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    weight = float(data.get('weight', 0))
                    if weight <= 0:
                        continue

                    # 이미지 파일 찾기
                    img_name = json_path.stem + ".png"
                    img_path = img_dir / img_name

                    if not img_path.exists():
                        # PNG가 없으면 jpg 시도
                        img_name = json_path.stem + ".jpg"
                        img_path = img_dir / img_name

                    if img_path.exists():
                        self.samples.append((str(img_path), weight))

                except Exception as e:
                    continue

        print(f"[{self.split}] Loaded {len(self.samples)} samples with weight labels")
        if self.samples:
            weights = [w for _, w in self.samples]
            print(f"  - Weight range: {min(weights):.1f}g ~ {max(weights):.1f}g")
            print(f"  - Weight mean: {np.mean(weights):.1f}g, std: {np.std(weights):.1f}g")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, weight = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 무게 정규화
        if self.normalize_weight:
            weight = (weight - self.weight_mean) / (self.weight_std + 1e-6)

        return image, torch.tensor(weight, dtype=torch.float32)

    def denormalize_weight(self, normalized_weight):
        """정규화된 무게를 원래 값으로 복원"""
        return normalized_weight * self.weight_std + self.weight_mean


def get_qc_transforms(img_size: int = 224, is_training: bool = True, apply_preprocessing: bool = True):
    """
    QC 데이터용 변환 파이프라인
    """
    from torchvision import transforms
    from .preprocessing import PreprocessTransform

    transform_list = []

    # 1. White Balance + CLAHE 전처리
    if apply_preprocessing:
        transform_list.append(PreprocessTransform(apply_wb=True, apply_clahe=True))

    # 2. 크기 조정 및 증강
    if is_training:
        transform_list.extend([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    else:
        transform_list.append(transforms.Resize((img_size, img_size)))

    # 3. 텐서 변환 및 정규화
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transforms.Compose(transform_list)


def create_size_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    apply_preprocessing: bool = True
):
    """
    크기 분류용 DataLoader 생성
    """
    train_transform = get_qc_transforms(img_size, is_training=True, apply_preprocessing=apply_preprocessing)
    val_transform = get_qc_transforms(img_size, is_training=False, apply_preprocessing=apply_preprocessing)

    train_dataset = QCSizeDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform
    )

    val_dataset = QCSizeDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def create_weight_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    apply_preprocessing: bool = True
):
    """
    무게 예측용 DataLoader 생성
    """
    train_transform = get_qc_transforms(img_size, is_training=True, apply_preprocessing=apply_preprocessing)
    val_transform = get_qc_transforms(img_size, is_training=False, apply_preprocessing=apply_preprocessing)

    # 먼저 학습 데이터셋 생성하여 정규화 파라미터 계산
    train_dataset = QCWeightDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform,
        normalize_weight=True
    )

    # 검증 데이터셋은 학습 데이터의 정규화 파라미터 사용
    val_dataset = QCWeightDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform,
        normalize_weight=True,
        weight_mean=train_dataset.weight_mean,
        weight_std=train_dataset.weight_std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.weight_mean, train_dataset.weight_std


def create_diameter_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    apply_preprocessing: bool = True
):
    """
    횡경(크기) 회귀용 DataLoader 생성
    """
    train_transform = get_qc_transforms(img_size, is_training=True, apply_preprocessing=apply_preprocessing)
    val_transform = get_qc_transforms(img_size, is_training=False, apply_preprocessing=apply_preprocessing)

    # 먼저 학습 데이터셋 생성하여 정규화 파라미터 계산
    train_dataset = QCDiameterDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform,
        normalize=True
    )

    # 검증 데이터셋은 학습 데이터의 정규화 파라미터 사용
    val_dataset = QCDiameterDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform,
        normalize=True,
        diameter_mean=train_dataset.diameter_mean,
        diameter_std=train_dataset.diameter_std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.diameter_mean, train_dataset.diameter_std

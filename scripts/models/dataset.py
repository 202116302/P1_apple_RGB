"""
이미지 분류용 데이터셋 클래스
"""
import os
import json
from pathlib import Path
from typing import Tuple, List, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class DiseaseDataset(Dataset):
    """
    과수화상병 데이터셋 (병 감염 분류)
    클래스: 0=정상, 1=질병
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',  # 'train' or 'val'
        transform: Optional[Callable] = None,
        area_filter: Optional[List[int]] = None,  # 필터링할 area 값 (예: [1] = 과실만, [3] = 잎만)
        crop_to_bbox: bool = False  # bbox 영역만 크롭할지 여부
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.area_filter = area_filter
        self.crop_to_bbox = crop_to_bbox
        self.samples: List[Tuple[str, str, int]] = []  # (img_path, json_path, label)

        self._load_samples()

    def _load_samples(self):
        """데이터 샘플 로드"""
        if self.split == 'train':
            img_base = self.data_dir / "Training" / "images"
            label_base = self.data_dir / "Training" / "label"
        else:
            img_base = self.data_dir / "Validation" / "images"
            label_base = self.data_dir / "Validation" / "label"

        # 정상 이미지 (클래스 0)
        normal_img_dirs = [d for d in img_base.iterdir() if '정상' in d.name and d.is_dir()]
        normal_label_dirs = [d for d in label_base.iterdir() if '정상' in d.name and d.is_dir()]

        for img_dir in normal_img_dirs:
            # 매칭되는 라벨 폴더 찾기
            label_dir = None
            for ld in normal_label_dirs:
                if ld.exists():
                    label_dir = ld
                    break

            if label_dir is None:
                continue

            for img_path in list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg")):
                json_path = label_dir / (img_path.name + ".json")
                if json_path.exists():
                    if self._check_area_filter(json_path):
                        self.samples.append((str(img_path), str(json_path), 0))

        # 질병 이미지 (클래스 1)
        disease_img_dirs = [d for d in img_base.iterdir() if '질병' in d.name and d.is_dir()]
        disease_label_dirs = [d for d in label_base.iterdir() if '질병' in d.name and d.is_dir()]

        for img_dir in disease_img_dirs:
            label_dir = None
            for ld in disease_label_dirs:
                if ld.exists():
                    label_dir = ld
                    break

            if label_dir is None:
                continue

            for img_path in list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg")):
                json_path = label_dir / (img_path.name + ".json")
                if json_path.exists():
                    if self._check_area_filter(json_path):
                        self.samples.append((str(img_path), str(json_path), 1))

        print(f"[{self.split}] Loaded {len(self.samples)} samples (area_filter={self.area_filter})")
        print(f"  - Normal: {sum(1 for _, _, l in self.samples if l == 0)}")
        print(f"  - Disease: {sum(1 for _, _, l in self.samples if l == 1)}")

    def _check_area_filter(self, json_path) -> bool:
        """area 필터 체크"""
        if self.area_filter is None:
            return True
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            area = data['annotations'].get('area')
            return area in self.area_filter
        except:
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]

        # 이미지 로드
        image = Image.open(img_path).convert('RGB')

        # bbox 크롭 (선택적)
        if self.crop_to_bbox:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                points = data['annotations'].get('points', [])
                if points:
                    p = points[0]
                    x1, y1, x2, y2 = p['xtl'], p['ytl'], p['xbr'], p['ybr']
                    image = image.crop((x1, y1, x2, y2))
            except:
                pass  # 크롭 실패 시 원본 사용

        if self.transform:
            image = self.transform(image)

        return image, label


class SugarGradeDataset(Dataset):
    """
    당도 등급 데이터셋 (숙도 분류)
    클래스: 0=A등급, 1=B등급, 2=C등급
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        crop_to_bbox: bool = True  # bbox 영역만 크롭할지 여부
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.crop_to_bbox = crop_to_bbox
        self.samples: List[Tuple[str, str, int]] = []  # (img_path, json_path, label)

        self.grade_to_label = {'A': 0, 'B': 1, 'C': 2}
        self._load_samples()

    def _load_samples(self):
        """데이터 샘플 로드"""
        if self.split == 'train':
            img_base = self.data_dir / "01.데이터" / "1.Training" / "원천데이터"
            label_base = self.data_dir / "01.데이터" / "1.Training" / "라벨링데이터_230525_add"
            subdirs = ["후지1"]
        else:
            img_base = self.data_dir / "01.데이터" / "2.Validation" / "원천데이터"
            label_base = self.data_dir / "01.데이터" / "2.Validation" / "라벨링데이터_230525_add"
            subdirs = ["후지"]

        for subdir in subdirs:
            for grade in ['당도A등급', '당도B등급', '당도C등급']:
                grade_letter = grade[2]  # 'A', 'B', or 'C'
                label = self.grade_to_label.get(grade_letter)
                if label is None:
                    continue

                img_dir = img_base / subdir / grade
                label_dir = label_base / subdir / grade

                if not img_dir.exists() or not label_dir.exists():
                    continue

                for json_path in label_dir.glob("*.json"):
                    img_name = json_path.stem + ".jpg"
                    img_path = img_dir / img_name

                    if img_path.exists():
                        self.samples.append((str(img_path), str(json_path), label))

        print(f"[{self.split}] Loaded {len(self.samples)} samples")
        for grade, label in self.grade_to_label.items():
            count = sum(1 for _, _, l in self.samples if l == label)
            print(f"  - Grade {grade}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]

        # 이미지 로드
        image = Image.open(img_path).convert('RGB')

        # bbox 크롭 (선택적)
        if self.crop_to_bbox:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                bbox = data['annotations']['bbox']
                # bbox: [x, y, width, height]
                x, y, w, h = bbox
                image = image.crop((x, y, x + w, y + h))
            except Exception:
                pass  # 크롭 실패 시 원본 사용

        if self.transform:
            image = self.transform(image)

        return image, label


class CachedDiseaseDataset(Dataset):
    """
    캐싱된 과수화상병 데이터셋 (전처리 완료된 이미지)
    폴더 구조로 라벨 추론: 정상=0, 질병=1
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        self._load_samples()

    def _load_samples(self):
        """폴더명으로 라벨 추론"""
        if self.split == 'train':
            base_path = self.data_dir / "Training" / "images"
        else:
            base_path = self.data_dir / "Validation" / "images"

        if not base_path.exists():
            print(f"Warning: {base_path} does not exist")
            return

        for img_dir in base_path.iterdir():
            if not img_dir.is_dir():
                continue

            # 폴더명으로 라벨 결정
            if '정상' in img_dir.name:
                label = 0
            elif '질병' in img_dir.name:
                label = 1
            else:
                continue

            for img_path in list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg")):
                self.samples.append((str(img_path), label))

        print(f"[{self.split}] Loaded {len(self.samples)} cached samples")
        print(f"  - Normal: {sum(1 for _, l in self.samples if l == 0)}")
        print(f"  - Disease: {sum(1 for _, l in self.samples if l == 1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class CachedSugarDataset(Dataset):
    """
    캐싱된 당도 데이터셋 (전처리 완료된 이미지)
    폴더명으로 라벨 추론: A=0, B=1, C=2
    """
    def __init__(
        self,
        data_dir: str,
        label_dir: str,  # 원본 라벨 경로 (bbox 정보용)
        split: str = 'train',
        transform: Optional[Callable] = None,
        crop_to_bbox: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.split = split
        self.transform = transform
        self.crop_to_bbox = crop_to_bbox
        self.samples: List[Tuple[str, str, int]] = []

        self.grade_to_label = {'A': 0, 'B': 1, 'C': 2}
        self._load_samples()

    def _load_samples(self):
        if self.split == 'train':
            img_base = self.data_dir / "01.데이터" / "1.Training" / "원천데이터"
            label_base = self.label_dir / "01.데이터" / "1.Training" / "라벨링데이터_230525_add"
            subdirs = ["후지1"]
        else:
            img_base = self.data_dir / "01.데이터" / "2.Validation" / "원천데이터"
            label_base = self.label_dir / "01.데이터" / "2.Validation" / "라벨링데이터_230525_add"
            subdirs = ["후지"]

        for subdir in subdirs:
            for grade in ['당도A등급', '당도B등급', '당도C등급']:
                grade_letter = grade[2]
                label = self.grade_to_label.get(grade_letter)
                if label is None:
                    continue

                img_dir = img_base / subdir / grade
                lbl_dir = label_base / subdir / grade

                if not img_dir.exists():
                    continue

                for img_path in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG")):
                    json_path = lbl_dir / (img_path.stem + ".json") if lbl_dir.exists() else None
                    self.samples.append((str(img_path), str(json_path) if json_path and json_path.exists() else "", label))

        print(f"[{self.split}] Loaded {len(self.samples)} cached samples")
        for grade, lbl in self.grade_to_label.items():
            count = sum(1 for _, _, l in self.samples if l == lbl)
            print(f"  - Grade {grade}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.crop_to_bbox and json_path:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                bbox = data['annotations']['bbox']
                x, y, w, h = bbox
                image = image.crop((x, y, x + w, y + h))
            except:
                pass

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(img_size: int = 224, is_training: bool = True, apply_preprocessing: bool = True):
    """
    데이터 변환 파이프라인
    White Balance + CLAHE 전처리 포함

    Args:
        img_size: 출력 이미지 크기
        is_training: 학습용 여부 (True면 증강 적용)
        apply_preprocessing: White Balance + CLAHE 적용 여부
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


def create_dataloaders(
    dataset_class,
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    **dataset_kwargs
):
    """
    DataLoader 생성 헬퍼 함수
    """
    train_transform = get_transforms(img_size, is_training=True)
    val_transform = get_transforms(img_size, is_training=False)

    train_dataset = dataset_class(
        data_dir=data_dir,
        split='train',
        transform=train_transform,
        **dataset_kwargs
    )

    val_dataset = dataset_class(
        data_dir=data_dir,
        split='val',
        transform=val_transform,
        **dataset_kwargs
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

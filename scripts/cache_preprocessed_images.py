"""
이미지 전처리 캐싱 스크립트
White Balance + CLAHE 적용된 이미지를 미리 저장
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import cv2
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.preprocessing import preprocess_image


def process_and_save(args):
    """단일 이미지 처리 및 저장 (한글 경로 지원)"""
    src_path, dst_path = args
    try:
        # 한글 경로 지원: numpy로 읽기
        img_array = np.fromfile(str(src_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return False

        # 전처리 적용
        processed = preprocess_image(img, apply_wb=True, apply_clahe_flag=True)

        # 저장 (한글 경로 지원)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        _, ext = os.path.splitext(str(dst_path))
        result, encoded = cv2.imencode(ext, processed)
        if result:
            encoded.tofile(str(dst_path))
            return True
        return False
    except Exception as e:
        return False


def cache_disease_dataset(source_dir: str, cache_dir: str, area_filter: list = None):
    """
    과수화상병 데이터셋 전처리 캐싱
    """
    source = Path(source_dir)
    cache = Path(cache_dir)

    tasks = []

    for split in ['Training', 'Validation']:
        img_base = source / split / "images"
        label_base = source / split / "label"

        if not img_base.exists():
            continue

        for img_dir in img_base.iterdir():
            if not img_dir.is_dir():
                continue

            # 정상 또는 질병 폴더만
            if '정상' not in img_dir.name and '질병' not in img_dir.name:
                continue

            # 매칭되는 라벨 폴더 찾기
            label_dir = None
            for ld in label_base.iterdir():
                if '정상' in img_dir.name and '정상' in ld.name:
                    label_dir = ld
                    break
                elif '질병' in img_dir.name and '질병' in ld.name:
                    label_dir = ld
                    break

            for img_path in list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg")):
                # Area 필터링
                if area_filter and label_dir:
                    json_path = label_dir / (img_path.name + ".json")
                    if json_path.exists():
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            area = data['annotations'].get('area')
                            if area not in area_filter:
                                continue
                        except:
                            continue
                    else:
                        continue

                # 캐시 경로 생성
                rel_path = img_path.relative_to(source)
                dst_path = cache / rel_path

                tasks.append((img_path, dst_path))

    print(f"Processing {len(tasks)} images...")

    # 병렬 처리
    success = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_and_save, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks)):
            if future.result():
                success += 1

    print(f"Cached {success}/{len(tasks)} images to {cache}")
    return success


def cache_sugar_dataset(source_dir: str, cache_dir: str, max_per_grade: int = None):
    """
    당도 데이터셋 전처리 캐싱

    Args:
        source_dir: 원본 데이터 경로
        cache_dir: 캐시 저장 경로
        max_per_grade: 등급별 최대 이미지 수 (None이면 전체)
    """
    import random
    source = Path(source_dir)
    cache = Path(cache_dir)

    tasks = []

    # Training
    train_img_base = source / "01.데이터" / "1.Training" / "원천데이터"
    for subdir in train_img_base.iterdir():
        if not subdir.is_dir():
            continue
        for grade_dir in subdir.iterdir():
            if not grade_dir.is_dir():
                continue

            images = list(grade_dir.glob("*.jpg")) + list(grade_dir.glob("*.JPG"))

            # 샘플링
            if max_per_grade and len(images) > max_per_grade:
                images = random.sample(images, max_per_grade)
                print(f"  [Train] {grade_dir.name}: {max_per_grade}/{len(list(grade_dir.glob('*.jpg')) + list(grade_dir.glob('*.JPG')))} sampled")

            for img_path in images:
                rel_path = img_path.relative_to(source)
                dst_path = cache / rel_path
                tasks.append((img_path, dst_path))

    # Validation
    val_img_base = source / "01.데이터" / "2.Validation" / "원천데이터"
    for subdir in val_img_base.iterdir():
        if not subdir.is_dir():
            continue
        for grade_dir in subdir.iterdir():
            if not grade_dir.is_dir():
                continue

            images = list(grade_dir.glob("*.jpg")) + list(grade_dir.glob("*.JPG"))

            # Validation은 등급별 2000장
            val_max = min(2000, max_per_grade // 5) if max_per_grade else None
            if val_max and len(images) > val_max:
                images = random.sample(images, val_max)
                print(f"  [Val] {grade_dir.name}: {val_max} sampled")

            for img_path in images:
                rel_path = img_path.relative_to(source)
                dst_path = cache / rel_path
                tasks.append((img_path, dst_path))

    print(f"\nProcessing {len(tasks)} images...")

    success = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_and_save, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks)):
            if future.result():
                success += 1

    print(f"Cached {success}/{len(tasks)} images to {cache}")
    return success


if __name__ == "__main__":
    # 과수화상병 - 과실 (Area=1)
    print("\n" + "="*50)
    print("1. 과수화상병 데이터 캐싱 (과실, Area=1)")
    print("="*50)
    cache_disease_dataset(
        source_dir=r"C:\code\P1_apple_RGB\data\disease_data",
        cache_dir=r"C:\code\P1_apple_RGB\cached_data\disease_fruit",
        area_filter=[1]
    )

    # 과수화상병 - 잎 (Area=3)
    print("\n" + "="*50)
    print("2. 과수화상병 데이터 캐싱 (잎, Area=3)")
    print("="*50)
    cache_disease_dataset(
        source_dir=r"C:\code\P1_apple_RGB\data\disease_data",
        cache_dir=r"C:\code\P1_apple_RGB\cached_data\disease_leaf",
        area_filter=[3]
    )

    # 당도 데이터 (등급별 10,000장 샘플링)
    print("\n" + "="*50)
    print("3. 당도 데이터 캐싱 (등급별 10,000장 샘플링)")
    print("="*50)
    cache_sugar_dataset(
        source_dir=r"C:\code\P1_apple_RGB\data\sugar_data",
        cache_dir=r"C:\code\P1_apple_RGB\cached_data\sugar",
        max_per_grade=10000  # 등급별 최대 10,000장
    )

    print("\n" + "="*50)
    print("전처리 캐싱 완료!")
    print("="*50)

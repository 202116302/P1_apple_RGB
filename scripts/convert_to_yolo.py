"""
당도 데이터를 YOLO segmentation 형식으로 변환하는 스크립트
"""
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_segmentation_to_yolo(segmentation, img_width, img_height):
    """
    segmentation 좌표를 YOLO 형식(0~1 정규화)으로 변환
    """
    normalized = []
    for i in range(0, len(segmentation), 2):
        x = segmentation[i] / img_width
        y = segmentation[i + 1] / img_height
        # 범위 클리핑 (0~1)
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        normalized.extend([x, y])
    return normalized


def process_json_file(json_path, output_label_path, class_id=0):
    """
    JSON 라벨 파일을 YOLO 형식의 txt 파일로 변환
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_width = data['images']['img_width']
        img_height = data['images']['img_height']
        segmentation = data['annotations']['segmentation']

        if not segmentation or len(segmentation) < 6:
            return False

        # YOLO 형식으로 변환
        normalized_coords = convert_segmentation_to_yolo(segmentation, img_width, img_height)

        # YOLO 라벨 파일 작성
        with open(output_label_path, 'w') as f:
            coords_str = ' '.join([f"{coord:.6f}" for coord in normalized_coords])
            f.write(f"{class_id} {coords_str}\n")

        return True
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return False


def create_yolo_dataset(
    source_base_path,
    output_base_path,
    train_ratio=0.8
):
    """
    당도 데이터셋을 YOLO 형식으로 변환
    """
    output_path = Path(output_base_path)

    # 출력 폴더 생성
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 데이터 경로 설정
    source_path = Path(source_base_path)

    # Training 데이터 처리
    train_img_path = source_path / "01.데이터" / "1.Training" / "원천데이터" / "후지1" / "당도A등급"
    train_label_path = source_path / "01.데이터" / "1.Training" / "라벨링데이터_230525_add" / "후지1" / "당도A등급"

    # Validation 데이터 처리 (A, B, C 등급 모두 포함)
    val_base_img = source_path / "01.데이터" / "2.Validation" / "원천데이터" / "후지"
    val_base_label = source_path / "01.데이터" / "2.Validation" / "라벨링데이터_230525_add" / "후지"

    processed_count = {'train': 0, 'val': 0}
    error_count = 0

    # Training 데이터 처리
    print("Processing Training data...")
    if train_label_path.exists():
        json_files = list(train_label_path.glob("*.json"))
        for json_file in tqdm(json_files, desc="Training"):
            img_name = json_file.stem + ".jpg"
            img_file = train_img_path / img_name

            if img_file.exists():
                # 이미지 복사
                output_img = output_path / 'images' / 'train' / img_name
                output_label = output_path / 'labels' / 'train' / (json_file.stem + ".txt")

                if process_json_file(json_file, output_label, class_id=0):
                    shutil.copy2(img_file, output_img)
                    processed_count['train'] += 1
                else:
                    error_count += 1

    # Validation 데이터 처리
    print("Processing Validation data...")
    for grade in ['당도A등급', '당도B등급', '당도C등급']:
        val_img_path = val_base_img / grade
        val_label_path = val_base_label / grade

        if val_label_path.exists():
            json_files = list(val_label_path.glob("*.json"))
            for json_file in tqdm(json_files, desc=f"Validation {grade}"):
                img_name = json_file.stem + ".jpg"
                img_file = val_img_path / img_name

                if img_file.exists():
                    output_img = output_path / 'images' / 'val' / img_name
                    output_label = output_path / 'labels' / 'val' / (json_file.stem + ".txt")

                    if process_json_file(json_file, output_label, class_id=0):
                        shutil.copy2(img_file, output_img)
                        processed_count['val'] += 1
                    else:
                        error_count += 1

    # data.yaml 생성
    yaml_content = f"""# YOLO Apple Segmentation Dataset
path: {output_path.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: apple

# Dataset info
nc: 1  # number of classes
"""

    with open(output_path / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"\n=== Dataset Creation Complete ===")
    print(f"Training images: {processed_count['train']}")
    print(f"Validation images: {processed_count['val']}")
    print(f"Errors: {error_count}")
    print(f"Output path: {output_path.absolute()}")
    print(f"Config file: {output_path / 'data.yaml'}")


if __name__ == "__main__":
    # 경로 설정
    SOURCE_PATH = r"C:\code\P1_apple_RGB\data\sugar_data"
    OUTPUT_PATH = r"C:\code\P1_apple_RGB\datasets\yolo_apple_seg"

    create_yolo_dataset(SOURCE_PATH, OUTPUT_PATH)

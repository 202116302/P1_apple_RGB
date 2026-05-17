"""
전체 추론 파이프라인
이미지 입력 → YOLO 분할 → EfficientNet 분류 → 결과 출력

모델 구성:
- YOLO: 사과 객체 분할
- EfficientNet (과실 병): 과실 병 감염 분류
- EfficientNet (잎 병): 잎 병 감염 분류
- EfficientNet (숙도): 숙도/당도 등급 분류
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from models.efficientnet_classifier import load_model
from models.dataset import get_transforms
from models.preprocessing import preprocess_image


class AppleAnalysisPipeline:
    """
    사과 분석 파이프라인
    """
    def __init__(
        self,
        yolo_model_path: str,
        fruit_disease_model_path: str,
        leaf_disease_model_path: str,
        sugar_model_path: str,
        device: str = 'cuda'
    ):
        self.device = device

        # 모델 로드
        print("Loading models...")
        self.yolo = YOLO(yolo_model_path)
        self.fruit_disease_model = load_model(fruit_disease_model_path, num_classes=2, device=device)
        self.leaf_disease_model = load_model(leaf_disease_model_path, num_classes=2, device=device)
        self.sugar_model = load_model(sugar_model_path, num_classes=3, device=device)

        # 전처리 변환
        self.transform = get_transforms(img_size=224, is_training=False)

        # 클래스 이름
        self.disease_classes = ['정상', '질병']
        self.sugar_classes = ['A등급', 'B등급', 'C등급']

        print("All models loaded successfully!")

    def analyze_fruit_image(self, image_path: str, conf_threshold: float = 0.5):
        """
        과실 이미지 분석 (YOLO 분할 → 병 감염 + 숙도 분류)

        Args:
            image_path: 입력 이미지 경로
            conf_threshold: YOLO 신뢰도 임계값

        Returns:
            분석 결과 리스트
        """
        image = cv2.imread(image_path)

        # White Balance + CLAHE 전처리 적용
        image = preprocess_image(image, apply_wb=True, apply_clahe_flag=True)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # YOLO 추론
        yolo_results = self.yolo.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False
        )[0]

        results = []

        if yolo_results.masks is not None:
            for i, (mask, box) in enumerate(zip(yolo_results.masks.data, yolo_results.boxes)):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])

                # 사과 영역 크롭
                apple_crop = image_rgb[y1:y2, x1:x2]
                if apple_crop.size == 0:
                    continue

                apple_pil = Image.fromarray(apple_crop)
                apple_tensor = self.transform(apple_pil).unsqueeze(0).to(self.device)

                # 과실 병 감염 분류
                with torch.no_grad():
                    disease_output = self.fruit_disease_model(apple_tensor)
                    disease_probs = torch.softmax(disease_output, dim=1)
                    disease_pred = disease_probs.argmax(dim=1).item()
                    disease_conf = disease_probs[0][disease_pred].item()

                # 숙도 분류
                with torch.no_grad():
                    sugar_output = self.sugar_model(apple_tensor)
                    sugar_probs = torch.softmax(sugar_output, dim=1)
                    sugar_pred = sugar_probs.argmax(dim=1).item()
                    sugar_conf = sugar_probs[0][sugar_pred].item()

                result = {
                    'id': i + 1,
                    'type': '과실',
                    'bbox': [x1, y1, x2, y2],
                    'yolo_confidence': confidence,
                    'disease': {
                        'class': self.disease_classes[disease_pred],
                        'confidence': disease_conf
                    },
                    'sugar_grade': {
                        'class': self.sugar_classes[sugar_pred],
                        'confidence': sugar_conf
                    }
                }
                results.append(result)

        return results

    def analyze_leaf_image(self, image_path: str):
        """
        잎 이미지 분석 (전체 이미지 → 병 감염 분류)

        Args:
            image_path: 입력 이미지 경로

        Returns:
            분석 결과
        """
        # 이미지 로드 및 전처리 (White Balance + CLAHE)
        image_cv = cv2.imread(image_path)
        image_cv = preprocess_image(image_cv, apply_wb=True, apply_clahe_flag=True)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            disease_output = self.leaf_disease_model(image_tensor)
            disease_probs = torch.softmax(disease_output, dim=1)
            disease_pred = disease_probs.argmax(dim=1).item()
            disease_conf = disease_probs[0][disease_pred].item()

        return {
            'type': '잎',
            'disease': {
                'class': self.disease_classes[disease_pred],
                'confidence': disease_conf
            }
        }

    def visualize_results(self, image_path: str, results: list, output_path: str = None):
        """
        결과 시각화
        """
        image = cv2.imread(image_path)

        for result in results:
            if 'bbox' not in result:
                continue

            x1, y1, x2, y2 = result['bbox']

            # 병 감염 상태에 따른 색상
            if result['disease']['class'] == '질병':
                color = (0, 0, 255)  # 빨강
            else:
                color = (0, 255, 0)  # 초록

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 라벨
            disease = result['disease']['class']
            sugar = result.get('sugar_grade', {}).get('class', '')
            label = f"#{result['id']} {disease}"
            if sugar:
                label += f" | {sugar}"

            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Result saved to: {output_path}")

        return image


def main():
    """테스트 실행"""
    # 전처리 적용된 최신 모델 사용
    YOLO_MODEL = r"C:\code\P1_apple_RGB\models\yolo_apple_seg\weights\best.pt"
    FRUIT_DISEASE_MODEL = r"C:\code\P1_apple_RGB\models\disease_fruit_classifier_preprocessed\best_model.pth"
    LEAF_DISEASE_MODEL = r"C:\code\P1_apple_RGB\models\disease_leaf_classifier_preprocessed\best_model.pth"
    SUGAR_MODEL = r"C:\code\P1_apple_RGB\models\sugar_classifier_preprocessed\best_model.pth"

    # 테스트 이미지
    TEST_IMAGE = r"C:\code\P1_apple_RGB\apple1.jpg"

    pipeline = AppleAnalysisPipeline(
        yolo_model_path=YOLO_MODEL,
        fruit_disease_model_path=FRUIT_DISEASE_MODEL,
        leaf_disease_model_path=LEAF_DISEASE_MODEL,
        sugar_model_path=SUGAR_MODEL,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 과실 이미지 분석
    print("\n=== 과실 이미지 분석 ===")
    results = pipeline.analyze_fruit_image(TEST_IMAGE)

    for result in results:
        print(f"\n사과 #{result['id']}")
        print(f"  - 위치: {result['bbox']}")
        print(f"  - 병 감염: {result['disease']['class']} ({result['disease']['confidence']*100:.1f}%)")
        print(f"  - 숙도: {result['sugar_grade']['class']} ({result['sugar_grade']['confidence']*100:.1f}%)")

    pipeline.visualize_results(
        TEST_IMAGE,
        results,
        output_path=r"C:\code\P1_apple_RGB\test_result.jpg"
    )


if __name__ == "__main__":
    main()

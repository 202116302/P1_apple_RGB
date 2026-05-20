# RGB 이미지 기반 사과 품질 분석 플랫폼

RGB 카메라 이미지를 활용한 사과 품질(병해, 크기, 무게) 자동 분석 웹 플랫폼

## 프로젝트 개요

본 프로젝트는 사과 RGB 이미지를 입력받아 개별 과실을 자동으로 탐지하고, 각 과실에 대해 병해 유무, 크기 등급(L/M/S), 예측 무게를 실시간으로 분석하는 웹 기반 플랫폼입니다.

### 주요 기능

- **사과 탐지**: YOLOv8n-seg 기반 인스턴스 세그멘테이션
- **병해 분류**: EfficientNet-B0 기반 이진 분류 (정상/질병)
- **크기 분류**: EfficientNet-B0 기반 3클래스 분류 (L/M/S)
- **무게 예측**: EfficientNet-B0 기반 회귀 모델
- **GPS 시각화**: EXIF 정보 기반 촬영 위치 지도 표시

## 시스템 아키텍처

```
Input Image (RGB)
       │
       ▼
┌─────────────────┐
│  Preprocessing  │  White Balance + CLAHE
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YOLOv8n-seg    │  Instance Segmentation (mAP@50: 99.44%)
└────────┬────────┘
         │
    ┌────┴────┬────────────┐
    ▼         ▼            ▼
┌───────┐ ┌───────┐  ┌───────────┐
│Disease│ │ Size  │  │  Weight   │
│ Clf.  │ │ Clf.  │  │ Regressor │
└───┬───┘ └───┬───┘  └─────┬─────┘
    │         │            │
    ▼         ▼            ▼
 정상/질병   L/M/S      무게(g)
```

## 모델 성능

| Component | Model | Input Size | Metric | Performance |
|-----------|-------|------------|--------|-------------|
| Apple Detection | YOLOv8n-seg | 640×640 | mAP@50 | **99.44%** |
| Disease Detection | EfficientNet-B0 | 224×224 | Accuracy | **100.00%** |
| Size Classification | EfficientNet-B0 | 224×224 | Accuracy | 66.88% |
| Weight Prediction | EfficientNet-B0 | 224×224 | MAE | **30.11g** |

## 크기 등급 기준

| 등급 | 한국 표준 | 무게 범위 | 평균 무게 |
|-----|----------|----------|----------|
| L | 특 (特) | 327 ~ 450 g | 363 g |
| M | 상 (上) | 255 ~ 326 g | 282 g |
| S | 보통 | 185 ~ 254 g | 245 g |

## 설치 및 실행

### 요구사항

- Python 3.8+
- CUDA (GPU 사용 시)

### 설치

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 서버 실행

```bash
# Windows
run_server.bat

# 또는 직접 실행
python web/app.py
```

브라우저에서 `http://localhost:8000` 접속

## 프로젝트 구조

```
P1_apple_RGB/
├── web/
│   ├── app.py                 # FastAPI 백엔드
│   ├── templates/
│   │   └── index.html         # 프론트엔드 UI
│   └── uploads/               # 업로드 이미지 저장
├── scripts/
│   ├── models/
│   │   ├── efficientnet_classifier.py
│   │   ├── efficientnet_regressor.py
│   │   ├── preprocessing.py   # White Balance, CLAHE
│   │   ├── dataset.py
│   │   ├── qc_dataset.py
│   │   └── trainer.py
│   ├── train_disease_fruit.py
│   ├── train_disease_leaf.py
│   ├── train_size_classifier.py
│   ├── train_weight_regressor.py
│   └── train_yolo.py
├── models/                    # 학습된 모델 가중치 (gitignore)
├── data/                      # 원본 데이터 (gitignore)
├── datasets/                  # YOLO 데이터셋 (gitignore)
├── requirements.txt
└── README.md
```

## 데이터 출처

- **AI Hub**: RGB 카메라 이미지 기반 과실 품질 분류 AI 데이터
- **품종**: 후지(富士) 사과
- **데이터셋 규모**:
  - 세그멘테이션: 76,856장 (Train: 50,156 / Val: 26,700)
  - QC 이미지: 10,948장 (크기/무게 라벨)

## 전처리 파이프라인

1. **White Balance (Gray World Algorithm)**
   - 조명 조건에 따른 색상 왜곡 보정
   - 이미지의 평균 색상이 회색이 되도록 조정

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - LAB 색공간의 L채널에만 적용
   - clipLimit: 2.0, tileGridSize: 8×8
   - 대비 향상 및 밝기 편차 보정

## 학습 설정

### 공통 설정
- Optimizer: AdamW
- LR Schedule: CosineAnnealingLR
- Pretrained: ImageNet

### YOLO 세그멘테이션
- Epochs: 100 (Early Stopping: 20)
- Batch Size: 16
- Data Augmentation: Mosaic, Mixup, HSV, Rotation

### EfficientNet 분류/회귀
- Epochs: 30-100 (Early Stopping: 7)
- Batch Size: 32-64
- Loss: CrossEntropy (분류) / Huber (회귀)

## API 엔드포인트

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | 메인 페이지 |
| POST | `/api/upload` | 이미지 업로드 및 분석 |
| GET | `/api/results` | 분석 결과 조회 |
| GET | `/api/stats` | 통계 정보 |
| DELETE | `/api/results` | 결과 삭제 |

## 라이선스

본 프로젝트는 연구 및 교육 목적으로 개발되었습니다.

## 참고문헌

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- EfficientNet: https://arxiv.org/abs/1905.11946
- AI Hub: https://aihub.or.kr/

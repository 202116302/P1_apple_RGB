"""
EfficientNet-B0 기반 이미지 분류 모델
병 감염 분류 및 숙도 분류에 공통으로 사용
"""
import torch
import torch.nn as nn
import timm


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 기반 분류 모델
    """
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()

        # EfficientNet-B0 백본 로드
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # feature extractor로 사용
            global_pool='avg'
        )

        # 분류 헤드
        num_features = self.backbone.num_features  # 1280 for B0
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        """특징 벡터 추출 (시각화/분석용)"""
        return self.backbone(x)


def create_model(num_classes: int, pretrained: bool = True, device: str = 'cuda'):
    """
    모델 생성 헬퍼 함수
    """
    model = EfficientNetClassifier(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model


def load_model(model_path: str, num_classes: int, device: str = 'cuda'):
    """
    저장된 모델 로드 (체크포인트 형식 지원)
    """
    model = EfficientNetClassifier(num_classes=num_classes, pretrained=False)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 체크포인트 형식인 경우 (model_state_dict 키가 있음)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 순수 state_dict인 경우
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model

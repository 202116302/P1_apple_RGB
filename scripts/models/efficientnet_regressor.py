"""
EfficientNet-B0 기반 회귀 모델
무게 예측에 사용
"""
import torch
import torch.nn as nn
import timm


class EfficientNetRegressor(nn.Module):
    """
    EfficientNet-B0 기반 회귀 모델
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()

        # EfficientNet-B0 백본 로드
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # feature extractor로 사용
            global_pool='avg'
        )

        # 회귀 헤드
        num_features = self.backbone.num_features  # 1280 for B0
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features).squeeze(-1)

    def get_features(self, x):
        """특징 벡터 추출 (시각화/분석용)"""
        return self.backbone(x)


def create_regressor(pretrained: bool = True, device: str = 'cuda'):
    """
    회귀 모델 생성 헬퍼 함수
    """
    model = EfficientNetRegressor(pretrained=pretrained)
    model = model.to(device)
    return model


def load_regressor(model_path: str, device: str = 'cuda'):
    """
    저장된 회귀 모델 로드
    """
    model = EfficientNetRegressor(pretrained=False)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model

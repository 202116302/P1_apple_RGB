"""
ResNet50 기반 병해 분류 모델 (AI Hub 모델)
"""
import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet_model(num_classes: int = 3, pretrained: bool = False):
    """
    ResNet50 모델 생성
    """
    network = models.resnet50(pretrained=pretrained)
    network.fc = nn.Linear(network.fc.in_features, num_classes)
    return network


def load_resnet_model(model_path: str, num_classes: int = 3, device: str = 'cuda'):
    """
    저장된 ResNet50 모델 로드 (AI Hub 형식)

    Args:
        model_path: 모델 가중치 경로
        num_classes: 클래스 수 (3: 정상, 질병1, 질병2)
        device: 실행 디바이스

    Returns:
        로드된 모델
    """
    model = get_resnet_model(num_classes=num_classes, pretrained=False)

    checkpoint = torch.load(model_path, map_location=device)

    # AI Hub 모델은 model_state_dict 키로 저장됨
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


# 클래스 매핑 (AI Hub 79번 데이터셋 기준)
# 00: 정상, 07: 과수화상병(가지), 08: 과수화상병(잎)
DISEASE_CLASSES = {
    0: "정상",
    1: "과수화상병(가지)",
    2: "과수화상병(잎)"
}

# 간단한 이진 분류용 매핑
DISEASE_BINARY = {
    0: "정상",
    1: "질병",
    2: "질병"
}

"""
이미지 전처리 모듈
- White Balance: 색 왜곡 보정
- CLAHE: 밝기 편차 보정
"""
import cv2
import numpy as np
from PIL import Image


def apply_white_balance(image: np.ndarray, method: str = 'gray_world') -> np.ndarray:
    """
    White Balance 적용하여 색 왜곡 보정

    Args:
        image: BGR 이미지 (numpy array)
        method: 'gray_world' 또는 'white_patch'

    Returns:
        White Balance 적용된 이미지
    """
    if method == 'gray_world':
        # Gray World 알고리즘
        # 이미지의 평균 색상이 회색이 되도록 조정
        result = image.copy().astype(np.float32)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3

        result[:, :, 0] = result[:, :, 0] * (avg_gray / (avg_b + 1e-6))
        result[:, :, 1] = result[:, :, 1] * (avg_gray / (avg_g + 1e-6))
        result[:, :, 2] = result[:, :, 2] * (avg_gray / (avg_r + 1e-6))

        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    elif method == 'white_patch':
        # White Patch 알고리즘
        # 가장 밝은 픽셀이 흰색이 되도록 조정
        result = image.copy().astype(np.float32)
        max_b = np.percentile(result[:, :, 0], 99)
        max_g = np.percentile(result[:, :, 1], 99)
        max_r = np.percentile(result[:, :, 2], 99)

        result[:, :, 0] = result[:, :, 0] * (255.0 / (max_b + 1e-6))
        result[:, :, 1] = result[:, :, 1] * (255.0 / (max_g + 1e-6))
        result[:, :, 2] = result[:, :, 2] * (255.0 / (max_r + 1e-6))

        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    else:
        return image


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
    밝기 편차를 보정하여 대비 향상

    Args:
        image: BGR 이미지 (numpy array)
        clip_limit: 대비 제한 값 (높을수록 대비 강함)
        tile_grid_size: 타일 크기

    Returns:
        CLAHE 적용된 이미지
    """
    # LAB 색공간으로 변환 (L 채널에만 CLAHE 적용)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    # 다시 합치고 BGR로 변환
    lab_clahe = cv2.merge([l_clahe, a, b])
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return result


def preprocess_image(
    image: np.ndarray,
    apply_wb: bool = True,
    apply_clahe_flag: bool = True,
    wb_method: str = 'gray_world',
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: tuple = (8, 8)
) -> np.ndarray:
    """
    전체 전처리 파이프라인

    Args:
        image: BGR 이미지 (numpy array)
        apply_wb: White Balance 적용 여부
        apply_clahe_flag: CLAHE 적용 여부
        wb_method: White Balance 방법
        clahe_clip_limit: CLAHE 클립 제한값
        clahe_tile_size: CLAHE 타일 크기

    Returns:
        전처리된 이미지
    """
    result = image.copy()

    # 1. White Balance 적용
    if apply_wb:
        result = apply_white_balance(result, method=wb_method)

    # 2. CLAHE 적용
    if apply_clahe_flag:
        result = apply_clahe(result, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_size)

    return result


def preprocess_pil_image(
    image: Image.Image,
    apply_wb: bool = True,
    apply_clahe_flag: bool = True
) -> Image.Image:
    """
    PIL Image 전처리 (Dataset에서 사용)

    Args:
        image: PIL Image (RGB)

    Returns:
        전처리된 PIL Image
    """
    # PIL -> numpy (RGB -> BGR)
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 전처리 적용
    processed = preprocess_image(img_bgr, apply_wb=apply_wb, apply_clahe_flag=apply_clahe_flag)

    # numpy -> PIL (BGR -> RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return Image.fromarray(processed_rgb)


class PreprocessTransform:
    """
    torchvision transforms와 호환되는 전처리 클래스
    """
    def __init__(self, apply_wb: bool = True, apply_clahe: bool = True):
        self.apply_wb = apply_wb
        self.apply_clahe = apply_clahe

    def __call__(self, image: Image.Image) -> Image.Image:
        return preprocess_pil_image(
            image,
            apply_wb=self.apply_wb,
            apply_clahe_flag=self.apply_clahe
        )


# 테스트 코드
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 테스트 이미지 로드
    test_image_path = r"C:\code\P1_apple_RGB\test_image.jpg"

    img = cv2.imread(test_image_path)
    if img is None:
        print("테스트 이미지를 찾을 수 없습니다.")
    else:
        # 전처리 적용
        wb_only = apply_white_balance(img)
        clahe_only = apply_clahe(img)
        both = preprocess_image(img)

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original')
        axes[0, 1].imshow(cv2.cvtColor(wb_only, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('White Balance')
        axes[1, 0].imshow(cv2.cvtColor(clahe_only, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('CLAHE')
        axes[1, 1].imshow(cv2.cvtColor(both, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('White Balance + CLAHE')

        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(r"C:\code\P1_apple_RGB\preprocessing_comparison.png")
        print("비교 이미지 저장: preprocessing_comparison.png")

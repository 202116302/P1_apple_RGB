"""
당도 데이터만 전처리 캐싱 (등급별 10,000장 샘플링)
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from cache_preprocessed_images import cache_sugar_dataset

if __name__ == "__main__":
    print("="*50)
    print("당도 데이터 캐싱 (등급별 10,000장 샘플링)")
    print("="*50)

    cache_sugar_dataset(
        source_dir=r"C:\code\P1_apple_RGB\data\sugar_data",
        cache_dir=r"C:\code\P1_apple_RGB\cached_data\sugar",
        max_per_grade=10000
    )

    print("\n" + "="*50)
    print("당도 데이터 캐싱 완료!")
    print("="*50)

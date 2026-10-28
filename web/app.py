"""
RGB 이미지 기반 과실 분석 웹 서비스
FastAPI 백엔드
"""
import os
import sys
import uuid
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from io import BytesIO

# 모델 경로 추가 (scripts 폴더를 맨 앞에 추가하여 우선 검색)
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import httpx

import torch
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from ultralytics import YOLO

from models.efficientnet_classifier import load_model
from models.efficientnet_regressor import load_regressor
from models.dataset import get_transforms
from models.preprocessing import preprocess_image

app = FastAPI(title="Apple Analysis Platform", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 및 템플릿 설정
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/uploads", StaticFiles(directory=BASE_DIR / "uploads"), name="uploads")
app.mount("/images", StaticFiles(directory=PROJECT_DIR / "images"), name="images")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# 모델 경로
MODEL_DIR = Path(__file__).parent.parent / "models"
YOLO_MODEL_PATH = MODEL_DIR / "yolo_apple_seg" / "weights" / "best.pt"
FRUIT_DISEASE_MODEL_PATH = MODEL_DIR / "disease_fruit_classifier_preprocessed" / "best_model.pth"
DIAMETER_MODEL_PATH = MODEL_DIR / "diameter_regressor" / "best_model.pth"
WEIGHT_MODEL_PATH = MODEL_DIR / "weight_regressor" / "best_model.pth"

# 전역 모델 변수
yolo_model = None
fruit_disease_model = None
diameter_model = None
weight_model = None
diameter_mean = 82.55  # 학습 시 사용된 정규화 파라미터
diameter_std = 5.83
weight_mean = 297.23
weight_std = 54.28
transform = None
device = None

# 분석 결과 저장 (실제 서비스에서는 DB 사용)
analysis_results = []

# 기상 API 설정
WEATHER_API_BASE = "http://weather-rda.digitalag.kr:8001"
JEONJU_STATION_CODE = "5211300000"

# DEM 타일 디렉토리
DEM_TILES_DIR = Path(__file__).parent.parent / "dem" / "tiles"

# 투명 타일 (캐싱용)
TRANSPARENT_TILE = None


def get_transparent_tile():
    """투명한 256x256 PNG 타일 생성"""
    global TRANSPARENT_TILE
    if TRANSPARENT_TILE is None:
        img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        TRANSPARENT_TILE = buffer.getvalue()
    return TRANSPARENT_TILE


def load_models():
    """모델 로드"""
    global yolo_model, fruit_disease_model, diameter_model, weight_model, diameter_mean, diameter_std, weight_mean, weight_std, transform, device

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading models on {device}...")

    # YOLO 모델
    if YOLO_MODEL_PATH.exists():
        yolo_model = YOLO(str(YOLO_MODEL_PATH))
        print("YOLO model loaded")
    else:
        print(f"Warning: YOLO model not found at {YOLO_MODEL_PATH}")

    # 과실 병해 모델
    if FRUIT_DISEASE_MODEL_PATH.exists():
        fruit_disease_model = load_model(str(FRUIT_DISEASE_MODEL_PATH), num_classes=2, device=device)
        print("Fruit disease model loaded")
    else:
        print(f"Warning: Fruit disease model not found")

    # 횡경(크기) 회귀 모델
    if DIAMETER_MODEL_PATH.exists():
        diameter_model = load_regressor(str(DIAMETER_MODEL_PATH), device=device)
        checkpoint = torch.load(str(DIAMETER_MODEL_PATH), map_location=device, weights_only=False)
        if 'diameter_mean' in checkpoint:
            diameter_mean = checkpoint['diameter_mean']
            diameter_std = checkpoint['diameter_std']
        print(f"Diameter regressor model loaded (mean={diameter_mean:.2f}mm, std={diameter_std:.2f}mm)")
    else:
        print(f"Warning: Diameter regressor model not found at {DIAMETER_MODEL_PATH}")

    # 무게 예측 모델
    if WEIGHT_MODEL_PATH.exists():
        weight_model = load_regressor(str(WEIGHT_MODEL_PATH), device=device)
        # 학습 시 저장된 정규화 파라미터 로드
        checkpoint = torch.load(str(WEIGHT_MODEL_PATH), map_location=device, weights_only=False)
        if 'weight_mean' in checkpoint:
            weight_mean = checkpoint['weight_mean']
            weight_std = checkpoint['weight_std']
        print(f"Weight regressor model loaded (mean={weight_mean:.2f}g, std={weight_std:.2f}g)")
    else:
        print(f"Warning: Weight regressor model not found at {WEIGHT_MODEL_PATH}")

    # Transform
    transform = get_transforms(img_size=224, is_training=False, apply_preprocessing=False)

    print("All models loaded!")


def extract_gps_from_exif(image_path: str) -> Optional[dict]:
    """EXIF에서 GPS 정보 추출"""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()

        if not exif_data:
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value

        if not gps_info:
            return None

        # 위도 계산
        if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
            lat = gps_info["GPSLatitude"]
            lat_ref = gps_info["GPSLatitudeRef"]
            latitude = float(lat[0]) + float(lat[1]) / 60 + float(lat[2]) / 3600
            if lat_ref == "S":
                latitude = -latitude
        else:
            return None

        # 경도 계산
        if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
            lon = gps_info["GPSLongitude"]
            lon_ref = gps_info["GPSLongitudeRef"]
            longitude = float(lon[0]) + float(lon[1]) / 60 + float(lon[2]) / 3600
            if lon_ref == "W":
                longitude = -longitude
        else:
            return None

        return {
            "latitude": latitude,
            "longitude": longitude
        }
    except Exception as e:
        print(f"GPS extraction error: {e}")
        return None


def analyze_image(image_path: str) -> dict:
    """이미지 분석 수행"""
    results = {
        "image_path": image_path,
        "filename": Path(image_path).name,
        "gps": None,
        "apples": [],
        "timestamp": datetime.now().isoformat()
    }

    # GPS 추출
    gps = extract_gps_from_exif(image_path)
    results["gps"] = gps

    # 이미지 로드
    img_array = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        return results

    # 전처리 적용
    processed_image = preprocess_image(image, apply_wb=True, apply_clahe_flag=True)
    image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # YOLO 탐지
    if yolo_model:
        yolo_results = yolo_model.predict(source=image_path, conf=0.5, verbose=False)[0]

        if yolo_results.boxes is not None:
            for i, box in enumerate(yolo_results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])

                # 사과 영역 크롭
                apple_crop = image_rgb[y1:y2, x1:x2]
                if apple_crop.size == 0:
                    continue

                apple_result = {
                    "id": i + 1,
                    "bbox": [x1, y1, x2, y2],
                    "yolo_confidence": round(confidence, 3),
                    "disease": None,
                    "size": None,
                    "weight": None
                }

                # PIL 이미지로 변환
                apple_pil = Image.fromarray(apple_crop)
                apple_tensor = transform(apple_pil).unsqueeze(0).to(device)

                # 병해 분류
                if fruit_disease_model:
                    with torch.no_grad():
                        output = fruit_disease_model(apple_tensor)
                        probs = torch.softmax(output, dim=1)
                        pred = probs.argmax(dim=1).item()
                        conf = probs[0][pred].item()

                    disease_classes = ["정상", "질병"]
                    apple_result["disease"] = {
                        "class": disease_classes[pred],
                        "confidence": round(conf, 3)
                    }

                # 횡경(크기) 예측
                if diameter_model:
                    with torch.no_grad():
                        output = diameter_model(apple_tensor)
                        # 정규화 역변환
                        predicted_diameter = output.item() * diameter_std + diameter_mean

                    # mm → 등급 변환 (L: 85mm 이상, M: 75-85mm, S: 75mm 미만)
                    if predicted_diameter >= 85:
                        size_class = "L"
                    elif predicted_diameter >= 75:
                        size_class = "M"
                    else:
                        size_class = "S"

                    apple_result["size"] = {
                        "class": size_class,
                        "diameter_mm": round(predicted_diameter, 1)
                    }

                # 무게 예측
                if weight_model:
                    with torch.no_grad():
                        output = weight_model(apple_tensor)
                        # 정규화 역변환
                        predicted_weight = output.item() * weight_std + weight_mean

                    apple_result["weight"] = {
                        "value": round(predicted_weight, 1),
                        "unit": "g"
                    }

                results["apples"].append(apple_result)

    return results


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    load_models()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """이미지 업로드 및 분석"""
    results = []

    for file in files:
        # 파일 확장자 검증
        ext = Path(file.filename).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue

        # 파일 저장
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{ext}"
        save_path = BASE_DIR / "uploads" / filename

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 분석 수행
        analysis = analyze_image(str(save_path))
        analysis["id"] = file_id
        analysis["original_filename"] = file.filename
        analysis["url"] = f"/uploads/{filename}"

        # 결과 저장
        analysis_results.append(analysis)
        results.append(analysis)

    return JSONResponse(content={"success": True, "results": results})


@app.get("/api/results")
async def get_results():
    """모든 분석 결과 조회"""
    return JSONResponse(content={"results": analysis_results})


@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    """특정 분석 결과 조회"""
    for result in analysis_results:
        if result.get("id") == result_id:
            return JSONResponse(content=result)
    raise HTTPException(status_code=404, detail="Result not found")


@app.delete("/api/results/{result_id}")
async def delete_result(result_id: str):
    """특정 분석 결과 삭제"""
    global analysis_results
    for i, result in enumerate(analysis_results):
        if result.get("id") == result_id:
            # 파일 삭제
            if result.get("image_path"):
                try:
                    Path(result["image_path"]).unlink(missing_ok=True)
                except:
                    pass
            analysis_results.pop(i)
            return JSONResponse(content={"success": True})
    raise HTTPException(status_code=404, detail="Result not found")


@app.delete("/api/results")
async def clear_results():
    """모든 결과 삭제"""
    global analysis_results
    analysis_results = []

    # 업로드된 파일도 삭제
    upload_dir = BASE_DIR / "uploads"
    for f in upload_dir.glob("*"):
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            f.unlink()

    return JSONResponse(content={"success": True})


@app.get("/api/stats")
async def get_stats():
    """통계 정보"""
    total_images = len(analysis_results)
    total_apples = sum(len(r.get("apples", [])) for r in analysis_results)

    disease_stats = {"정상": 0, "질병": 0}
    size_stats = {"L": 0, "M": 0, "S": 0}
    weights = []

    for result in analysis_results:
        for apple in result.get("apples", []):
            if apple.get("disease"):
                disease_stats[apple["disease"]["class"]] += 1
            if apple.get("size"):
                size_stats[apple["size"]["class"]] += 1
            if apple.get("weight"):
                weights.append(apple["weight"]["value"])

    gps_count = sum(1 for r in analysis_results if r.get("gps"))

    weight_stats = None
    if weights:
        weight_stats = {
            "min": round(min(weights), 1),
            "max": round(max(weights), 1),
            "avg": round(sum(weights) / len(weights), 1)
        }

    return JSONResponse(content={
        "total_images": total_images,
        "total_apples": total_apples,
        "images_with_gps": gps_count,
        "disease_stats": disease_stats,
        "size_stats": size_stats,
        "weight_stats": weight_stats
    })


# ========== 예시 이미지 API ==========

@app.get("/api/sample-images")
async def get_sample_images():
    """예시 이미지 목록 조회"""
    images_dir = PROJECT_DIR / "images"
    if not images_dir.exists():
        return JSONResponse(content={"images": []})

    images = []
    for f in images_dir.glob("*"):
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            images.append({
                "filename": f.name,
                "url": f"/images/{f.name}"
            })

    return JSONResponse(content={"images": images})


# ========== 기상 API ==========

@app.get("/api/weather/current")
async def get_current_weather():
    """현재 기상 조회 (완주군)"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{WEATHER_API_BASE}/api/kma/realtime/region/완주군",
                timeout=10.0
            )
            data = response.json()
            # KMA 실황 데이터를 간단한 형식으로 변환
            items = data.get("data", []) if isinstance(data, dict) else data
            if isinstance(items, list) and len(items) > 0:
                result = {"region": "완주군", "temp": None, "hum": None, "wind": None, "rn": None}
                for item in items:
                    cat = item.get("category")
                    val = item.get("obsrvalue")
                    if cat == "T1H": result["temp"] = val
                    elif cat == "REH": result["hum"] = val
                    elif cat == "WSD": result["wind"] = val
                    elif cat == "RN1": result["rn"] = val
                return result
            return data
        except Exception as e:
            return {"error": str(e), "temp": "-", "hum": "-"}


# ========== DEM 타일 서빙 ==========

@app.get("/tiles/slope/{z}/{x}/{y}.png")
async def get_slope_tile(z: int, x: int, y: int):
    """경사도 타일 서빙 (없는 타일은 투명 타일 반환)"""
    tile_path = DEM_TILES_DIR / str(z) / str(x) / f"{y}.png"
    if tile_path.exists():
        return FileResponse(tile_path, media_type="image/png")
    # 타일이 없으면 투명 타일 반환 (404 대신)
    return Response(content=get_transparent_tile(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

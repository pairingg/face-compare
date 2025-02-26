from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import List
import os
import io
import torch
import requests
from PIL import Image
import tempfile
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging
import numpy as np
import cv2

# FastAPI 인스턴스 생성
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 얼굴 감지 및 전처리 함수
def preprocess(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        img_np = img_np[y:y+h, x:x+w]

    img_resized = cv2.resize(img_np, (224, 224))
    img_resized = img_resized / 255.0
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(device)

# ResNet50 특징 추출기
class FeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        return self.fc(torch.flatten(self.model(x), start_dim=1))

# 모델 로드
extractor = FeatureExtractor().to(device).eval()
uploaded_feature_vector = None

# 유사도 계산 함수
def is_similar(vec1, vec2):
    cos_sim = F.cosine_similarity(vec1, vec2).item()
    euc_dist = torch.norm(vec1 - vec2, p=2).item()
    manh_dist = torch.norm(vec1 - vec2, p=1).item()

    return (
        cos_sim >= 0.6 or  # cosine similarity 기준
        euc_dist <= 20.0 or  # 유클리드 거리 기준
        manh_dist <= 70.0  # 맨하탄 거리 기준
    )

# 이미지 파일을 임시로 저장하는 함수
def save_image_from_url(image_url):
    # URL로 이미지 요청
    response = requests.get(image_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Unable to download image from URL")
    
    # 임시 파일로 저장
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(temp_file.name, 'wb') as f:
        f.write(response.content)
    
    return temp_file.name

# 이미지 업로드 및 URL 비교를 위한 API
@app.post("/member/face/")
async def upload_and_compare(
    file: UploadFile = File(...),  # 업로드된 이미지 파일
    urls: str = Form(...),         # 여러 개의 URL을 하나의 문자열로 받아옴 (쉼표로 구분)
):
    global uploaded_feature_vector
    logger.debug("Starting image processing")

    try:
        # 업로드된 이미지 처리
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        with torch.no_grad():
            uploaded_feature_vector = extractor(preprocess(img))
        logger.debug("Image processed successfully")

        # 쉼표로 구분된 URL 문자열을 리스트로 변환
        url_list = urls.split(",")  # 쉼표를 기준으로 URL 분리

        results = {}
        is_same_person = False

        for url in url_list:
            try:
                logger.debug(f"Processing URL: {url.strip()}")
                # URL을 통해 이미지 다운로드 후 임시 저장
                temp_image_path = save_image_from_url(url.strip())  # 공백 제거 후 처리

                # 이미지 로드
                img = Image.open(temp_image_path).convert("RGB")
                with torch.no_grad():
                    feature_vector = extractor(preprocess(img))

                # 비교 수행
                match = is_similar(uploaded_feature_vector, feature_vector)
                results[url.strip()] = match  # URL에 공백 제거 후 저장
                if match:
                    is_same_person = True

                # 임시 파일 삭제
                os.remove(temp_image_path)

            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                results[url.strip()] = f"Error: {str(e)}"

        return {"is_same_person": is_same_person, "detailed_results": results}

    except Exception as e:
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

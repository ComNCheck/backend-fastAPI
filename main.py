from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import Tuple
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import math
import logging
from google.cloud import vision
import re
app = FastAPI()

SAVED_IMAG_PATH = Path("Comparative-image.png")

SIMILARITY_THRESHOLD= 99.0
logging.basicConfig(level=logging.INFO)

def calculate_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    
    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return hist_similarity  

def resize_min_size(img: np.ndarray, min_side: int = 7) -> np.ndarray:
    h, w = img.shape[:2]
    if h < min_side or w < min_side:
        scale = max(min_side / h, min_side / w)
        new_h = math.ceil(h * scale)
        new_w = math.ceil(w * scale)
        logging.info(f"이미지 크기를 리사이즈: ({h}, {w}) -> ({new_h}, {new_w})")
        img = cv2.resize(img, (new_w, new_h))
    return img

def calculate_ssim(img1: np.ndarray, img2: np.ndarray, win_size: int = 3) -> float:
    img1 = resize_min_size(img1, min_side=win_size)
    img2 = resize_min_size(img2, min_side=win_size)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    logging.info(f"SSIM 계산: img1 크기 {img1.shape}, img2 크기 {img2.shape}, win_size={win_size}")

    similarity_index, _ = ssim(img1, img2, full=True, multichannel=True, win_size=win_size)
    return similarity_index

def combined_similarity(img1: np.ndarray, img2: np.ndarray, weight_hist=0.5, weight_ssim=0.5) -> float:
    hist_sim = calculate_histogram_similarity(img1, img2)
    ssim_sim = calculate_ssim(img1, img2, win_size=3) 
    combined = (weight_hist * hist_sim) + (weight_ssim * ssim_sim)
    return combined

def calculate_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    sim_score_0_to_1 = combined_similarity(img1, img2)
    sim_score_0_to_100 = sim_score_0_to_1 * 100.0
    return sim_score_0_to_100

def perform_ocr(image: np.ndarray) -> dict:
    try:
        logging.info("perform_ocr 함수 실행")
        client = vision.ImageAnnotatorClient()

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.medianBlur(gray_image, 3)
        binary_image = cv2.adaptiveThreshold(
            blurred_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        height, width = binary_image.shape
        if height < 800 or width < 800:
            resized_image = cv2.resize(binary_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        else:
            resized_image = binary_image

        success, encoded_image = cv2.imencode('.png', resized_image)
        if not success:
            raise HTTPException(status_code=500, detail="이미지 인코딩 실패")
        
        content = encoded_image.tobytes()
        vision_image = vision.Image(content=content)

        image_context = vision.ImageContext(language_hints=["ko", "en"]) 
        response = client.document_text_detection(image=vision_image, image_context=image_context)
        if response.error.message:
            logging.error(f"Vision API 오류: {response.error.message}")
            raise HTTPException(status_code=500, detail=f"Vision API 오류: {response.error.message}")

        texts = response.text_annotations

        logging.debug(f"인식된 텍스트: {texts}")

        if not texts:
            return {"error": "텍스트를 인식할 수 없습니다."}

        ocr_text = texts[0].description
        logging.info(f"OCR 추출 텍스트: {ocr_text}")

        fields = extract_fields(ocr_text)
        logging.info(f"추출된 필드: {fields}")

        return fields

    except FileNotFoundError as fnf_error:
        logging.error(f"키 파일 오류: {fnf_error}")
        raise HTTPException(status_code=500, detail="Google Cloud Vision API 키 파일을 찾을 수 없습니다.")
    except Exception as e:
        logging.error(f"OCR 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="OCR 처리 중 오류가 발생했습니다.")

def extract_fields(ocr_text: str) -> dict:

    name = None
    student_id = None
    major = None
    lines = ocr_text.splitlines()

    student_id_pattern = re.compile(r'\b\d{9}\b')  
    major_pattern = re.compile(r'.*전공.*')  

    previous_line = None

    for line in lines:
        line = line.strip()
        if not line:
            continue 

        if not student_id:
            match = student_id_pattern.search(line)
            if match:
                student_id = match.group()
                if previous_line:
                    name = previous_line
                continue

        if not major and major_pattern.match(line):
            major = line
            continue

        previous_line = line

        if name and student_id and major:
            break

    return {
        'name': name,
        'student_id': student_id,
        'major': major
    }

@app.post("/compare-and-ocr/")
async def compare_and_ocr(file: UploadFile = File(...)):
    try:
        saved_image = cv2.imread(str(SAVED_IMAG_PATH))
        if saved_image is None:
            raise HTTPException(status_code=500, detail="저장된 이미지를 로드할 수 없습니다.")
        
        uploaded_content = await file.read()
        np_uploaded_image = np.frombuffer(uploaded_content, np.uint8)
        uploaded_image = cv2.imdecode(np_uploaded_image, cv2.IMREAD_COLOR)
        if uploaded_image is None:
            raise HTTPException(status_code=400, detail="업로드된 이미지를 읽을 수 없습니다.")
    
        similarity = calculate_similarity(saved_image, uploaded_image)
        if similarity < SIMILARITY_THRESHOLD:
            return JSONResponse(
                {
                    "message": f"유사도가 기준({SIMILARITY_THRESHOLD}%) 미만입니다.",
                    "similarity": f"{similarity:.2f}%"
                },
                status_code=400
            )
        extracted_text = perform_ocr(uploaded_image)

        return JSONResponse(
            {
                "message": "유사도 일치합니다.",
                "similarity": f"{similarity:.2f}%",
                "extracted_text": extracted_text
            }
        )
    
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logging.error(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))



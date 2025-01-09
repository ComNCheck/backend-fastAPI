from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import Tuple
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

from google.cloud import vision

app = FastAPI()

SAVED_IMAG_PATH = Path("Comparative-image.png")

SIMILARITY_THRESHOLD= 70.0

def calculate_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    
    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return hist_similarity  

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    similarity_index, _ = ssim(img1, img2, full=True, multichannel=True)
    return similarity_index  # 0.0 ~ 1.0


def combined_similarity(img1: np.ndarray, img2: np.ndarray, weight_hist=0.5, weight_ssim=0.5) -> float:

    hist_sim = calculate_histogram_similarity(img1, img2)
    ssim_sim = calculate_ssim(img1, img2)
    
    combined = (weight_hist * hist_sim) + (weight_ssim * ssim_sim)
    
    return combined 

def calculate_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    sim_score_0_to_1 = combined_similarity(img1, img2)
    sim_score_0_to_100 = sim_score_0_to_1 * 100.0
    return sim_score_0_to_100


@app.post("/compare-and-ocr/")
async def compare_and_ocr(file: UploadFile = File(...)):
    saved_image = cv2.imread(str(SAVED_IMAG_PATH))

    uploaded_content = await file.read()
    np_uploaded_image = np.frombuffer(uploaded_content, np.uint8)
    uploaded_image = cv2.imdecode(np_uploaded_image, cv2.IMREAD_COLOR)

    similarity = calculate_similarity(saved_image, uploaded_image)



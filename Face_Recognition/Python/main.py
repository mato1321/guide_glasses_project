"""
FastAPI 伺服器 - 將 FaceEngine 包成 HTTP API
之後 Android App 透過這些 API 與 Python 後端溝通
"""

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from face_engine import FaceEngine

# ========== 初始化 ==========
app = FastAPI(
    title="Rokid 人臉辨識 API",
    description="基於 InsightFace 的人臉辨識服務",
    version="1.0.0"
)

# 初始化人臉辨識引擎
engine = FaceEngine(db_path="face_database", similarity_threshold=0.4)
engine.load_database()


# ========== 工具函數 ==========
async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """將上傳的檔案轉成 OpenCV 圖片"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


# ========== API 端點 ==========

@app.get("/")
async def root():
    """伺服器狀態檢查"""
    return {
        "status": "running",
        "registered_faces": engine.get_registered_names(),
        "total_people": len(engine.get_registered_names())
    }


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """
    🔍 人臉辨識 API
    上傳一張圖片 → 回傳辨識結果
    """
    image = await read_image_from_upload(file)

    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "無法解析圖片"}
        )

    results = engine.recognize(image)

    return {
        "success": True,
        "face_count": len(results),
        "faces": results
    }


@app.post("/register")
async def register(
    name: str = Query(..., description="要註冊的人名"),
    file: UploadFile = File(...)
):
    """
    📝 人臉註冊 API
    上傳照片 + 姓名 → 加入人臉資料庫
    """
    image = await read_image_from_upload(file)

    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "無法解析圖片"}
        )

    result = engine.register_face(name, image)

    if result["success"]:
        return result
    else:
        return JSONResponse(status_code=400, content=result)


@app.delete("/faces/{name}")
async def delete_face(name: str):
    """🗑️ 刪除已註冊的人臉"""
    result = engine.delete_face(name)

    if result["success"]:
        return result
    else:
        return JSONResponse(status_code=404, content=result)


@app.get("/faces")
async def list_faces():
    """📋 列出所有已註冊的人臉"""
    return {
        "faces": engine.get_registered_names(),
        "total": len(engine.get_registered_names())
    }


@app.post("/reload")
async def reload_database():
    """🔄 重新載入人臉資料庫"""
    engine.face_database.clear()
    engine.load_database()
    return {
        "message": "資料庫已重新載入",
        "faces": engine.get_registered_names()
    }

from admin import router as admin_router
app.include_router(admin_router)
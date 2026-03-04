#FastAPI 伺服器 - 將 FaceEngine 包成 HTTP AP，I之後 Android App 透過這些 API 與 Python 後端溝通
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from face_engine import FaceEngine
app = FastAPI(
    title="Rokid glasses人臉辨識 API",
    description="基於 InsightFace 的人臉辨識",
    version="1.0.0"
)
engine = FaceEngine(db_path="face_database", similarity_threshold=0.4)
engine.load_database()


# tools.py - 將上傳的圖片轉換為 OpenCV 格式
async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


# API routes
@app.get("/")
async def root():
    return {
        "status": "running",
        "registered_faces": engine.get_registered_names(),
        "total_people": len(engine.get_registered_names())
    }

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    image = await read_image_from_upload(file)
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "無法解析圖片"}
        )

    # debug - 儲存收到的圖片到本地，方便檢查
    import cv2
    cv2.imwrite("debug.jpg", image)
    print(f"Debug image saved to debug.jpg")
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
    result = engine.delete_face(name)
    if result["success"]:
        return result
    else:
        return JSONResponse(status_code=404, content=result)

@app.get("/faces")
async def list_faces():
    return {
        "faces": engine.get_registered_names(),
        "total": len(engine.get_registered_names())
    }


@app.post("/reload")
async def reload_database():
    engine.face_database.clear()
    engine.load_database()
    return {
        "message": "資料庫已重新載入",
        "faces": engine.get_registered_names()
    }

from admin import router as admin_router
app.include_router(admin_router)
# Rokid Face Recognition 人臉辨識系統

基於 Rokid AR 眼鏡的即時人臉辨識系統。透過眼鏡攝像頭捕捉影像，送到 Python 伺服器進行辨識，結果顯示在眼鏡鏡片上並語音播報。

## 系統架構

```
┌─────────────┐  USB-C  ┌──────────────┐  WiFi  ┌──────────────┐
│ Rokid 眼鏡   │◄───────►│ Android 手機  │◄──────►│ Python 伺服器 │
│ 📷 攝像頭    │         │ 📱 App        │        │ 🧠 InsightFace│
│ 🖥️ AR 顯示   │         │ 送出圖片      │        │ 人臉辨識     │
│ 🔊 語音播報  │         │ 接收結果      │        │ REST API     │
└─────────────┘         └──────────────┘        └──────────────┘
```

## 功能

- ✅ 即時人臉偵測與辨識
- ✅ 持續自動偵測（每 2 秒）
- ✅ 多人臉偵測，只顯示信心度最高的人
- ✅ TTS 語音播報（每 10 秒一次）
- ✅ Rokid 眼鏡鏡片 AR 顯示（全螢幕大字體）
- ✅ 無眼鏡時自動切換手機攝像頭模式
- ✅ 網頁管理介面（新增/改名/刪除人臉）
- ✅ RESTful API

## 快速開始

### 1. Python 伺服器

```bash
cd python-server
python -m venv venv

# Windows
venv\Scripts\activate

# 安裝套件
pip install -r requirements.txt

# 在 face_database/ 裡放入人臉照片（參考 python-server/README.md）

# 啟動伺服器
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Android App

1. 用 Android Studio 打開 `android/` 資料夾
2. 修改 `ApiClient.kt` 裡的 `BASE_URL` 為你電腦的 IP
   - 模擬器：`http://10.0.2.2:8000`
   - 實體手機：`http://你的電腦IP:8000`
3. Build 並安裝到手機
4. 連接 Rokid 眼鏡（USB-C）

### 3. 使用

1. 確保 Python 伺服器正在運行
2. 打開 Android App
3. 按「🔗 連線測試」確認連線
4. 開啟「🔄 自動持續偵測」開關
5. 對準人臉，眼鏡鏡片會顯示辨識結果

## 管理人臉資料庫

打開瀏覽器：`http://你的電腦IP:8000/admin`

可以在網頁上直接：
- ➕ 新增人臉（上傳照片）
- ✏️ 修改名字
- 🗑️ 刪除人臉
- 🔄 重新載入資料庫

## 技術棧

| 元件 | 技術 |
|------|------|
| 人臉辨識引擎 | InsightFace (buffalo_l) |
| 後端 API | FastAPI + Uvicorn |
| Android App | Kotlin + CameraX + Retrofit |
| 眼鏡投射 | Android Presentation API |
| 語音播報 | Android TTS |
| 眼鏡連接 | UVC Camera (USB-C) |

## 環境需求

- Python 3.8+
- Android Studio (API 26+)
- Rokid AR 眼鏡（選配，沒有也能用手機測試）

## License

MIT License
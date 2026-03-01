# Python 人臉辨識伺服器

基於 InsightFace 的人臉辨識 API 伺服器，提供 RESTful API 給 Android App 呼叫。

## 快速開始

### 1. 建立虛擬環境

```bash
cd python-server
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 新增人臉資料

在 `face_database/` 資料夾裡，為每個人建立一個子資料夾，並放入照片：

```
face_database/
├── 王小明/
│   ├── photo1.jpg    （正面照）
│   ├── photo2.jpg    （微側面）
│   └── photo3.jpg    （不同光線）
└── 李小華/
    └── photo1.jpg
```

> 每個人建議放 3~5 張不同角度/光線的照片，辨識效果更好。

### 4. 啟動伺服器

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 測試

- 伺服器狀態：`http://localhost:8000`
- API 文件：`http://localhost:8000/docs`
- 管理頁面：`http://localhost:8000/admin`

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/` | 伺服器狀態 |
| POST | `/recognize` | 上傳圖片辨識人臉 |
| POST | `/register?name=人名` | 註冊新人臉 |
| DELETE | `/faces/{name}` | 刪除已註冊的人臉 |
| GET | `/faces` | 列出所有已註冊的人 |
| POST | `/reload` | 重新載入資料庫 |
| GET | `/admin` | 網頁管理介面 |

## 管理人臉資料庫

有三種方式：

1. **網頁管理**：打開 `http://localhost:8000/admin`
2. **API 管理**：打開 `http://localhost:8000/docs`
3. **資料夾管理**：直接操作 `face_database/` 資料夾，然後重啟伺服器
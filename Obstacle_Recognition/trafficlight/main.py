import cv2
from ultralytics import YOLO

# 1. 載入模型
model = YOLO("trafficlight.pt")

# 2. 讀取圖片
img = cv2.imread("3.jpg")

# 3. 推論
results = model(img)

# ---【修改名稱邏輯】---
for key, value in results[0].names.items():
    if value == "countdown_go":
        results[0].names[key] = "go"
# ---------------------

# 4. 繪圖 (調整這裡！)
# font_size: 字體大小 (原本100太大，建議改為 20~40 試試看)
# line_width: 框線粗細 (字體變大時，建議框線也加粗到 3~5)
annotated_frame = results[0].plot(conf=True, font_size=20, line_width=3)

# 5. 儲存結果
cv2.imwrite("output_final_large_text.jpg", annotated_frame)
print("已輸出 output_final_large_text.jpg")
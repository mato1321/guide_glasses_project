import cv2
from ultralytics import YOLO

# 載入模型
model = YOLO("yolo-seg.pt")

# 讀取圖片
img = cv2.imread("1.jpg")

# 推論
results = model(img)

# 直接使用官方的繪圖工具
# conf=True 顯示信心分數, font_size 設定字體大小, line_width 設定框線粗細
annotated_frame = results[0].plot(conf=True, font_size=1, line_width=2)

# 儲存結果
cv2.imwrite("output_improved_plot.jpg", annotated_frame)
print("已輸出 output_improved_plot.jpg")
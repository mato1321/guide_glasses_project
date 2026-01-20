import cv2
import inspireface as isf
import numpy as np
import os

def create_face_database(db_path="face_database"):
    face_database = {}
    session = isf.InspireFaceSession(
        isf.HF_ENABLE_FACE_RECOGNITION,
        isf.HF_DETECT_MODE_ALWAYS_DETECT
    )
    
    # 人臉資料庫
    for person_name in os.listdir(db_path):
        person_path = os.path.join(db_path, person_name)
        if not os.path.isdir(person_path):
            continue
        print(f"加載 {person_name} 的人臉特徵")
        face_features = []
        
        for img_file in os.listdir(person_path):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(person_path, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # 檢測人臉
                faces = session.face_detection(image)
                if faces:
                    try:
                        feature = session.face_feature_extract(image, faces[0])
                        if feature is not None:
                            face_features.append(feature)
                            print(f"已加載 {img_file}")
                        else:
                            print(f"失敗：{img_file}")
                    except Exception as e:
                        print(f"錯誤：{e}")
        
        if face_features: # 計算平均特徵向量
            face_database[person_name] = np.mean(face_features, axis=0)
            print(f"{person_name} 的人臉資料庫已準備 (總共 {len(face_features)} 張圖片)\n")
        else:
            print(f"錯誤： {person_name}\n")
    return face_database, session

# 計算人臉相似度（使用餘弦相似度）
def calculate_similarity(feature1, feature2):
    if feature1 is None or feature2 is None:
        return 0
    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(feature1, feature2) / (norm1 * norm2)

# 主程序：實時人臉識別
def main():
    face_database, session = create_face_database()
    if not face_database:
        print("沒有人臉資料庫")
        return
    print(f"總共人臉資料庫: {list(face_database.keys())}")
    print("正在啟動攝像頭")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("攝像頭無法打開")
        return
    
    SIMILARITY_THRESHOLD = 0.4
    frame_count = 0
    
    while True:  
        ret, frame = cap.read()
        if not ret:  
            break
        frame_count += 1
        faces = session.face_detection(frame)
        draw = frame.copy()
        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face.location
            try:
                current_feature = session.face_feature_extract(frame, face)
                if current_feature is None:
                    print(f"錯誤")
                    continue
                
                # 與資料庫中的人臉進行比對
                best_match = None
                best_similarity = 0
                for person_name, db_feature in face_database.items():
                    similarity = calculate_similarity(current_feature, db_feature)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = person_name
                
                # 如果相似度超過閾值，則認為識別成功
                if best_similarity > SIMILARITY_THRESHOLD:
                    label = f"{best_match}"
                    color = (0, 255, 0)  # 綠色
                    if frame_count % 30 == 0:  # 每30幀輸出一次
                        print(f"識別出:  {best_match} (相似度:  {best_similarity:.2f})")
                else:
                    label = f"Unknown"
                    color = (0, 0, 255)  # 紅色
                    if frame_count % 30 == 0:
                        print(f"未知人員 (最佳匹配相似度: {best_similarity:.2f})")
                cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(draw, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow('Face Recognition System', draw)          
            except Exception as e: 
                print(f"錯誤")
                continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("\n程序已退出")
if __name__ == "__main__":
    main()
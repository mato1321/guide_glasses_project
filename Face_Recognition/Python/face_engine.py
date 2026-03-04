import cv2
import numpy as np
import os
import shutil
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(self, db_path="face_database", similarity_threshold=0.4):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.face_database = {}
        self.app = FaceAnalysis(
            name="buffalo_l",     
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace Successfully Initialized\n")
    
    # database management
    def load_database(self):
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            print(f"已建立資料夾：face_database")
            return

        for person_name in os.listdir(self.db_path):
            person_path = os.path.join(self.db_path, person_name)
            if not os.path.isdir(person_path):
                continue
            print(f"loading {person_name}")
            face_features = []

            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(person_path, img_file)
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    # Detect face and extract features
                    faces = self.app.get(image)
                    if faces:
                        face_features.append(faces[0].embedding)
                        print(f"Successfully loaded {img_file}")
                    else:
                        print(f"Failed to detect face in {img_file}")

            if face_features:
                self.face_database[person_name] = np.mean(face_features, axis=0)
                print(f"{person_name}： {len(face_features)} pictures\n")
            else:
                print(f"{person_name}: No valid face images\n")

        print(f"database loaded, total {len(self.face_database)} people")
        print(f"{list(self.face_database.keys())}\n")

    def register_face(self, name, image):
        faces = self.app.get(image)
        if not faces:
            return {"success": False, "message": "圖片中未偵測到人臉"}
        if len(faces) > 1:
            return {"success": False, "message": f"偵測到 {len(faces)} 張人臉，請確保只有一個人"}

        try:
            feature = faces[0].embedding

            # 儲存圖片
            person_path = os.path.join(self.db_path, name)
            os.makedirs(person_path, exist_ok=True)
            existing_count = len([
                f for f in os.listdir(person_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            img_filename = f"{name}_{existing_count + 1}.jpg"
            cv2.imwrite(os.path.join(person_path, img_filename), image)

            # 更新記憶體中的資料庫
            if name in self.face_database:
                old_feature = self.face_database[name]
                self.face_database[name] = np.mean([old_feature, feature], axis=0)
                msg = f"已更新 {name}（現有 {existing_count + 1} 張照片）"
            else:
                self.face_database[name] = feature
                msg = f"已註冊新人臉：{name}"

            return {"success": True, "message": msg}
        except Exception as e:
            return {"success": False, "message": f"註冊失敗：{str(e)}"}

    def delete_face(self, name):
        
        if name not in self.face_database:
            return {"success": False, "message": f"找不到 {name}"}

        del self.face_database[name]
        person_path = os.path.join(self.db_path, name)
        if os.path.exists(person_path):
            shutil.rmtree(person_path)

        return {"success": True, "message": f"已刪除 {name}"}

    def get_registered_names(self):
        return list(self.face_database.keys())

    @staticmethod
    def cosine_similarity(feature1, feature2):
        if feature1 is None or feature2 is None:
            return 0.0
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(feature1, feature2) / (norm1 * norm2))

    def recognize(self, image):
        if not self.face_database:
            return []

        faces = self.app.get(image)
        results = []

        for face in faces:
            bbox = face.bbox  # [x1, y1, x2, y2]
            current_feature = face.embedding
            best_match = None
            best_similarity = 0.0

            for person_name, db_feature in self.face_database.items():
                similarity = self.cosine_similarity(current_feature, db_feature)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_name

            if best_similarity > self.similarity_threshold:
                results.append({
                    "name": best_match,
                    "confidence": round(best_similarity, 4),
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                })
            else:
                results.append({
                    "name": "unknown",
                    "confidence": round(best_similarity, 4),
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                })

        return results
#人臉辨識引擎（基於 InsightFace）
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
        print("InsightFace Initialized with model: buffalo_l")

    def load_database(self):
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            return

        for person_name in os.listdir(self.db_path):
            person_path = os.path.join(self.db_path, person_name)
            if not os.path.isdir(person_path):
                continue

            print(f"loading face features for {person_name}...")
            face_features = []
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(person_path, img_file)
                    try:
                        with open(img_path, 'rb') as f:
                            image_bytes = np.frombuffer(f.read(), np.uint8)
                        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                        if image is None:
                            print(f"Failed to decode image: {img_file}")
                            continue
                        faces = self.app.get(image)
                        if faces:
                            face_features.append(faces[0].embedding)
                            print(f"Success: {img_file}")
                        else:
                            print(f"Failed to detect face: {img_file}")
                            
                    except Exception as e:
                        print(f"Failed to process {img_file}: {str(e)}")
                        continue

            if face_features:
                self.face_database[person_name] = np.mean(face_features, axis=0)
                print(f"{person_name}： {len(face_features)} photos\n")
            else:
                print(f"{person_name}：no valid photos\n")

        print(f"database loaded with {len(self.face_database)} people")
        print(f"   names: {list(self.face_database.keys())}\n")

    def register_face(self, name, image):
        faces = self.app.get(image)
        if not faces:
            return {"success": False, "message": "圖片中未偵測到人臉"}
        if len(faces) > 1:
            return {"success": False, "message": f"偵測到 {len(faces)} 張人臉，請確保只有一個人"}
        try:
            feature = faces[0].embedding
            person_path = os.path.join(self.db_path, name)
            os.makedirs(person_path, exist_ok=True)
            existing_count = len([
                f for f in os.listdir(person_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            img_filename = f"{name}_{existing_count + 1}.jpg"
            img_full_path = os.path.join(person_path, img_filename)
            success, img_encoded = cv2.imencode('.jpg', image)
            if success:
                with open(img_full_path, 'wb') as f:
                    f.write(img_encoded.tobytes())
            else:
                return {"success": False, "message": "圖片編碼失敗"}
            if name in self.face_database:
                old_feature = self.face_database[name]
                self.face_database[name] = np.mean([old_feature, feature], axis=0)
                msg = f"Updated {name}（currently has {existing_count + 1} photos）"
            else:
                self.face_database[name] = feature
                msg = f"Registered new face：{name}"

            return {"success": True, "message": msg}
        except Exception as e:
            return {"success": False, "message": f"註冊失敗：{str(e)}"}

    def delete_face(self, name):
        if name not in self.face_database:
            return {"success": False, "message": f"找不到 {name}"}
        try:
            del self.face_database[name]
            person_path = os.path.join(self.db_path, name)
            if os.path.exists(person_path):
                shutil.rmtree(person_path)
            
            return {"success": True, "message": f"已刪除 {name}"}
        except Exception as e:
            return {"success": False, "message": f"刪除失敗：{str(e)}"}

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
            # within the database, find the best match based on cosine similarity
            best_match = None
            best_similarity = 0.0
            for person_name, db_feature in self.face_database.items():
                similarity = self.cosine_similarity(current_feature, db_feature)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_name
            # if similarity is above the threshold, return the person's name; otherwise return "unknown"
            if best_match and best_similarity >= self.similarity_threshold:
                name = best_match
            else:
                name = "unknown"
            results.append({
                "name": name,
                "confidence": float(best_similarity),
                "bbox": bbox.tolist()
            })
        return results
"""
人臉辨識引擎 - 改用 InsightFace（支援 Windows）
功能跟你原本用 InspireFace 的程式碼完全一樣
"""

import cv2
import numpy as np
import os
import shutil
from insightface.app import FaceAnalysis


class FaceEngine:
    """人臉辨識引擎（基於 InsightFace）"""

    def __init__(self, db_path="face_database", similarity_threshold=0.4):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.face_database = {}

        # ===== 改動 1：模型從 buffalo_sc 改成 buffalo_l =====
        self.app = FaceAnalysis(
            name="buffalo_l",      # ← 改這裡！從 buffalo_sc 改成 buffalo_l
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace 引擎初始化完成（buffalo_l 高精度模型）")

    # ==============================================
    # 資料庫管理
    # ==============================================

    def load_database(self):
        """從資料夾載入所有已知人臉"""
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            print(f"📁 已建立資料夾：{self.db_path}")
            return

        for person_name in os.listdir(self.db_path):
            person_path = os.path.join(self.db_path, person_name)
            if not os.path.isdir(person_path):
                continue

            print(f"📥 載入 {person_name} 的人臉特徵...")
            face_features = []

            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(person_path, img_file)
                    image = cv2.imread(img_path)
                    if image is None:
                        continue

                    # 偵測人臉 + 提取特徵（InsightFace 一步搞定）
                    faces = self.app.get(image)
                    if faces:
                        face_features.append(faces[0].embedding)
                        print(f"   ✅ 已載入 {img_file}")
                    else:
                        print(f"   ❌ 未偵測到人臉：{img_file}")

            if face_features:
                self.face_database[person_name] = np.mean(face_features, axis=0)
                print(f"   📊 {person_name}：共 {len(face_features)} 張圖片\n")
            else:
                print(f"   ⚠️ {person_name}：沒有有效的人臉圖片\n")

        print(f"🗄️ 資料庫載入完成，共 {len(self.face_database)} 人")
        print(f"   名單：{list(self.face_database.keys())}\n")

    def register_face(self, name, image):
        """註冊一張新的人臉"""
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
        """刪除已註冊的人臉"""
        if name not in self.face_database:
            return {"success": False, "message": f"找不到 {name}"}

        del self.face_database[name]
        person_path = os.path.join(self.db_path, name)
        if os.path.exists(person_path):
            shutil.rmtree(person_path)

        return {"success": True, "message": f"已刪除 {name}"}

    def get_registered_names(self):
        """取得所有已註冊的人名"""
        return list(self.face_database.keys())

    # ==============================================
    # 人���辨識核心
    # ==============================================

    @staticmethod
    def cosine_similarity(feature1, feature2):
        """計算餘弦相似度（跟你原本的 calculate_similarity 一樣）"""
        if feature1 is None or feature2 is None:
            return 0.0
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(feature1, feature2) / (norm1 * norm2))

    def recognize(self, image):
        """
        核心辨識函數：輸入一張圖片，回傳辨識結果
        回傳格式跟你原本的程式碼完全一致
        """
        if not self.face_database:
            return []

        faces = self.app.get(image)
        results = []

        for face in faces:
            bbox = face.bbox  # [x1, y1, x2, y2]
            current_feature = face.embedding

            # 跟資料庫比對（你原本的邏輯）
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
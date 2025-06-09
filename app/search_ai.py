import os
import json
import cv2
import insightface
import numpy as np
from database import get_db_connection

IMAGES_FOLDER = "/app/images_search"

# โหลดโมเดล InsightFace
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1)

# ---------- Cosine similarity ----------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

# ---------- อ่านภาพและสกัด embedding ----------
def get_embedding(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ ไม่พบภาพ: {image_path}")
        return None

    faces = model.get(img)
    if len(faces) == 0:
        print(f"😐 ไม่พบใบหน้าในภาพ: {image_path}")
        return None

    return faces[0].embedding.tolist()

# ---------- ค้นหาใบหน้าที่คล้ายที่สุด ----------
def find_most_similar_faces(embedding):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT fe.embedding, i.images_name, fe.images_id
        FROM face_embeddings fe
        JOIN images i ON fe.images_id = i.images_id
    """)
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    scored_results = []

    for row in results:
        try:
            db_embedding = json.loads(row["embedding"])
            score = cosine_similarity(embedding, db_embedding)
            scored_results.append({
                "matched_images_name": row["images_name"],
                "matched_images_id": row["images_id"],
                "similarity": round(score, 4)
            })
        except Exception as e:
            print(f"⚠️ Error comparing embedding: {e}")

    # เรียงจาก similarity มาก -> น้อย
    scored_results.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_results

# ---------- 🔁 สำหรับ API เรียกใช้ ----------
def perform_face_search(images_name: str):
    image_path = os.path.join(IMAGES_FOLDER, images_name)

    if not os.path.isfile(image_path):
        return {
            "images_name": images_name,
            "detect_images": False
        }

    embedding = get_embedding(image_path)
    if embedding is None:
        return {
            "images_name": images_name,
            "detect_images": True,
            "face_found": False
        }

    matches = find_most_similar_faces(embedding)
    return {
        "images_name": images_name,
        "detect_images": True,
        "face_found": True,
        "matches": matches  # <-- ทั้งหมด ไม่จำกัด 10
    }

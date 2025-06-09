import os
import json
import cv2
import insightface
import numpy as np
from database import get_db_connection

IMAGES_FOLDER = "/app/images_search"

# โหลดโมเดล InsightFace
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1)  # -1 หมายถึงใช้ CPU (0 ใช้ GPU ถ้ามี)

# คำนวณ Cosine similarity
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ดึง embedding จากภาพ
def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ ไม่พบไฟล์ภาพ: {image_path}")
        return None
    faces = model.get(img)
    if len(faces) == 0:
        print(f"❌ ไม่พบใบหน้าในภาพ: {image_path}")
        return None
    face = faces[0]  # ใช้ใบหน้าแรก
    return face.embedding.tolist()

# ค้นหาภาพที่มีใบหน้าคล้ายกันในฐานข้อมูล
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

    scored_results.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_results

# ฟังก์ชันหลักสำหรับ API / ใช้งานทั่วไป
def perform_face_search(images_name: str):
    image_path = os.path.join(IMAGES_FOLDER, images_name)

    if not os.path.isfile(image_path):
        return {
            "images_name": images_name,
            "detect_images": False,
            "face_found": False,
            "matches": []
        }

    embedding = get_embedding(image_path)
    if embedding is None:
        return {
            "images_name": images_name,
            "detect_images": True,
            "face_found": False,
            "matches": []
        }

    matches = find_most_similar_faces(embedding)
    return {
        "images_name": images_name,
        "detect_images": True,
        "face_found": True,
        "matches": matches
    }

# สำหรับทดสอบแบบ run ด้วยตัวเอง
if __name__ == "__main__":
    test_image = "test.jpg"  # เปลี่ยนชื่อภาพตามที่ต้องการ
    result = perform_face_search(test_image)
    print(json.dumps(result, indent=2, ensure_ascii=False))

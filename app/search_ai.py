import os
import json
import cv2
import insightface
import numpy as np
from database import get_db_connection

IMAGES_FOLDER = "/app/images_search"

model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1)


# ---------- Cosine similarity ----------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- Search most similar face ----------
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

    # เรียงจาก similarity มากไปน้อย (ไม่ตัดทิ้ง)
    scored_results.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_results

# ---------- 🔁 NEW: Function for direct API usage ----------
def perform_face_search(images_name: str):
    image_path = os.path.join("/app/images_search", images_name)

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
        # "embedding": embedding,
        "matches": matches
    }

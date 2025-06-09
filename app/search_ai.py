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
def find_most_similar_face(embedding):
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

    best_match = None
    best_score = -1

    for row in results:
        try:
            db_embedding = json.loads(row["embedding"])
            score = cosine_similarity(embedding, db_embedding)
            if score > best_score:
                best_score = score
                best_match = {
                    "matched_images_name": row["images_name"],
                    "matched_images_id": row["images_id"],
                    "similarity": round(score, 4)
                }
        except Exception as e:
            print(f"⚠️ Error comparing embedding: {e}")

    return best_match


def get_embedding(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    faces = model.get(img)
    if not faces:
        return None
    return faces[0].embedding.tolist()


# ---------- 🔁 NEW: Function for direct API usage ----------
def perform_face_search(images_name: str) -> dict:
    image_path = os.path.join(IMAGES_FOLDER, images_name)
    print(f"🔍 Looking for image at: {image_path}")

    if not os.path.isfile(image_path):
        print(f"❌ Image not found: {images_name}")
        return {
            "images_name": images_name,
            "detect_images": False
        }

    embedding = get_embedding(image_path)
    if embedding is None:
        print("⚠️ No face detected in image")
        return {
            "images_name": images_name,
            "detect_images": True,
            "face_found": False
        }

    print("✅ Face detected, embedding created")
    match = find_most_similar_face(embedding)
    if match:
        print(f"🔁 Match found: {match}")
    else:
        print("ℹ️ No similar face found")

    return {
        "images_name": images_name,
        "detect_images": True,
        "face_found": True,
        "embedding": embedding,
        "match": match  # None if not found
    }

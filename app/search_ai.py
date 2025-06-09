import os
import json
import cv2
import insightface
import numpy as np
from database import get_db_connection

IMAGES_FOLDER = "/app/images_search"
THRESHOLD = 0.65

model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(640, 640))

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = model.get(img)
    if len(faces) == 0:
        return None
    face = faces[0]
    return face.embedding.tolist()

def find_most_similar_faces(embedding):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT fe.embedding, i.images_name,i.images_preview_name,fe.images_id
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
            if score >= THRESHOLD:
                scored_results.append({
                    "matched_images_name": row["images_name"],
                    "matched_images_id": row["images_id"],
                    "images_preview_name": row["images_preview_name"],
                    "similarity": round(score, 4)
                })
        except Exception as e:
            print(f"⚠️ Error comparing embedding: {e}")

    scored_results.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_results

def perform_face_search(image_path: str):
    image_path = IMAGES_FOLDER + "/" + image_path
    print(f"testpath :{image_path}")
    if not os.path.isfile(image_path):
        return {
            "detect_images": False,
            "face_found": False,
            "matches": []
        }

    embedding = get_embedding(image_path)
    if embedding is None:
        return {
            "detect_images": True,
            "face_found": False,
            "matches": []
        }

    matches = find_most_similar_faces(embedding)
    return {
        "detect_images": True,
        "face_found": True,
        "matches": matches
    }
# def perform_face_search(image_path: str):
#     image_path = os.path.join(IMAGES_FOLDER, image_path)
#     if not os.path.isfile(image_path):
#         return []

#     embedding = get_embedding(image_path)
#     if embedding is None:
#         return []

#     matches = find_most_similar_faces(embedding)
#     return matches

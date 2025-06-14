import os
import json
import cv2
import insightface
import faiss
import numpy as np
from database import get_db_connection

IMAGES_FOLDER = "/app/images_search"
# THRESHOLD = 0.60
THRESHOLD = 0.4
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(480, 480))

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
def find_most_similar_faces(embedding, event_sub_id=0):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    base_query = """
        SELECT fe.embedding, i.images_name, i.images_preview_name, fe.images_id
        FROM face_embeddings fe
        JOIN images i ON fe.images_id = i.images_id
    """
    params = []
    if event_sub_id:
        base_query += " WHERE i.events_sub_id = %s"
        params.append(event_sub_id)

    cursor.execute(base_query, params)
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if not results:
        return []

    # โหลด embeddings และ metadata
    embeddings_list = []
    metadata = []
    for row in results:
        try:
            db_embedding = json.loads(row["embedding"])
            embeddings_list.append(db_embedding)
            metadata.append({
                "images_name": row["images_name"],
                "images_id": row["images_id"],
                "images_preview_name": row["images_preview_name"]
            })
        except Exception as e:
            print(f"⚠️ Error loading embedding: {e}")

    if not embeddings_list:
        return []

    # เตรียมข้อมูลสำหรับ FAISS
    db_embeddings_np = np.array(embeddings_list).astype("float32")
    query_embedding = np.array(embedding).astype("float32").reshape(1, -1)

    # Normalize เพื่อใช้ cosine similarity
    faiss.normalize_L2(db_embeddings_np)
    faiss.normalize_L2(query_embedding)

    index = faiss.IndexFlatIP(db_embeddings_np.shape[1])  # cosine similarity
    index.add(db_embeddings_np)

    # ดึงทั้งหมด (k = จำนวน embedding ทั้งหมด)
    k = db_embeddings_np.shape[0]
    D, I = index.search(query_embedding, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if score >= THRESHOLD:
            results.append({
                "matched_images_name": metadata[idx]["images_name"],
                "matched_images_id": metadata[idx]["images_id"],
                "images_preview_name": metadata[idx]["images_preview_name"],
                "similarity": round(float(score), 4)
            })

    return results
# เก่า def find_most_similar_faces(embedding, event_sub_id=0):
#     conn = get_db_connection()
#     cursor = conn.cursor(dictionary=True)

#     base_query = """
#         SELECT fe.embedding, i.images_name, i.images_preview_name, fe.images_id
#         FROM face_embeddings fe
#         JOIN images i ON fe.images_id = i.images_id
#     """

#     params = []
#     if event_sub_id:  # ถ้า event_sub_id มีค่า (ไม่ว่าง)
#         base_query += " WHERE i.events_sub_id = %s"
#         params.append(event_sub_id)

#     cursor.execute(base_query, params)
#     results = cursor.fetchall()
#     cursor.close()
#     conn.close()

#     scored_results = []
#     for row in results:
#         try:
#             db_embedding = json.loads(row["embedding"])
#             score = cosine_similarity(embedding, db_embedding)
#             if score >= THRESHOLD:
#                 scored_results.append({
#                     "matched_images_name": row["images_name"],
#                     "matched_images_id": row["images_id"],
#                     "images_preview_name": row["images_preview_name"],
#                     "similarity": round(score, 4)
#                 })
#         except Exception as e:
#             print(f"⚠️ Error comparing embedding: {e}")

#     scored_results.sort(key=lambda x: x["similarity"], reverse=True)
#     return scored_results

# def find_most_similar_faces(embedding,event_sub_id):
    
#     conn = get_db_connection()
#     cursor = conn.cursor(dictionary=True)
#     cursor.execute("""
#         SELECT fe.embedding, i.images_name,i.images_preview_name,fe.images_id
#         FROM face_embeddings fe
#         JOIN images i ON fe.images_id = i.images_id
#     """)
#     results = cursor.fetchall()
#     cursor.close()
#     conn.close()

#     scored_results = []
#     for row in results:
#         try:
#             db_embedding = json.loads(row["embedding"])
#             score = cosine_similarity(embedding, db_embedding)
#             if score >= THRESHOLD:
#                 scored_results.append({
#                     "matched_images_name": row["images_name"],
#                     "matched_images_id": row["images_id"],
#                     "images_preview_name": row["images_preview_name"],
#                     "similarity": round(score, 4)
#                 })
#         except Exception as e:
#             print(f"⚠️ Error comparing embedding: {e}")

#     scored_results.sort(key=lambda x: x["similarity"], reverse=True)
#     return scored_results

def perform_face_search(image_path: str, events_sub_id: int):
    full_path = os.path.join(IMAGES_FOLDER, image_path)
    print(f"testpath : {full_path}")

    if not os.path.isfile(full_path):
        return {
            "detect_images": False,
            "face_found": False,
            "matches": []
        }

    embedding = get_embedding(full_path)
    if embedding is None:
        return {
            "detect_images": True,
            "face_found": False,
            "matches": []
        }

    #  embed_search ในตาราง search_image
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        update_query = """
            UPDATE search_image
            SET embed_search = %s
            WHERE search_image_name = %s
        """
        cursor.execute(update_query, (json.dumps(embedding), image_path))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ Failed to update embed_search: {e}")

    matches = find_most_similar_faces(embedding, events_sub_id)
    return {
        "detect_images": True,
        "face_found": True,
        "embedding": embedding,
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

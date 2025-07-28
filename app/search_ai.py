import os
import json
import cv2
import insightface
import faiss
import numpy as np
from functools import lru_cache
from database import get_db_connection  # <-- แก้ให้ตรงกับของคุณ

IMAGES_FOLDER = "/app/images_search"
THRESHOLD = 0.4

model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(480, 480))


def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = model.get(img)
    if len(faces) == 0:
        return None
    face = faces[0]
    return face.embedding.tolist()


@lru_cache(maxsize=10)
def get_faiss_index_cached(event_sub_id: int = None, event_id: int = None):
    if event_sub_id:
        print(f"⏳ Loading FAISS index for event_sub_id: {event_sub_id}")
    elif event_id:
        print(f"⏳ Loading FAISS index for event_id: {event_id}")
    else:
        print("⚠️ Must provide either event_sub_id or event_id")
        return None, []

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
         SELECT i.events_sub_id,evs.events_id,fe.embedding, i.images_name, i.images_preview_name, fe.images_id
        FROM face_embeddings fe
        JOIN images i ON fe.images_id = i.images_id
		JOIN events_sub evs ON evs.events_sub_id=i.events_sub_id
        WHERE {}
    """.format("i.events_sub_id = %s" if event_sub_id else "evs.events_id = %s")

    cursor.execute(query, (event_sub_id or event_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        return None, []

    embeddings = []
    metadata = []

    for row in rows:
        try:
            embedding = json.loads(row["embedding"])
            embeddings.append(embedding)
            metadata.append({
                "images_name": row["images_name"],
                "images_id": row["images_id"],
                "images_preview_name": row["images_preview_name"]
            })
        except:
            continue

    np_embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(np_embeddings)

    index = faiss.IndexFlatIP(np_embeddings.shape[1])
    index.add(np_embeddings)

    return index, metadata

def invalidate_faiss_cache(event_sub_id: int):
    get_faiss_index_cached.cache_clear()
    print(f"🧹 Cleared FAISS cache (all entries). Reload on next use.")

def find_most_similar_faces(embedding, event_sub_id: int = None, event_id: int = None):
    index, metadata = get_faiss_index_cached(event_sub_id=event_sub_id, event_id=event_id)
    if index is None:
        return []

    query_vec = np.array(embedding).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vec)

    D, I = index.search(query_vec, len(metadata))
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


def perform_face_search(image_path: str, event_sub_id: int = None, event_id: int = None):
    full_path = os.path.join(IMAGES_FOLDER, image_path)
    print(f"🔍 Searching face for image: {full_path}")

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

    # Save search embedding to DB
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

    matches = find_most_similar_faces(
        embedding,
        event_sub_id=event_sub_id,
        event_id=event_id
    )  
    return {
        "detect_images": True,
        "face_found": True,
        "embedding": embedding,
        "matches": matches
    }
import os
import aio_pika
import asyncio
import json
import cv2
import insightface
import numpy as np
from database import get_db_connection

IMAGES_FOLDER = "/app/images_search"  # แมป docker volume ให้ถูกต้อง

model = insightface.app.FaceAnalysis(name="buffalo_l")  # รุ่นใหญ่ (large)
model.prepare(ctx_id=-1)


# ---------- Cosine similarity ----------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- Search most similar face ----------
def find_most_similar_face(embedding, event_sub_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT fe.embedding, i.images_name, fe.images_id
        FROM face_embeddings fe
        JOIN images i ON fe.images_id = i.images_id
        WHERE i.events_sub_id = %s
    """, (event_sub_id,))
    
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
        return None  # ไม่เจอหน้า
    return faces[0].embedding.tolist()  # แปลงเป็น list เพื่อส่ง JSON

# ---------- Main worker ----------
async def face_search_worker():
    connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq/")
    input_queue = "face_search_tasks"
    embedding_queue = "face_embeddings_tasks"

    async with connection:
        channel = await connection.channel()
        await channel.declare_queue(input_queue, durable=True)
        await channel.declare_queue(embedding_queue, durable=True)

        queue = await channel.declare_queue(input_queue, durable=True)
        print(f"📥 Waiting for messages on queue: {input_queue}")

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        data = json.loads(message.body.decode())
                        images_name = data.get("images_name").strip()
                        event_sub_id = data.get("event_sub_id")
                        image_path = os.path.join(IMAGES_FOLDER, images_name)

                        print(f"🔍 Looking for image at: {image_path}")

                        if not os.path.isfile(image_path):
                            print(f"❌ Image not found: {images_name}")
                            response = {
                                "images_name": images_name,
                                "event_sub_id": event_sub_id,
                                "detect_images": False,
                            }
                        else:
                            embedding = get_embedding(image_path)
                            if embedding is None:
                                print("⚠️ No face detected in image")
                                response = {
                                    "images_name": images_name,
                                    "event_sub_id": event_sub_id,
                                    "detect_images": True,
                                    "face_found": False
                                }
                            else:
                                print("✅ Face detected, embedding created")

                                match = find_most_similar_face(embedding, event_sub_id)
                                if match:
                                    print(f"🔁 Match found: {match}")
                                else:
                                    print("ℹ️ No similar face found")

                                response = {
                                    "images_name": images_name,
                                    "event_sub_id": event_sub_id,
                                    "detect_images": True,
                                    "face_found": True,
                                    "embedding": embedding,
                                    "match": match  # None if not found
                                }

                        response_body = json.dumps(response).encode()
                        await channel.default_exchange.publish(
                            aio_pika.Message(body=response_body),
                            routing_key=embedding_queue
                        )
                        print(f"✅ Sent result to '{embedding_queue}': {response}")

                    except Exception as e:
                        print(f"❌ Error processing message: {e}")

if __name__ == "__main__":
    print("🚀 Starting Face Search Worker...")
    asyncio.run(face_search_worker())
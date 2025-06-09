import os
import aio_pika
import asyncio
import json
import cv2
import insightface
from database import get_db_connection

model = insightface.app.FaceAnalysis(name="buffalo_l")  # รุ่นใหญ่ (large)
model.prepare(ctx_id=-1, det_size=(800, 800))  # กำหนดขนาดภาพสำหรับ detector ให้ใหญ่ขึ้น
model.threshold = 0.4 
def process_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    faces = model.get(img)
    return faces

async def rabbitmq_consumer():
    connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq/")
    queue_name = "image_tasks"

    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(queue_name, durable=True)
        print(f"📥 Waiting for messages on queue: {queue_name}")

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    data = json.loads(message.body.decode())
                    images_id = data.get("images_id")
                    images_name = data.get("images_name")

                    conn = get_db_connection()
                    cursor = conn.cursor()

                    # Update process_status_id = 2 (start processing)
                    cursor.execute("""
                        UPDATE images SET process_status_id = %s WHERE images_id = %s
                    """, (2, images_id))
                    conn.commit()

                    image_path = os.path.join("/images", images_name)
                    print(f"🖼️ Processing image at: {image_path}")

                    faces = process_image(image_path)
                    print(f"👤 Faces detected: {len(faces)}")

                    # Clear old embeddings if any
                    cursor.execute("DELETE FROM face_embeddings WHERE images_id = %s", (images_id,))

                    for face in faces:
                        embedding_list = face.embedding.tolist()  # แปลงเป็น list
                        embedding_json = json.dumps(embedding_list)  # แปลงเป็น JSON string
                        cursor.execute("""
                            INSERT INTO face_embeddings (images_id, embedding)
                            VALUES (%s, %s)
                        """, (images_id, embedding_json))

                    # Update process_status_id = 3 (done)
                    cursor.execute("""
                        UPDATE images SET process_status_id = %s ,faces= %s WHERE images_id = %s
                    """, (3,len(faces), images_id))

                    conn.commit()
                    cursor.close()
                    conn.close()

if __name__ == "__main__":
    print("Starting RabbitMQ consumer...")
    asyncio.run(rabbitmq_consumer())

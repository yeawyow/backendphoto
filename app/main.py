from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import aio_pika
from database import get_db_connection

app = FastAPI()

rabbit_conn = None
image_channel = None
face_search_channel = None

class ImageRequest(BaseModel):
    filename: str

class SearchRequest(BaseModel):
    images_name: str
    event_sub_id: int
rabbit_conn = None
image_channel = None
face_search_channel = None
@app.on_event("startup")
async def startup_event():
    global rabbit_conn, image_channel, face_search_channel

    # สร้าง connection ครั้งเดียว
    rabbit_conn = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq/")

    # สร้าง channel แยก สำหรับ image task
    image_channel = await rabbit_conn.channel()
    await image_channel.declare_queue("image_tasks", durable=True)

    # สร้าง channel แยก สำหรับ face search task
    face_search_channel = await rabbit_conn.channel()
    await face_search_channel.declare_queue("face_search_tasks", durable=True)

    print("✅ RabbitMQs connected and channels declared")

@app.on_event("shutdown")
async def shutdown_event():
    global rabbit_conn
    if rabbit_conn:
        await rabbit_conn.close()
        print("🔌 RabbitMQ disconnected")

async def send_to_rabbitmq(payload: dict, queue_name: str):
    global image_channel, face_search_channel

    # เลือก channel ตาม queue_name
    if queue_name == "image_tasks":
        channel = image_channel
    elif queue_name == "face_search_tasks":
        channel = face_search_channel
    else:
        raise ValueError(f"Unknown queue: {queue_name}")

    if not channel or channel.is_closed:
        raise RuntimeError(f"RabbitMQ channel for queue '{queue_name}' is not initialized or closed")

    try:
        message_body = json.dumps(payload).encode()
        await channel.default_exchange.publish(
            aio_pika.Message(body=message_body),
            routing_key=queue_name
        )
    except Exception as e:
        print(f"Failed to send message to RabbitMQ on queue {queue_name}: {e}")
        raise

@app.post("/process-image")
async def process_image_api(request: ImageRequest):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT images_id, images_name 
            FROM images 
            WHERE images_name = %s AND process_status_id = 1
        """, (request.filename,))
        image = cursor.fetchone()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found or not in a valid processing state.")

        await send_to_rabbitmq({
            "images_id": image["images_id"],
            "images_name": image["images_name"]
        }, queue_name="image_tasks")

        return {
            "status": "queued",
            "message": "Image task has been sent to RabbitMQ.",
            "data": image
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

# @app.post("/face-search")
# async def face_search(req: SearchRequest):
#     await send_to_rabbitmq({
#         "images_name": req.images_name,
#         "event_sub_id": req.event_sub_id
#     }, queue_name="face_search_tasks")
#     return JSONResponse(content={
#         "status": "queued",
#         "images_name": req.images_name
#     })
    @router.post("/face-search")
async def face_search(req: SearchRequest):
    result = perform_face_search(req.images_name)
    return JSONResponse(content={
        "status": "completed",
        "result": result
    })

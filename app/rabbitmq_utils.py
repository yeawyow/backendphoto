import aio_pika
import asyncio
import json

rabbit_conn = None
rabbit_channel = None

async def connect_rabbitmq(url: str = "amqp://guest:guest@rabbitmq/"):
    global rabbit_conn, rabbit_channel
    rabbit_conn = await aio_pika.connect_robust(url)
    rabbit_channel = await rabbit_conn.channel()
    print("✅ RabbitMQ connected")

async def close_rabbitmq():
    global rabbit_conn
    if rabbit_conn:
        await rabbit_conn.close()
        print("🔌 RabbitMQ disconnected")

async def send_to_rabbitmq(payload: dict, routing_key: str = "image_tasks"):
    global rabbit_channel
    if not rabbit_channel:
        raise RuntimeError("RabbitMQ channel is not initialized")
    message_body = json.dumps(payload).encode()
    await rabbit_channel.default_exchange.publish(
        aio_pika.Message(body=message_body),
        routing_key=routing_key
    )

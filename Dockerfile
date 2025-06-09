FROM python:3.11-slim

# ติดตั้งระบบเบื้องต้น
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements.txt แล้วติดตั้ง dependencies
COPY app/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy โค้ดเข้า container
COPY app /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

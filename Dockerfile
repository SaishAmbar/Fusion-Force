FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Run training on build to generate results.png
RUN python train.py

# Environment variables for inference (override at runtime)
ENV API_BASE_URL=https://api-inference.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
# HF_TOKEN must be provided at runtime via Space Secrets

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
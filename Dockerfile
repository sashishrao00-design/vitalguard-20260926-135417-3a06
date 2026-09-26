FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build a fresh synthetic demo model with the installed sklearn version.
RUN python train.py --n 6000 --seed 42 && mkdir -p data
EXPOSE 8000
CMD ["sh", "-c", "exec python -m uvicorn vitalguard.server:app --host 0.0.0.0 --port ${PORT:-8000}"]

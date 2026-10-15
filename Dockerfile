FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
# CPU-only torch keeps the image small
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
# bake the embedding model into the image so first start needs no internet
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
COPY . .
EXPOSE 8000
CMD ["uvicorn", "docq.api:app", "--host", "0.0.0.0", "--port", "8000"]

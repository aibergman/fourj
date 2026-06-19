FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app
COPY pyproject.toml README.md MANIFEST.in ./
COPY src ./src
COPY fourj.png ./fourj.png
COPY app.py ./app.py

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[dashboard]

EXPOSE 7860
CMD ["python", "app.py"]

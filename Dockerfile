FROM paddlepaddle/paddle:3.3.0-gpu-cuda12.9-cudnn9.9

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt requirements-paddle.txt ./
RUN python -m pip install --no-cache-dir -r requirements-paddle.txt

COPY app ./app
COPY models ./models
COPY README.md ./README.md

EXPOSE 8800

CMD ["python", "-m", "app"]


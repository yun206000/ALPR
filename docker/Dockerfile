# ============================================================
# Stage 1: builder — 安裝依賴
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# 系統依賴（OpenCV headless 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 升級 pip
RUN pip install --upgrade pip

COPY requirements.txt .

# 安裝 Python 依賴（不使用快取）
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        git+https://github.com/roboflow/rf-detr.git \
        git+https://github.com/baudm/parseq.git

# ============================================================
# Stage 2: runtime — 最小化映像
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="your-team@example.com"
LABEL description="LPR Pipeline — RF-DETR + PARSeq 車牌辨識系統"
LABEL version="1.0.0"

# 建立非 root 使用者
RUN groupadd -r lpr && useradd -r -g lpr lpr

WORKDIR /app

# 系統執行期依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 從 builder 複製 Python 套件
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 複製應用程式碼
COPY src/ ./src/
COPY main.py .

# 建立必要目錄，設定權限
RUN mkdir -p models outputs data/samples \
    && chown -R lpr:lpr /app

# 複製 entrypoint 腳本
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 預設環境變數（可在 docker run 或 compose 中覆寫）
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DETECTOR_DEVICE=cpu
ENV RECOGNIZER_DEVICE=cpu
ENV LOG_LEVEL=INFO

# 模型掛載點
VOLUME ["/app/models", "/app/outputs", "/app/data"]

EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 使用 entrypoint 腳本（運行為 root，script 會切換到 lpr 使用者）
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py", "--serve", "--host", "0.0.0.0", "--port", "8000"]

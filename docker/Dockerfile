# ============================================================
# ALPR — 以 conda「ALPR」環境（environment.yml）建置的映像
#   CPU / GPU 共用同一個映像：torch 2.11 (cu130) 的 pip wheel 已內含 CUDA runtime，
#   只要主機有 NVIDIA driver + nvidia-container-toolkit，容器就能用 GPU。
#
# 兩個 target：
#   runtime（compose 使用）：只有環境與程式碼，權重 / 資料以 volume 掛載
#   bundle：runtime + 模型權重 + 預訓練權重，不需要掛載任何東西就能推論、離線訓練
#     docker build -f docker/Dockerfile --target bundle -t alpr-bundle:1.0.0 .
# ============================================================
ARG BASE_IMAGE=continuumio/miniconda3:latest
FROM ${BASE_IMAGE} AS runtime

# 非互動接受 Anaconda channel 的 ToS；建置時不留 pip 快取
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes \
    PIP_NO_CACHE_DIR=1

# OpenCV（opencv-python 非 headless 版）需要的系統函式庫
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- 建立 conda 環境（先只複製 environment.yml，讓這層可以被快取）----
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && rm -f /tmp/environment.yml

# 之後所有指令都直接使用 ALPR 環境的 python
ENV PATH=/opt/conda/envs/ALPR/bin:$PATH \
    CONDA_DEFAULT_ENV=ALPR \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/opt/torch \
    YOLO_CONFIG_DIR=/opt/ultralytics \
    MPLCONFIGDIR=/opt/matplotlib \
    HOME=/tmp

# compose 用主機的 UID 執行容器，這個 UID 在容器的 /etc/passwd 裡不存在，
# 會讓 getpass.getuser()（torch 啟動時會呼叫）丟出 KeyError。
# getuser() 會先讀這兩個環境變數，有設就不會去查 /etc/passwd。
ENV USER=alpr \
    LOGNAME=alpr

# 預先下載 PARSeq 的 torch.hub 程式碼（只抓程式碼，不抓預訓練權重），
# 讓容器在離線時也能載入 models/parseq/ 裡的權重；並先產生 Ultralytics 設定檔與 matplotlib 快取。
# 權限開放給所有使用者，因為 compose 會用主機的 UID 執行容器。
RUN mkdir -p /opt/torch /opt/ultralytics /opt/matplotlib \
    && python -c "import torch; torch.hub.load('baudm/parseq:main', 'parseq', pretrained=False, trust_repo=True)" \
    && python -c "import ultralytics, matplotlib.pyplot" \
    && chmod -R a+rwX /opt/torch /opt/ultralytics /opt/matplotlib

WORKDIR /app

# ---- 專案程式碼 ----
COPY src/ ./src/
COPY main.py config.yaml ./

# 權重 / 資料 / 輸出都以 volume 掛載，不打包進映像
RUN mkdir -p models data outputs logs && chmod -R a+rwX /app

# 預設不以 root 執行（直接 docker run 時也一樣）；compose 會再用 user: 換成主機的 UID/GID
USER 1000:1000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 預設啟動 API Server；也可以覆寫成任何指令，例如：
#   docker compose run --rm alpr python main.py --image data/xxx.jpg
#   docker compose run --rm alpr python src/train/train.py ...
CMD ["python", "main.py", "--serve", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================
# bundle：映像內含模型權重（推論用 + fine-tune / 訓練的起點）
# ============================================================
FROM runtime AS bundle

ARG VERSION=1.0.0
LABEL org.opencontainers.image.title="ALPR" \
      org.opencontainers.image.description="台灣車牌辨識（YOLOv8n 車牌偵測 + PARSeq 字元辨識），內含模型權重" \
      org.opencontainers.image.version="${VERSION}"

USER root

# PARSeq 官方預訓練權重（train_parseq.finetune_from 為 null 時的起點），讓容器離線也能訓練
RUN python -c "import torch; torch.hub.load('baudm/parseq:main', 'parseq', pretrained=True, trust_repo=True)" \
    && chmod -R a+rwX /opt/torch/hub/checkpoints

# 推論用權重（config.yaml 的 detector / recognizer.weights_path）與 YOLO fine-tune 的起點。
# 資料夾開放給所有使用者寫入：容器可能用主機的 UID 執行，訓練完成時要能取代 models/ 底下的資料夾
# （只改資料夾權限；在另一層改檔案權限會讓權重檔在映像裡多存一份）
COPY --chmod=644 models/yolo/finetuned/best.pt                     models/yolo/finetuned/best.pt
COPY --chmod=644 models/parseq/finetuned/best.ckpt                 models/parseq/finetuned/best.ckpt
COPY --chmod=644 models/yolo/pretrained/license_plate_detector.pt  models/yolo/pretrained/license_plate_detector.pt
RUN find /app/models -type d -exec chmod a+rwx {} +

COPY --chmod=644 MODEL_CARD.md ./

USER 1000:1000

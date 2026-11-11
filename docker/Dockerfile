# ============================================================
# ALPR — 以 conda「ALPR」環境（environment.yml）建置的映像
#   CPU / GPU 共用同一個映像：torch 2.11 (cu130) 的 pip wheel 已內含 CUDA runtime，
#   只要主機有 NVIDIA driver + nvidia-container-toolkit，容器就能用 GPU。
# ============================================================
ARG BASE_IMAGE=continuumio/miniconda3:latest
FROM ${BASE_IMAGE}

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

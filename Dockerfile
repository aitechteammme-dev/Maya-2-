# syntax=docker/dockerfile:1.7
# MME Maya 2 Native (maya3 engine) - RunPod custom container. linux/amd64, CUDA 12.8 wheels, no GPU needed to build.
#
#   docker buildx build --platform linux/amd64 -t YOUR_DOCKER_USERNAME/mme-maya2-native:v1 --load .
#
# NOT build-tested by the author of this file (no Docker daemon was available) - see BUILD_REPORT.md.

ARG CUDA_BASE=nvidia/cuda:12.8.1-base-ubuntu22.04
# Upstream engine commit audited in docs/UPSTREAM_AUDIT.md
ARG MAYA3_REPO=https://github.com/MayaResearch/maya2-native.git
ARG MAYA3_REF=c6b376868e268551d065c4206193e96257bef4a1

# ---------------------------------------------------------------- builder: venv + wheels + upstream source
FROM ${CUDA_BASE} AS builder
ARG MAYA3_REPO
ARG MAYA3_REF
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Deterministic, hash-checked install of the resolved lock (torch 2.8.0+cu128 / triton 3.4.0 come from the cu128 index).
COPY requirements-lock.txt /tmp/requirements-lock.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/requirements-lock.txt

# Upstream engine, pinned to an exact commit, installed WITHOUT re-resolving dependencies.
RUN git init /src/maya2-native \
    && cd /src/maya2-native \
    && git remote add origin "${MAYA3_REPO}" \
    && git fetch --depth 1 origin "${MAYA3_REF}" \
    && git checkout FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${MAYA3_REF}"
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps /src/maya2-native

# ---------------------------------------------------------------- runtime
FROM ${CUDA_BASE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
# libsndfile1: soundfile (wav/flac/mp3/opus encode). gcc + python headers: Triton JIT-compiles its launchers at first use.
# tini: PID 1 signal forwarding. FFmpeg is NOT needed (upstream encodes with libsndfile).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev libsndfile1 gcc libc6-dev tini ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin mme

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /src/maya2-native/LICENSE /licenses/maya2-native.LICENSE

WORKDIR /app
COPY server/ /app/server/
COPY scripts/ /app/scripts/
COPY config/ /app/config/
RUN chmod 0755 /app/scripts/entrypoint.sh

# Defaults only - NO credentials. MME_API_KEYS (and optionally MME_ADMIN_KEYS) must be provided at runtime.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MME_PROFILE=production \
    MODEL_ID=maya-research/maya2-native \
    HF_HOME=/workspace/.cache/huggingface \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    HOST=0.0.0.0 \
    PORT=8080 \
    PORT_HEALTH=8080 \
    MAX_SEATS=1 \
    MAX_PENDING=4 \
    LOG_LEVEL=INFO

# Build-time smoke test (CPU only, no model): the wrapper and the pinned upstream both import.
RUN python -c "import server.api, maya3.server, maya3.hub; print('import ok')"

EXPOSE 8080
VOLUME ["/workspace"]
STOPSIGNAL SIGTERM
# Healthy only when /ping == 200 (model loaded and engine ready). Generous start period for download + warm-up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 CMD ["python", "/app/scripts/health_check.py"]

# Starts as root only to fix cache-volume ownership, then drops to uid 10001 (see scripts/entrypoint.sh).
ENTRYPOINT ["/usr/bin/tini", "--", "/app/scripts/entrypoint.sh"]

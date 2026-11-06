# mme-maya2-native-runpod

Production-oriented container for **Maya 2 Native** (Hindi / Hinglish / English TTS, `maya-research/maya2-native`) as the TTS engine of the MME AI voice-calling SaaS, targeting RunPod. It wraps the **unmodified** upstream engine ([MayaResearch/maya2-native](https://github.com/MayaResearch/maya2-native), commit `c6b3768`) with authentication, `/ping`, metrics, tenant isolation, fail-closed startup and a chunk-level incremental-text endpoint.

> **Status: built and CPU-tested only.** The Docker image has **not** been built (no Docker daemon was available), and no GPU inference has ever been run. Latency, Hindi/Hinglish quality, voice-clone similarity, concurrency and RunPod proxy/WebSocket behaviour are **unverified**. See [BUILD_REPORT.md](BUILD_REPORT.md) and [docs/PERFORMANCE_TESTING.md](docs/PERFORMANCE_TESTING.md).

Key facts (details in [docs/UPSTREAM_AUDIT.md](docs/UPSTREAM_AUDIT.md)):
- Upstream streams **audio** incrementally but takes **complete text per request** - there is no token-level text streaming. We add a sentence/clause chunker (`WS /v1/tts/incremental`), see [docs/STREAMING_ARCHITECTURE.md](docs/STREAMING_ARCHITECTURE.md).
- Upstream has **no authentication and no watermark**; this repo adds auth and neither adds nor removes watermarking.
- Output is 24 kHz mono PCM s16le; no resampling is done here.

## Layout
```
Dockerfile  requirements.in / requirements-lock.txt  config/{development,production,benchmark}.yaml
server/     api.py (server + runner)  auth.py  config.py  health.py  metrics.py  model_store.py
            voice_manager.py  text_chunker.py  incremental.py  errors.py
scripts/    entrypoint.sh  startup_check.py  health_check.py  download_model.py  register_voice.py
tests/      CPU-only suite (uses upstream's FakeEngine)         docs/  audit, API, streaming, RunPod, performance
```

## Setup commands

```bash
# 1. clone this repo, then the upstream source (needed only for CPU tests / audit; the image fetches its own pinned copy)
git clone https://github.com/aitechteammme-dev/Maya-2-.git mme-maya2-native-runpod && cd mme-maya2-native-runpod
git clone https://github.com/MayaResearch/maya2-native upstream_src/maya2-native
git -C upstream_src/maya2-native checkout c6b376868e268551d065c4206193e96257bef4a1

# 2. CPU tests (Python 3.10-3.12; no torch/GPU needed)
python -m pip install aiohttp pytest pytest-asyncio pytest-aiohttp pytest-timeout soundfile numpy pyyaml huggingface_hub
python -m pytest tests

# 3. static checks
bash -n scripts/entrypoint.sh && python -m compileall -q server scripts

# 4. build (needs Docker with BuildKit; no GPU required)
docker buildx build --platform linux/amd64 -t YOUR_DOCKER_USERNAME/mme-maya2-native:v1 --load .

# 5. inspect
docker image ls YOUR_DOCKER_USERNAME/mme-maya2-native:v1
docker run --rm --entrypoint python YOUR_DOCKER_USERNAME/mme-maya2-native:v1 -c "import server.api, maya3.server; print('ok')"

# 6. local smoke WITHOUT a GPU (expected: /ping 503 and exit 1 - proves fail-closed, not inference)
docker run --rm -e MME_API_KEYS=local-test-key-0123456789 -e MME_SKIP_GPU_CHECK=1 -e MAYA_DEVICE=cpu -e HF_HUB_OFFLINE=1 \
  -p 8080:8080 YOUR_DOCKER_USERNAME/mme-maya2-native:v1

# 7. tag / login / push  (only when you decide to)
docker tag  YOUR_DOCKER_USERNAME/mme-maya2-native:v1 YOUR_DOCKER_USERNAME/mme-maya2-native:latest
docker login -u YOUR_DOCKER_USERNAME
docker push YOUR_DOCKER_USERNAME/mme-maya2-native:v1
```

### Run on a GPU host
```bash
docker run --gpus all -p 8080:8080 -v mme_cache:/workspace \
  -e MME_API_KEYS="mme=$(openssl rand -hex 24)" YOUR_DOCKER_USERNAME/mme-maya2-native:v1
curl -i localhost:8080/ping        # 204 while loading, 200 when ready
curl -sN localhost:8080/v1/audio/speech -H "X-MME-Api-Key: <token>" -H 'content-type: application/json' \
  -d '{"input":"आपका ऑर्डर कल पहुँच जाएगा।","language":"hi","voice":"Ahana","response_format":"wav"}' -o out.wav
```
Serving command used inside the container: `python -m server.api` (see [API_REFERENCE.md](docs/API_REFERENCE.md) for env vars). It is the wrapper around upstream's `python -m maya3.server`, not that CLI itself.

### Regenerating the lock
```bash
python -m pip install uv
uv pip compile requirements.in --python-version 3.10 --python-platform x86_64-manylinux_2_28 \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --generate-hashes -o requirements-lock.txt
```
The lock is *resolved* (torch 2.8.0+cu128, triton 3.4.0, transformers 5.6.2, ...) but its install inside the image has not been run.

## Security notes
No credentials are stored in the repo or image (`.env`, audio files and `upstream_src/` are git/docker-ignored). `MME_API_KEYS` is mandatory; the production profile refuses to run without auth. Reference voice recordings live on a mounted volume only; use `scripts/register_voice.py`. Use cloning only with the speaker's consent and disclose synthetic speech to listeners (upstream's responsible-use terms).

## Docs
[UPSTREAM_AUDIT](docs/UPSTREAM_AUDIT.md) · [API_REFERENCE](docs/API_REFERENCE.md) · [STREAMING_ARCHITECTURE](docs/STREAMING_ARCHITECTURE.md) · [RUNPOD_DEPLOYMENT](docs/RUNPOD_DEPLOYMENT.md) · [PERFORMANCE_TESTING](docs/PERFORMANCE_TESTING.md) · [BUILD_REPORT](BUILD_REPORT.md)

License: wrapper code for MME internal use; upstream engine and weights are Apache-2.0 (license copied into the image at `/licenses`).

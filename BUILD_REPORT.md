# Build report

Date: 2026-11-06. Environment: Windows 11, Python 3.11, aiohttp 3.14.5, **no Docker** (not on PATH in Windows or WSL Ubuntu-24.04), **no GPU**, torch not installed.

## What was verified (actually executed)
| Check | Result |
|---|---|
| CPU test suite `python -m pytest tests` | **84 passed** (auth, validation, health states, fail-closed load, WS lifecycle for upstream and incremental routes, chunker incl. 200-run no-loss property test, voice/audio validation, model-file verification, runner lifecycle: separate health port, SIGTERM-equivalent shutdown cancelling in-flight work, load-failure exit 1) |
| Real process smoke (`python -m server.api`, offline, empty cache, CPU) | `/ping` 503, `/health` status `failed` with clear error, unauthenticated speech 401, process exit code 1 |
| `startup_check.py` with empty offline cache | exit 1 with message |
| `bash -n scripts/entrypoint.sh` | syntax ok |
| Lock resolution (`uv pip compile`, py3.10, manylinux_2_28, cu128 index) | succeeded: torch 2.8.0+cu128, torchaudio 2.8.0+cu128, triton 3.4.0, transformers 5.6.2, numpy 2.2.6, aiohttp 3.14.5, nvidia-cuda-runtime-cu12 12.8.90, nvidia-cudnn-cu12 9.10.2.21 (2070 lines, hashed) |
| CUDA base image tags exist on Docker Hub | `nvidia/cuda:12.8.1-base-ubuntu22.04` -> HTTP 200 |

## NOT verified
- **`docker buildx build` has not been run.** The Dockerfile is unbuilt: `pip install --require-hashes` of the lock with the cu128 extra index, non-editable `pip install --no-deps` of upstream (and its namespace packages `serve`, `scripts`, `configs`, `tts`), Triton JIT needing `gcc` + `python3.10-dev` at runtime, the `setpriv` privilege drop on RunPod volumes, and image size are all unconfirmed. **Image size: unknown** (expect several GB because of torch + CUDA wheels; do not rely on this estimate).
- Any GPU inference, TTFA/RTF, GPU memory, Hindi/Hinglish quality, voice-clone similarity to Chatterbox, concurrency.
- RunPod proxy behaviour (header forwarding, WebSocket lifetime, buffering), volume mount path, scale-in behaviour.
- Mimi / ECAPA downloads (unpinned upstream) and the offline-first cache path with real files.
- Chunk-seam prosody of the incremental adapter.
- Telephony resampling and the approved watermarking stage (outside this repo; upstream has no watermark).

## Spend / side effects
No GPU credits spent, nothing deployed, no production endpoint touched, no calls made, no weights downloaded, no HF/RunPod credentials used.

## Remaining before a RunPod deployment
1. Build the image on a machine with Docker; fix whatever the first build reveals; record the size here.
2. Pre-populate the network volume (`scripts/download_model.py`), run `docs/PERFORMANCE_TESTING.md` G1-G14 on the target GPU.
3. Verify the auth header path through the RunPod proxy (G13) and WebSocket stability (G5); decide Path A vs B.
4. Register the Utkarsh voice with consent and run the A/B similarity test (G4).
5. Decide chunker thresholds from listening tests (G6); consider prefetching the next chunk.
6. Pin Mimi/ECAPA revisions by baking them into the volume and running with `HF_HUB_OFFLINE=1`.

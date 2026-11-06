# RunPod deployment guide

> Nothing in this repo deploys anything. These are manual steps. **Not yet executed**: every RunPod-specific behaviour below is from RunPod's public docs (https://docs.runpod.io/serverless/load-balancing/overview) or is marked UNVERIFIED and must be checked during the first deployment test.

## Path A - Serverless, Load Balancing endpoint (persistent HTTP + WebSocket)

| Setting | Value |
|---|---|
| Image | `YOUR_DOCKER_USERNAME/mme-maya2-native:v1` |
| Endpoint type | Load balancing (not queue-based; there is no queue handler in this image) |
| Expose HTTP port | `8080` (must equal `PORT`) |
| `PORT` / `PORT_HEALTH` | `8080` / `8080` (RunPod defaults both to 80 separately: set both) |
| Health check path | `/ping` (RunPod default). 200 = healthy, 204 = initializing, anything else = unhealthy |
| GPU | one with >= 16 GB for `MAX_SEATS` 1-16 (upstream memory table: 7.7-11.7 GB steady + ~0.5 GB context; 9.2-12.8 GB peak during load). Choose CUDA 12.8-capable hosts (allowed-CUDA-version filter, if offered) |
| Container disk | >= 25 GB (image + Triton cache); model cache lives on the volume |
| Network volume | mount so that `HF_HOME` points into it (see below) |
| Workers | start with min 1 (avoid cold start per call) while testing; max per budget |
| Start command | leave default (ENTRYPOINT runs `scripts/entrypoint.sh`) |

Environment variables (set secrets as RunPod **secrets**, not plain text):

```
MME_API_KEYS=<tenant>=<random 32+ char token>      # required, secret
MME_ADMIN_KEYS=<random token>                      # only if MME_ALLOW_VOICE_REGISTRATION=1
MME_PROFILE=production
PORT=8080  PORT_HEALTH=8080
MAX_SEATS=1  MAX_PENDING=4                         # raise only after benchmarking
HF_HOME=/runpod-volume/.cache/huggingface          # <- serverless volumes mount at /runpod-volume (verify); pods use /workspace
TRITON_CACHE_DIR=/runpod-volume/.cache/triton
VOICES_FILE=/runpod-volume/voices/voices.json      # optional, see Voice setup
```
`HF_TOKEN` is not needed (public model). Never bake any token into the image.

**Persistent cache.** First start downloads ~5 GB of weights + Mimi + ECAPA (the `/ping` stays 204). Pre-populate once to avoid doing it on a billed GPU worker: run the same image on a cheap CPU pod with the volume attached and `python /app/scripts/download_model.py`; later workers start offline-capable (`HF_HUB_OFFLINE=1` is then safe and also prevents surprise downloads of unpinned Mimi/ECAPA revisions).

**Startup order** (all inside one process): port opens -> `/ping` 204 -> download/verify (`model_store`) -> engine load -> CUDA-graph warm-up -> `/ping` 200. On any failure `/ping` is 503 and the process exits 1 after `MME_EXIT_DELAY_S`.

**Unverified for Path A (test before relying on it):**
1. Whether the proxy forwards `Authorization` to the worker. Use `X-MME-Api-Key` for the app token if it does not.
2. WebSocket upgrade through `https://ENDPOINT_ID.api.runpod.ai/v1/tts/stream` and `/v1/tts/incremental`, maximum connection lifetime, idle timeouts, and whether frames are buffered (which would add latency).
3. Whether chunked/streamed HTTP bodies are passed through unbuffered. RunPod documents a 30 MB payload limit and a 5.5 min processing timeout per request; calls longer than that must use WebSocket sessions or reconnect.
4. Worker scale-down behaviour with open WebSockets (sessions could be cut on scale-in).
5. Real-time stability: measure inter-packet jitter at the client (PERFORMANCE_TESTING.md, test G5).

## Path B - dedicated GPU Pod (fallback)

Use if any item above fails or jitter through the proxy is unacceptable.
- Deploy the same image as a Pod, GPU as above, expose HTTP port 8080 (RunPod proxy URL `https://<pod-id>-8080.proxy.runpod.net`) or a TCP port for a direct connection (lower latency, no proxy timeouts; then put TLS termination in front yourself).
- Volume mounts at `/workspace`; defaults (`HF_HOME=/workspace/.cache/huggingface`) already match.
- Env vars as above; no `PORT_HEALTH` needed.
- Trade-offs: you pay for idle time, no autoscaling, you own restarts/monitoring.

## Voice setup (Utkarsh reference)
The recording and transcript stay **outside git and the image**. Use a CPU pod / SSH into the Pod with the volume mounted:
```bash
python /app/scripts/register_voice.py persist --voices-dir /runpod-volume/voices --tenant mme --name utkarsh \
    --audio /path/utkarsh_ref.wav --transcript-file /path/utkarsh_ref.txt --lang hi --consent
python /app/scripts/register_voice.py check --manifest /runpod-volume/voices/voices.json
```
Then set `VOICES_FILE=/runpod-volume/voices/voices.json` and restart; the voice id is `mme__utkarsh`, usable only with tenant `mme` credentials. Validation enforced: 3-10 s, WAV/FLAC, >= 16 kHz, not silent, transcript + language present, consent flag. **Voice similarity to Chatterbox is not validated** - run test G4.

## Rollout checklist
1. Build + push image (README). 2. Create the endpoint with the table above. 3. Wait for `/ping` 200. 4. `curl` `/health`. 5. Run the GPU acceptance plan. 6. Only then point MME staging at it. Keep the existing TTS as fallback.

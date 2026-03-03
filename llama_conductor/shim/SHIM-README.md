# MoA Chat UI Shim

This shim serves `llama-server` WebUI and reroutes API calls so your stack can use:

- `llama-conductor` for `/v1/*` chat/completions
- `llama-conductor` for `/v1/models` (plus shim capability normalization)
- `llama-server` for the WebUI and non-`/v1` endpoints

## Runtime config (environment variables)

- `MOA_CHAT_HOST` (default: `0.0.0.0`)
- `MOA_CHAT_PORT` (default: `8088`)
- `MOA_CHAT_LLAMASERVER_URL` (default: `http://127.0.0.1:8010`)
- `MOA_CHAT_ROUTER_URL` (default: `http://127.0.0.1:9000`)
- `MOA_CHAT_AUTH_USER` (default: `moa`)
- `MOA_CHAT_PASSWORD` (default: empty / auth disabled)
- `MOA_CHAT_FORCE_VISION` (default: `0`; set `1` to force WebUI image upload support)
- `MOA_CHAT_RENDER_MODE` (default: `stream`; options: `stream`, `buffered`)
- `MOA_CHAT_INJECT_UI` (default: `1`; set `0` to disable toolbar injection)
- `MOA_CHAT_UI_MODEL_ALIAS` (optional; override model name shown in patched model metadata)
- `MOA_CHAT_ROUTER_MODEL_ID` (optional; router model id used for WebUI alias mapping)

## Render mode

- `stream`:
  - passes token streaming through as-is (normal live typing effect)
- `buffered`:
  - keeps upstream streaming transport internally
  - buffers the full assistant output server-side
  - emits one final SSE content chunk (plus done marker) to the UI
  - gives "show full answer at once after a delay" behavior while preserving SSE compatibility
  - note: capture is scoped to the active chat key; if you switch chats, send one message in that chat before exporting

## Shim endpoints

- `GET /shim/healthz`: shim health and effective config
- `POST /shim/export/md`: convert conversation JSON payload into downloadable markdown

When UI injection is enabled, the root WebUI page gets a small toolbar with:

- `EXPORT MD` action (`/shim/export/md`)

## Launch

Use your stack launcher (for example, `python -m llama_conductor.launch_stack up`).

The launcher sets these environment variables before starting `shim_server.py`.

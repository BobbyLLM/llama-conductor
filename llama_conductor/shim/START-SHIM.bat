@echo off
setlocal

cd /d "%~dp0"

set "MOA_CHAT_AUTH_USER=moa"
set "MOA_CHAT_FORCE_VISION=1"
set "MOA_CHAT_RENDER_MODE=buffered"
set "MOA_CHAT_INJECT_UI=1"
set "MOA_CHAT_UI_MODEL_ALIAS=MOA"
set "MOA_CHAT_HOST=0.0.0.0"
set "MOA_CHAT_PORT=8088"
set "MOA_CHAT_LLAMASERVER_URL=http://127.0.0.1:8010"
set "MOA_CHAT_ROUTER_URL=http://127.0.0.1:9000"
set "MOA_CHAT_ROUTER_MODEL_ID=moa-router"

py shim_server.py

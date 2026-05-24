# EditIQ

Local TikTok intelligence dashboard: extracts video/audio features, predicts view performance, combines account context when connected, stores history in SQLite, and supports optional ML training.

## Quick start

```powershell
cd c:\Users\gabsc\Documents\Code\EditIQ
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) · API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Optional integrations

Create a local `.env` or set these environment variables before starting the server:

```powershell
$env:TOKEN_ENCRYPTION_KEY="replace-with-a-long-random-secret"
$env:TIKTOK_CLIENT_KEY="replace-with-your-tiktok-client-key"
$env:TIKTOK_CLIENT_SECRET="replace-with-your-tiktok-client-secret"
$env:TIKTOK_REDIRECT_URI="http://127.0.0.1:8000/api/tiktok/callback"
$env:TIKTOK_SCOPES="user.info.basic,user.info.profile,user.info.stats,video.list"
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:OLLAMA_MODEL="llama3:latest"
```

You can also copy `.env.example` to `.env` and fill in the TikTok app values from the TikTok Developer portal. The redirect URI in TikTok must match `TIKTOK_REDIRECT_URI` exactly.

TikTok OAuth runs only through the backend. Tokens are encrypted before storage and are never exposed to the frontend. Ollama is used as a local helper for explanations and advice; if it is offline, EditIQ falls back to deterministic guidance.

## Verify

```powershell
python scripts\verify.py
```

## Project layout

| Path | Role |
|------|------|
| `app/main.py` | FastAPI routes + static frontend |
| `app/analyzer.py` | FFmpeg + OpenCV feature extraction |
| `app/scorer.py` | Heuristic scores & feedback |
| `app/views_model.py` | CatBoost views forecasting |
| `app/intelligence.py` | Account-aware prediction layer |
| `app/tiktok_client.py` | Server-side TikTok OAuth/API client |
| `app/database.py` | SQLite persistence |
| `static/` | Dashboard (HTML/CSS/JS) |
| `docs/` | Architecture, roadmap, ML guide |

Uploaded videos are processed in a temp folder and deleted after analysis. Only features, scores, and timeline curves are kept in `data/editiq.db`.

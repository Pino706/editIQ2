# ML Integration Guide — Views Prediction

EditIQ learns from **your real TikTok view counts**, not from the heuristic viral score. The views model maps edit features (cuts, motion, audio sync, etc.) to expected performance using `log1p(views)`.

## Quick workflow (50-video goal)

1. Open the dashboard → **Train AI** tab.
2. Drop **many MP4s** (bulk) into the training zone.
3. Enter each video’s **view count** from TikTok Analytics (required before processing).
4. Click **Process all** — each file is analyzed and stored with features + views.
5. When you have **at least 10** labeled videos (50 recommended), click **Train views model**.
6. Switch to **Analyze** → upload a **new** edit (views optional) → see **Predicted views**.

## Requirements

| Item | Value |
|------|--------|
| Minimum to train | **10** videos with `views > 0` |
| Recommended dataset | **50** diverse view counts |
| Target | `log1p(views)` → prediction `expm1(model output)` |

**Important:** Use a spread of outcomes (e.g. 500, 5K, 50K, 500K views). Fifty videos all near the same count will not teach the model much.

## What is *not* the views model

- **Viral / Hook / Pacing scores** — rule-based edit quality (always shown).
- **Legacy `POST /api/train`** — trains on `viral_score` or `retention_pct` via `app/ml_model.py`; separate from views prediction.

Views prediction uses `app/views_model.py` and `data/models/views_model.pkl`, then `app/intelligence.py` adjusts the forecast with TikTok account context when an account is connected.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/dataset/stats` | `{ labeled, unlabeled, goal, can_train }` |
| GET | `/api/dataset/labeled` | Labeled videos sorted by views (for charts & ranking) |
| POST | `/api/train-views` | Train views model; returns R², MAE, feature importances |
| GET | `/api/views-model-status` | Model trained?, metrics, importances |
| PATCH | `/api/analysis/{id}` | Body `{ "views": 12345 }` — fix a label without re-upload |

After `POST /api/analyze`, if a views model exists, the response includes:

- `predicted_views` (integer)
- `views_model_ready: true`
- `training_samples` (dataset size used for training)

Heuristic `viral_score` is **not** replaced by the views model.

## Feature vector

Same 11 features for every video (see `app/views_model.py`):

`cuts_count`, `cuts_per_second`, `avg_scene_duration`, `motion_intensity`, `visual_change_rate`, `visual_stability`, `hook_speed`, `audio_energy`, `audio_spikes_count`, `audio_pacing`, `av_sync_score`

## Artifacts

| File | Description |
|------|-------------|
| `data/models/views_model.pkl` | CatBoost views model, with fallback training metadata |
| `data/models/views_model_meta.json` | `train_r2`, `cv_r2`, `mae_views`, `median_ae_views`, `feature_importances` |

## Account-aware intelligence

The prediction pipeline is modular:

1. `app/analyzer.py` extracts video features.
2. `app/views_model.py` predicts base views from labeled training data.
3. `app/account_intelligence.py` summarizes the connected TikTok account.
4. `app/intelligence.py` combines the base forecast, video strength, account momentum, account fit, and trend fit.
5. `app/text_model.py` asks the local Ollama model for natural-language advice, without replacing the numeric forecast.

## Accuracy expectations

- Predictions are **estimates** based on edit signals only (not hashtag, niche, or posting time).
- With ~50 diverse labels, use predictions to **compare edits** (A vs B), not as exact view guarantees.
- Re-train after every **10–15** new labeled posts.
- Check **feature importance** in the Train AI tab to see what your audience responds to (e.g. cuts/sec vs motion).

## Fixing labels

Wrong view count on an old row:

```bash
curl -X PATCH http://127.0.0.1:8000/api/analysis/3 \
  -H "Content-Type: application/json" \
  -d "{\"views\": 120000}"
```

Then call `POST /api/train-views` again.

## Export for notebooks

```sql
.mode csv
.headers on
SELECT id, filename, views, predicted_views, cuts_per_second, motion_intensity, viral_score
FROM videos
WHERE views IS NOT NULL AND views > 0;
```

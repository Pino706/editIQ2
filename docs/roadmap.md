# EditIQ Roadmap

## Phase 1 — Heuristic MVP (current)

- [x] FFmpeg via `imageio-ffmpeg` (no manual install)
- [x] OpenCV cut + motion features at 240p
- [x] SciPy audio energy + beat spikes
- [x] Heuristic viral / hook / pacing / retention scores
- [x] SQLite history + static dashboard
- [x] CLI verification script

## Phase 2 — ML calibration

- [ ] Collect 20+ labeled edits (views, retention, or self-rated viral)
- [ ] Train GradientBoosting on `viral_score` or `retention_pct`
- [ ] Compare ML vs heuristic on holdout set
- [ ] Feature importance export for editor feedback

## Phase 3 — Editor-specific profiles

- [ ] Presets: gaming, anime, vlog (different pacing targets)
- [ ] User-adjustable weight profiles in UI
- [ ] Batch folder analysis

## Phase 4 — Advanced signal processing

- [ ] Optional dense optical flow (toggle for accuracy vs speed)
- [ ] ONNX beat detector for music-heavy edits
- [ ] Thumbnail strip stored as BLOB for history preview

## Phase 5 — Export & integration

- [ ] CSV export of feature matrix for external notebooks
- [ ] Webhook / CLI for CI-style regression on reference clips

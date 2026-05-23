const API = "";
let motionChart = null;
let audioChart = null;
let datasetChart = null;
let importanceChart = null;
let qualityRadarChart = null;
let accountPerformanceChart = null;
let accountComparisonChart = null;

const form = document.getElementById("analyze-form");
const dropZone = document.getElementById("drop-zone");
const videoInput = document.getElementById("video-input");
const browseBtn = document.getElementById("browse-btn");
const fileNameEl = document.getElementById("file-name");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingStatus = document.getElementById("loading-status");
const reportSection = document.getElementById("report-section");
const historyTbody = document.getElementById("history-tbody");
const modelStatusEl = document.getElementById("model-status");

const trainDropZone = document.getElementById("train-drop-zone");
const trainFilesInput = document.getElementById("train-files-input");
const trainBrowseBtn = document.getElementById("train-browse-btn");
const trainingQueueWrap = document.getElementById("training-queue-wrap");
const trainingQueueTbody = document.getElementById("training-queue-tbody");
const processAllBtn = document.getElementById("process-all-btn");
const trainViewsBtn = document.getElementById("train-views-btn");
const processProgressEl = document.getElementById("process-progress");
const trainResultEl = document.getElementById("train-result");
const rankedTbody = document.getElementById("ranked-tbody");
const datasetProgressText = document.getElementById("dataset-progress-text");
const datasetProgressFill = document.getElementById("dataset-progress-fill");
const viewsModelBadge = document.getElementById("views-model-badge");
const accountEmpty = document.getElementById("account-empty");
const accountDashboard = document.getElementById("account-dashboard");
const refreshAccountBtn = document.getElementById("refresh-account-btn");

let trainingQueue = [];

const LOADING_STEPS = [
  "Extracting audio...",
  "Detecting cuts...",
  "Measuring motion...",
  "Calculating scores...",
];

initializeInterface();

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const activate = () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => {
        p.classList.remove("active", "entering");
      });
      tab.classList.add("active");
      const id = tab.dataset.tab;
      const panel = document.getElementById(`panel-${id}`);
      panel.classList.add("active", "entering");
      window.setTimeout(() => panel.classList.remove("entering"), 320);
      if (id === "train") refreshTrainingLab();
      if (id === "account") loadTikTokAccount();
    };

    if (document.startViewTransition) {
      document.startViewTransition(activate);
      return;
    }

    activate();
  });
});

browseBtn.addEventListener("click", () => videoInput.click());
dropZone.addEventListener("click", (e) => {
  if (e.target !== browseBtn) videoInput.click();
});

videoInput.addEventListener("change", () => {
  if (videoInput.files[0]) setSelectedFileName(videoInput.files[0].name);
});

setupDropZone(dropZone, (files) => {
  if (files.length) {
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    videoInput.files = dt.files;
    setSelectedFileName(files[0].name);
  }
});

trainBrowseBtn.addEventListener("click", () => trainFilesInput.click());
trainFilesInput.addEventListener("change", () => addFilesToQueue(trainFilesInput.files));
setupDropZone(trainDropZone, (files) => addFilesToQueue(files));
refreshAccountBtn?.addEventListener("click", refreshTikTokAccount);

function setupDropZone(zone, onFiles) {
  ["dragenter", "dragover"].forEach((ev) => {
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
    });
  });
  zone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files);
  });
}

function initializeInterface() {
  if (window.lucide) {
    window.lucide.createIcons({ attrs: { "aria-hidden": "true" } });
  }

  document.addEventListener("pointermove", (event) => {
    document.documentElement.style.setProperty("--mx", `${event.clientX}px`);
    document.documentElement.style.setProperty("--my", `${event.clientY}px`);
  }, { passive: true });

  const revealEls = document.querySelectorAll("[data-reveal]");
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  revealEls.forEach((el) => revealObserver.observe(el));

  document.querySelectorAll("button").forEach((button) => {
    button.addEventListener("pointerdown", () => {
      button.animate(
        [
          { transform: "translateY(-1px) scale(1)" },
          { transform: "translateY(0) scale(0.985)" },
          { transform: "translateY(-1px) scale(1)" },
        ],
        { duration: 180, easing: "cubic-bezier(0.22, 1, 0.36, 1)" }
      );
    });
  });
}

function setSelectedFileName(name) {
  fileNameEl.textContent = name;
  dropZone.classList.add("has-file");
}

function addFilesToQueue(fileList) {
  for (const file of fileList) {
    if (!file.type.startsWith("video/") && !file.name.match(/\.(mp4|mov|webm|mkv|avi)$/i)) continue;
    if (trainingQueue.some((q) => q.file.name === file.name && q.file.size === file.size)) continue;
    trainingQueue.push({ file, views: "", status: "pending", error: null });
  }
  renderTrainingQueue();
}

function syncViewsFromInputs() {
  trainingQueueTbody.querySelectorAll(".views-input").forEach((inp) => {
    const idx = Number(inp.dataset.idx);
    if (trainingQueue[idx]) trainingQueue[idx].views = inp.value;
  });
}

function normalizeViewsInput(raw) {
  if (raw == null || String(raw).trim() === "") return "";
  return String(raw).replace(/,/g, "").replace(/\s/g, "");
}

function formatApiError(data, status) {
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return `Request failed (${status})`;
}

function renderTrainingQueue() {
  trainingQueueWrap.classList.toggle("hidden", trainingQueue.length === 0);
  processAllBtn.disabled = trainingQueue.length === 0;
  trainingQueueTbody.innerHTML = trainingQueue
    .map(
      (row, i) => `
    <tr data-idx="${i}">
      <td title="${escapeHtml(row.file.name)}">${truncate(row.file.name, 32)}</td>
      <td><input type="text" inputmode="numeric" class="views-input" data-idx="${i}" placeholder="e.g. 50000" value="${escapeHtml(String(row.views || ""))}" ${row.status === "done" ? "disabled" : ""} /></td>
      <td><span class="status-pill status-${row.status}" title="${escapeHtml(row.error || "")}">${row.status}${row.error ? " !" : ""}</span></td>
      <td>${row.status === "pending" || row.status === "error" ? `<button type="button" class="link-btn remove-row" data-idx="${i}">Remove</button>` : ""}</td>
    </tr>`
    )
    .join("");

  trainingQueueTbody.querySelectorAll(".views-input").forEach((inp) => {
    inp.addEventListener("input", (e) => {
      trainingQueue[Number(e.target.dataset.idx)].views = e.target.value;
    });
  });
  trainingQueueTbody.querySelectorAll(".remove-row").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      trainingQueue.splice(Number(e.target.dataset.idx), 1);
      renderTrainingQueue();
    });
  });
}

processAllBtn.addEventListener("click", processAllTraining);

async function processAllTraining() {
  syncViewsFromInputs();
  const pending = trainingQueue.filter((r) => r.status === "pending" || r.status === "error");
  for (const row of pending) {
    row.views = normalizeViewsInput(row.views);
    const n = Number(row.views);
    if (!row.views || !Number.isFinite(n) || n <= 0) {
      alert(`Enter view count for: ${row.file.name}`);
      return;
    }
  }

  processAllBtn.disabled = true;
  trainViewsBtn.disabled = true;
  let done = 0;
  const total = pending.length;

  for (let i = 0; i < trainingQueue.length; i++) {
    const row = trainingQueue[i];
    if (row.status === "done") continue;
    if (!row.views || Number(row.views) <= 0) continue;

    row.status = "analyzing";
    renderTrainingQueue();
    processProgressEl.textContent = `${done}/${total} processing...`;

    const fd = new FormData();
    fd.append("file", row.file, row.file.name);
    fd.append("views", normalizeViewsInput(row.views));

    try {
      const res = await fetch(`${API}/api/analyze`, { method: "POST", body: fd });
      let data = {};
      try {
        data = await res.json();
      } catch {
        data = {};
      }
      if (!res.ok) throw new Error(formatApiError(data, res.status));
      row.status = "done";
      row.error = null;
      done++;
    } catch (err) {
      row.status = "error";
      row.error = err.message;
    }
    renderTrainingQueue();
  }

  processProgressEl.textContent = `Finished ${done}/${total}`;
  processAllBtn.disabled = false;
  trainingQueue = trainingQueue.filter((r) => r.status !== "done");
  renderTrainingQueue();
  await refreshTrainingLab();
  await loadHistory();
}

trainViewsBtn.addEventListener("click", async () => {
  trainViewsBtn.disabled = true;
  trainResultEl.textContent = "Training...";
  try {
    const res = await fetch(`${API}/api/train-views`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.message || "Training failed");
    trainResultEl.textContent =
      `ML trained (${data.model_type || "views model"}) on ${data.samples} videos - CV R2 ${data.cv_r2 ?? data.test_r2} - typical error +/-${formatViews(data.mae_views)} views`;
    if (data.feature_importances) renderImportanceChart(data.feature_importances);
    await refreshTrainingLab();
    await loadHistory();
  } catch (err) {
    trainResultEl.textContent = err.message;
  } finally {
    trainViewsBtn.disabled = false;
  }
});

async function refreshTrainingLab() {
  const [statsRes, labeledRes, statusRes] = await Promise.all([
    fetch(`${API}/api/dataset/stats`),
    fetch(`${API}/api/dataset/labeled`),
    fetch(`${API}/api/views-model-status`),
  ]);
  const stats = await statsRes.json();
  const labeled = await labeledRes.json();
  const status = await statusRes.json();

  const warnEl = document.getElementById("train-warning");
  if (stats.can_train && !status.trained) {
    warnEl.classList.remove("hidden");
    warnEl.textContent =
      `You have ${stats.labeled} labeled videos but the ML model is not trained yet. Click "Train views model" below. Process all saves the examples first.`;
  } else if (status.trained) {
    warnEl.classList.add("hidden");
  } else {
    warnEl.classList.remove("hidden");
    warnEl.textContent = `Need ${stats.min_train || 5}+ videos with view counts. Currently: ${stats.labeled} labeled (50+ recommended for accuracy).`;
  }

  const goal = stats.goal || 200;
  const labeledN = stats.labeled || 0;
  const pct = Math.min(100, (labeledN / goal) * 100);
  datasetProgressText.textContent = `${labeledN} / ${goal} labeled`;
  datasetProgressFill.style.width = `${pct}%`;
  trainViewsBtn.disabled = !stats.can_train;

  if (status.trained) {
    viewsModelBadge.textContent = `Model ready - ${status.samples} videos`;
    viewsModelBadge.className = "badge badge-ok";
    modelStatusEl.textContent =
      `Views model: ${status.samples} samples - R2 ${status.test_r2 ?? "-"} - MAE +/-${formatViews(status.mae_views)}`;
    if (status.feature_importances) renderImportanceChart(status.feature_importances);
  } else {
    viewsModelBadge.textContent = "Model not trained";
    viewsModelBadge.className = "badge badge-muted";
    modelStatusEl.textContent = status.message || "Train with 10+ labeled videos.";
  }

  renderRankedTable(labeled.items || []);
  renderDatasetChart(labeled.items || []);
}

function renderRankedTable(items) {
  const n = items.length;
  rankedTbody.innerHTML = items
    .map((row, i) => {
      const tier = tierForRank(i, n);
      return `
    <tr>
      <td>${i + 1}</td>
      <td title="${escapeHtml(row.filename)}">${truncate(row.filename, 28)}</td>
      <td>${formatViews(row.views)}</td>
      <td>${row.predicted_views != null ? formatViews(row.predicted_views) : "-"}</td>
      <td><span class="tier-badge tier-${tier}">${tier}</span></td>
    </tr>`;
    })
    .join("");
}

function tierForRank(index, total) {
  if (total === 0) return "mid";
  const p = index / total;
  if (p < 0.2) return "best";
  if (p > 0.8) return "worst";
  return "mid";
}

function renderDatasetChart(items) {
  const labels = items.map((r) => truncate(r.filename, 12));
  const values = items.map((r) => r.views);
  if (datasetChart) datasetChart.destroy();
  const ctx = document.getElementById("dataset-chart");
  if (!ctx || items.length === 0) return;
  datasetChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: "rgba(10, 132, 255, 0.18)",
        borderColor: "#0a84ff",
        borderWidth: 1,
      }],
    },
    options: chartOptions(true),
  });
}

function renderImportanceChart(importances) {
  const entries = Object.entries(importances).slice(0, 8);
  const labels = entries.map(([k]) => k.replace(/_/g, " "));
  const values = entries.map(([, v]) => v);
  if (importanceChart) importanceChart.destroy();
  const ctx = document.getElementById("importance-chart");
  if (!ctx) return;
  importanceChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: "rgba(88, 86, 214, 0.18)",
        borderColor: "#5856d6",
        borderWidth: 1,
      }],
    },
    options: {
      ...chartOptions(false),
      indexAxis: "y",
    },
  });
}

function chartOptions(logScale) {
  return {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#666d78", maxRotation: 45 }, grid: { color: "rgba(23,24,28,0.06)" } },
      y: {
        type: logScale ? "logarithmic" : "linear",
        ticks: { color: "#666d78" },
        grid: { color: "rgba(23,24,28,0.06)" },
      },
    },
  };
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = videoInput.files[0];
  if (!file) {
    alert("Please select a video file.");
    return;
  }

  const fd = new FormData();
  fd.append("file", file);
  ["views", "likes", "shares", "saves", "retention_pct"].forEach((name) => {
    const input = form.elements[name];
    if (input?.value) fd.append(name, input.value);
  });

  showLoading(true);
  let step = 0;
  const stepTimer = setInterval(() => {
    loadingStatus.textContent = LOADING_STEPS[step % LOADING_STEPS.length];
    step++;
  }, 1800);

  try {
    const res = await fetch(`${API}/api/analyze`, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(parseError(data));
    renderReport(data);
    await loadHistory();
    await loadViewsModelStatus();
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    clearInterval(stepTimer);
    showLoading(false);
  }
});

function parseError(data) {
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).join(", ");
  return "Request failed";
}

function showLoading(on) {
  loadingOverlay.classList.toggle("hidden", !on);
  document.getElementById("analyze-btn").disabled = on;
}

function setGauge(id, value) {
  const el = document.getElementById(`gauge-${id}`);
  const card = el.closest(".gauge-card");
  const circle = card.querySelector(".gauge-fill");
  const circumference = 326.7;
  const v = Math.min(100, Math.max(0, Number(value) || 0));
  el.textContent = Math.round(v);
  circle.style.strokeDashoffset = circumference - (v / 100) * circumference;
}

function renderReport(data) {
  reportSection.classList.remove("hidden");
  const s = data.scores || {};
  const f = data.features || data;

  document.getElementById("report-meta").textContent =
    `${data.filename} - ${(data.duration || 0).toFixed(1)}s - Account-aware forecast`;

  renderPredictedViews(data);
  renderViewsJustification(data.views_justification || {});
  renderTierProbabilities(data.tier_probabilities || []);
  renderFeatureInsights(data.feature_insights || []);

  setGauge("viral", s.viral_score);
  setGauge("hook", s.hook_score);
  setGauge("pacing", s.pacing_score);
  setGauge("retention", s.retention_risk);

  fillList("strengths-list", s.feedback?.strengths || []);
  fillList("improvements-list", s.feedback?.improvements || []);
  renderAdvancedPrediction(data);

  renderCharts(data.timeline || {});
  renderQualityRadar(data.prediction || {});
}

function renderViewsJustification(j) {
  const section = document.getElementById("justification-section");
  const summary = document.getElementById("justification-summary");
  const bullets = document.getElementById("justification-bullets");
  const similarWrap = document.getElementById("similar-videos");
  const similarList = document.getElementById("similar-videos-list");

  if (!j || !j.summary) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");
  summary.textContent = j.summary.replace(/\*\*/g, "");

  const items = [
    ...(j.narrative_bullets || []),
    ...(j.algorithm_tips || []).map((t) => `Algorithm: ${t}`),
  ];
  bullets.innerHTML = items
    .map((t) => `<li>${escapeHtml(t.replace(/\*\*/g, ""))}</li>`)
    .join("");

  const sim = j.similar_videos || [];
  if (sim.length) {
    similarWrap.classList.remove("hidden");
    similarList.innerHTML = sim
      .map(
        (v) =>
          `<li>${escapeHtml(v.filename)} - ${formatViews(v.views)} actual views</li>`
      )
      .join("");
  } else {
    similarWrap.classList.add("hidden");
  }
}

function renderTierProbabilities(tiers) {
  const section = document.getElementById("tier-probs-section");
  const list = document.getElementById("tier-probs-list");
  if (!tiers.length) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");
  list.innerHTML = tiers
    .map(
      (t) => `
    <div class="tier-row">
      <span class="tier-row-label">${escapeHtml(t.label)}</span>
      <div class="tier-bar-wrap"><div class="tier-bar-fill" style="width:${Math.min(100, t.probability_pct)}%"></div></div>
      <span class="tier-pct">${t.probability_pct}%</span>
    </div>`
    )
    .join("");
}

function renderFeatureInsights(insights) {
  const tbody = document.getElementById("feature-insights-tbody");
  if (!insights.length) {
    tbody.innerHTML = `<tr><td colspan="4">No insights yet. Train the views model with 10+ labeled videos.</td></tr>`;
    return;
  }
  tbody.innerHTML = insights
    .map(
      (row) => `
    <tr>
      <td>${escapeHtml(row.label)}</td>
      <td>${escapeHtml(String(row.value))}</td>
      <td>${escapeHtml(row.ml_learning || "")}</td>
      <td>${escapeHtml(row.algorithm_note || "")}</td>
    </tr>`
    )
    .join("");
}

function renderPredictedViews(data) {
  const card = document.getElementById("predicted-views-card");
  const valueEl = document.getElementById("predicted-views-value");
  const subEl = document.getElementById("predicted-views-sub");
  const compareEl = document.getElementById("predicted-views-compare");

  const predicted = data.predicted_views ?? data.prediction?.predicted_views ?? data.metadata?.predicted_views;
  const low = data.prediction_range_low ?? data.prediction?.prediction_range_low;
  const high = data.prediction_range_high ?? data.prediction?.prediction_range_high;
  const actual = data.metadata?.views ?? data.views;

  if (predicted != null) {
    card.classList.remove("hidden");
    valueEl.textContent = formatViews(predicted);
    const range = low != null && high != null ? ` - range ${formatViews(low)} to ${formatViews(high)}` : "";
    subEl.textContent = data.views_model_ready
      ? `Based on your ${data.training_samples || "trained"}-video dataset${range}`
      : `Heuristic estimate until the views model is trained${range}`;
    if (actual != null && actual > 0) {
      const diff = predicted - actual;
      const pct = actual > 0 ? ((diff / actual) * 100).toFixed(0) : 0;
      compareEl.classList.remove("hidden");
      compareEl.textContent =
        `Actual: ${formatViews(actual)} - Delta ${diff >= 0 ? "+" : ""}${formatViews(diff)} (${pct}%)`;
    } else {
      compareEl.classList.add("hidden");
    }
  } else {
    card.classList.remove("hidden");
    valueEl.textContent = "-";
    subEl.textContent =
      data.views_model_message ||
      "Train the views model with 10+ labeled videos in Train AI tab.";
    compareEl.classList.add("hidden");
  }
}

function renderAdvancedPrediction(data) {
  const prediction = data.prediction || {};
  const advice = data.ai_advice || {};
  const finalEl = document.getElementById("final-recommendation");
  const changeEl = document.getElementById("prediction-change-note");
  const grid = document.getElementById("advanced-score-grid");
  if (!finalEl || !grid) return;

  finalEl.textContent = advice.final_recommendation || prediction.final_recommendation || "No recommendation available yet.";
  changeEl.textContent = prediction.why_prediction_changed || advice.account_context_note || "";
  const cards = [
    ["Overall", prediction.overall_score],
    ["Viral probability", prediction.viral_probability, "%"],
    ["Video strength", prediction.video_strength_score],
    ["Account fit", prediction.account_fit_score],
    ["Trend fit", prediction.trend_fit_score],
    ["Visual quality", prediction.visual_quality_score],
  ];
  grid.innerHTML = cards
    .map(([label, value, suffix]) => `
      <div class="mini-score-card">
        <span>${label}</span>
        <strong>${value == null ? "-" : `${Math.round(Number(value))}${suffix || ""}`}</strong>
      </div>`)
    .join("");

  fillList("strengths-list", [
    ...(advice.what_is_strong || []),
    ...Array.from(document.getElementById("strengths-list")?.querySelectorAll("li") || []).map((li) => li.textContent),
  ].filter(Boolean).slice(0, 5));
  fillList("improvements-list", [
    ...(advice.what_is_weak || []),
    ...(advice.what_should_change || []),
    ...Array.from(document.getElementById("improvements-list")?.querySelectorAll("li") || []).map((li) => li.textContent),
  ].filter(Boolean).slice(0, 6));
}

function fillList(id, items) {
  document.getElementById(id).innerHTML = items.map((t) => `<li>${escapeHtml(t)}</li>`).join("");
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function renderCharts(timeline) {
  const motion = timeline.motion || [];
  const audio = timeline.audio_energy || [];
  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#666d78", maxTicksLimit: 8 }, grid: { color: "rgba(23,24,28,0.06)" } },
      y: { ticks: { color: "#666d78" }, grid: { color: "rgba(23,24,28,0.06)" } },
    },
  };

  if (motionChart) motionChart.destroy();
  motionChart = new Chart(document.getElementById("motion-chart"), {
    type: "line",
    data: {
      labels: motion.map((p) => p.t),
      datasets: [{ data: motion.map((p) => p.v), borderColor: "#0a84ff", backgroundColor: "rgba(10,132,255,0.08)", fill: true, tension: 0.3, pointRadius: 0 }],
    },
    options: chartDefaults,
  });

  if (audioChart) audioChart.destroy();
  audioChart = new Chart(document.getElementById("audio-chart"), {
    type: "line",
    data: {
      labels: audio.map((p) => p.t),
      datasets: [{ data: audio.map((p) => p.v), borderColor: "#2f9f6b", backgroundColor: "rgba(47,159,107,0.08)", fill: true, tension: 0.3, pointRadius: 0 }],
    },
    options: chartDefaults,
  });
}

function renderQualityRadar(prediction) {
  const ctx = document.getElementById("quality-radar-chart");
  if (!ctx) return;
  if (qualityRadarChart) qualityRadarChart.destroy();
  qualityRadarChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels: ["Hook", "Pacing", "Visual", "Account", "Trend", "Viral"],
      datasets: [{
        data: [
          prediction.hook_score || 0,
          prediction.pacing_score || 0,
          prediction.visual_quality_score || 0,
          prediction.account_fit_score ?? 50,
          prediction.trend_fit_score ?? 50,
          prediction.viral_probability || 0,
        ],
        borderColor: "#2f9f6b",
        backgroundColor: "rgba(47,159,107,0.16)",
        pointBackgroundColor: "#101216",
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: { display: false },
          grid: { color: "rgba(23,24,28,0.08)" },
          angleLines: { color: "rgba(23,24,28,0.08)" },
          pointLabels: { color: "#666d78", font: { weight: "700" } },
        },
      },
    },
  });
}

async function loadTikTokAccount() {
  try {
    const res = await fetch(`${API}/api/tiktok/account`);
    const data = await res.json();
    renderTikTokAccount(data);
  } catch (err) {
    renderTikTokAccount({ connected: false, configured: false, message: err.message });
  }
}

async function refreshTikTokAccount() {
  refreshAccountBtn.disabled = true;
  try {
    const res = await fetch(`${API}/api/tiktok/refresh`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Refresh failed");
    renderTikTokAccount(data);
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    refreshAccountBtn.disabled = false;
  }
}

function renderTikTokAccount(data) {
  const connected = Boolean(data.connected);
  accountEmpty.classList.toggle("hidden", connected);
  accountDashboard.classList.toggle("hidden", !connected);
  if (!connected) {
    const emptyText = accountEmpty.querySelector("p:last-child");
    if (emptyText) {
      emptyText.textContent = data.configured === false
        ? "TikTok OAuth env vars are not configured on the backend yet."
        : (data.message || "No TikTok account connected yet.");
    }
    return;
  }

  const profile = data.profile || {};
  const summary = data.summary || {};
  const videos = data.videos || [];
  document.getElementById("account-avatar").src = profile.avatar_url || "";
  document.getElementById("account-name").textContent = profile.display_name || "TikTok creator";
  document.getElementById("account-bio").textContent = profile.bio_description || profile.profile_deep_link || "";
  document.getElementById("account-summary-chip").textContent = `${videos.length} recent uploads`;

  const metrics = [
    ["Followers", profile.follower_count],
    ["Following", profile.following_count],
    ["Likes", profile.likes_count],
    ["Avg views", summary.average_views],
    ["Momentum", summary.account_momentum_score, "/100"],
    ["Posts / week", summary.posting_frequency_per_week],
  ];
  document.getElementById("account-metric-grid").innerHTML = metrics
    .map(([label, value, suffix]) => `
      <div class="account-metric-card">
        <span>${label}</span>
        <strong>${value == null ? "-" : `${formatViewsLike(value)}${suffix || ""}`}</strong>
      </div>`)
    .join("");

  renderRecentUploads(videos);
  renderAccountCharts(videos);
}

function renderRecentUploads(videos) {
  const grid = document.getElementById("recent-upload-grid");
  if (!videos.length) {
    grid.innerHTML = `<div class="account-metric-card"><span>No videos</span><strong>-</strong></div>`;
    return;
  }
  grid.innerHTML = videos.slice(0, 12).map((v) => `
    <article class="upload-tile">
      ${v.cover_image_url ? `<img src="${escapeHtml(v.cover_image_url)}" alt="">` : ""}
      <div class="upload-tile-body">
        <div class="upload-tile-title">${escapeHtml(v.title || v.video_description || "Untitled upload")}</div>
        <div class="upload-tile-stats">
          <span>${formatViewsLike(v.view_count)} views</span>
          <span>${formatViewsLike(v.like_count)} likes</span>
          <span>${formatViewsLike(v.comment_count)} comments</span>
          <span>${formatViewsLike(v.share_count)} shares</span>
        </div>
      </div>
    </article>
  `).join("");
}

function renderAccountCharts(videos) {
  const ordered = [...videos].filter((v) => v.create_time).sort((a, b) => a.create_time - b.create_time).slice(-12);
  const labels = ordered.map((v) => new Date(v.create_time * 1000).toLocaleDateString("en", { month: "short", day: "numeric" }));
  if (accountPerformanceChart) accountPerformanceChart.destroy();
  accountPerformanceChart = new Chart(document.getElementById("account-performance-chart"), {
    type: "line",
    data: {
      labels,
      datasets: [{
        data: ordered.map((v) => v.view_count || 0),
        borderColor: "#2f9f6b",
        backgroundColor: "rgba(47,159,107,0.12)",
        fill: true,
        tension: 0.35,
        pointRadius: 3,
      }],
    },
    options: chartOptions(true),
  });

  const top = [...videos].sort((a, b) => (b.view_count || 0) - (a.view_count || 0)).slice(0, 8);
  if (accountComparisonChart) accountComparisonChart.destroy();
  accountComparisonChart = new Chart(document.getElementById("account-comparison-chart"), {
    type: "bar",
    data: {
      labels: top.map((v) => truncate(v.title || v.video_description || v.id || "Video", 12)),
      datasets: [{
        data: top.map((v) => v.view_count || 0),
        backgroundColor: "rgba(16,18,22,0.82)",
        borderColor: "#2f9f6b",
        borderWidth: 1,
      }],
    },
    options: chartOptions(true),
  });
}

async function loadHistory() {
  const res = await fetch(`${API}/api/history`);
  const data = await res.json();
  historyTbody.innerHTML = (data.items || [])
    .map(
      (row) => `
    <tr data-id="${row.id}">
      <td>${row.id}</td>
      <td title="${escapeHtml(row.filename)}">${truncate(row.filename, 20)}</td>
      <td>${fmt(row.viral_score)}</td>
      <td>${row.views != null ? formatViews(row.views) : "-"}</td>
      <td>${row.predicted_views != null ? formatViews(row.predicted_views) : "-"}</td>
      <td>${fmt(row.hook_score)}</td>
      <td>${fmt(row.pacing_score)}</td>
      <td>${row.cuts_per_second != null ? Number(row.cuts_per_second).toFixed(2) : "-"}</td>
    </tr>`
    )
    .join("");

  historyTbody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => loadAnalysis(tr.dataset.id));
  });
}

async function loadAnalysis(id) {
  const res = await fetch(`${API}/api/analysis/${id}`);
  if (!res.ok) return;
  const row = await res.json();
  document.querySelector('.tab[data-tab="analyze"]').click();
  const features = { ...row, ...(row.features_extra || {}) };
  const pred = row.predicted_views;
  renderReport({
    filename: row.filename,
    duration: row.duration,
    features,
    scores: {
      viral_score: row.viral_score,
      hook_score: row.hook_score,
      pacing_score: row.pacing_score,
      retention_risk: row.retention_risk,
      feedback: row.feedback,
    },
    timeline: row.timeline || {},
    predicted_views: pred,
    views_model_ready: pred != null,
    prediction: row.prediction,
    prediction_range_low: row.prediction_range_low,
    prediction_range_high: row.prediction_range_high,
    ai_advice: row.ai_advice,
    metadata: { views: row.views, predicted_views: pred },
    tier_probabilities: row.tier_probabilities || [],
    feature_insights: row.feature_insights || [],
    views_justification: row.views_justification || {},
  });
}

async function loadViewsModelStatus() {
  const res = await fetch(`${API}/api/views-model-status`);
  const data = await res.json();
  if (data.trained) {
    modelStatusEl.textContent =
      `Views model: ${data.samples} samples - R2 ${data.test_r2 ?? "-"} - MAE +/-${formatViews(data.mae_views)}`;
  } else {
    modelStatusEl.textContent = data.message || "No views model trained.";
  }
}

function formatViews(n) {
  if (n == null || Number.isNaN(n)) return "-";
  const v = Number(n);
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return String(Math.round(v));
}

function formatViewsLike(n) {
  if (n == null || Number.isNaN(Number(n))) return "-";
  return formatViews(Number(n));
}

function fmt(v) {
  return v != null ? Math.round(v) : "-";
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n - 1) + "..." : s;
}

loadHistory();
loadViewsModelStatus();
refreshTrainingLab();

// -------------------------------------------------------------
// EditIQ Frontend Application Script
// Controls forms, API interactions, dynamic reports, and gauges
// -------------------------------------------------------------

const API_BASE = ""; // Empty string for relative path since backend serves frontend

// Gauge circle circumference calculator
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 40; // radius is 40

document.addEventListener("DOMContentLoaded", () => {
    // Initialise Gauges
    resetGauges();
    
    // Fetch History & ML Status
    fetchHistory();
    fetchModelStatus();
    
    // Drag and Drop implementation
    setupDragAndDrop();
});

// Reset progress rings to 0
function resetGauges() {
    const rings = ["viralScoreRing", "hookScoreRing", "pacingScoreRing", "riskScoreRing"];
    rings.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.style.strokeDasharray = `${GAUGE_CIRCUMFERENCE} ${GAUGE_CIRCUMFERENCE}`;
            element.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
        }
    });
}

// Animate progress ring to value
function setGaugeValue(ringId, valueId, targetValue) {
    const ring = document.getElementById(ringId);
    const textVal = document.getElementById(valueId);
    if (!ring || !textVal) return;

    // Animate Text Number
    let startVal = 0;
    const duration = 800; // ms
    const startTime = performance.now();
    
    function animateText(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const currentVal = Math.round(progress * targetValue);
        textVal.textContent = currentVal;
        
        if (progress < 1) {
            requestAnimationFrame(animateText);
        }
    }
    requestAnimationFrame(animateText);

    // Animate Ring Stroke
    // Since risk is high/low, we might color differently but we stick to the gradients defined
    const offset = GAUGE_CIRCUMFERENCE - (targetValue / 100) * GAUGE_CIRCUMFERENCE;
    ring.style.strokeDasharray = `${GAUGE_CIRCUMFERENCE} ${GAUGE_CIRCUMFERENCE}`;
    ring.style.strokeDashoffset = offset;
}

// Drag & Drop Setup
function setupDragAndDrop() {
    const zone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("videoFile");

    ["dragenter", "dragover"].forEach(eventName => {
        zone.addEventListener(eventName, (e) => {
            e.preventDefault();
            zone.classList.add("drag-over");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        zone.addEventListener(eventName, (e) => {
            e.preventDefault();
            zone.classList.remove("drag-over");
        }, false);
    });

    zone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(fileInput);
        }
    }, false);
}

// File Select Handlers
function handleFileSelect(input) {
    const badge = document.getElementById("selectedFileBadge");
    const label = document.getElementById("selectedFileName");
    
    if (input.files && input.files.length > 0) {
        label.textContent = input.files[0].name;
        badge.style.display = "flex";
    }
}

function clearFileSelect(event) {
    event.stopPropagation();
    const input = document.getElementById("videoFile");
    const badge = document.getElementById("selectedFileBadge");
    
    input.value = "";
    badge.style.display = "none";
}

// Fetch History List
async function fetchHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/history`);
        if (!response.ok) throw new Error("Errore nel caricamento dello storico");
        
        const list = await response.json();
        renderHistoryTable(list);
    } catch (error) {
        console.error(error);
    }
}

function renderHistoryTable(list) {
    const tbody = document.getElementById("historyTableBody");
    const counter = document.getElementById("historyCounter");
    
    if (!tbody) return;
    
    if (list.length === 0) {
        tbody.innerHTML = `<tr><td colspan="10" class="table-empty">Nessuna analisi salvata nel database. Carica il tuo primo video!</td></tr>`;
        counter.textContent = "0 video analizzati";
        return;
    }
    
    counter.textContent = `${list.length} video analizzati`;
    tbody.innerHTML = "";
    
    list.forEach(item => {
        const dateStr = new Date(item.timestamp).toLocaleString("it-IT", {
            day: "2-digit",
            month: "2-digit",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit"
        });
        
        const viewsStr = item.views !== null ? item.views.toLocaleString() : "-";
        const retStr = item.retention_pct !== null ? `${item.retention_pct}%` : "-";
        const durationStr = `${item.duration.toFixed(1)}s`;
        const mlPredict = item.is_ml_predicted ? `<span class="text-emerald" style="font-weight:700;"><i class="fa-solid fa-brain"></i> Si</span>` : `<span class="text-muted">No</span>`;
        
        const tr = document.createElement("tr");
        tr.onclick = () => loadReport(item.id);
        
        // Define color classes for scores
        const viralColor = item.viral_score >= 80 ? "text-emerald" : item.viral_score >= 50 ? "text-amber" : "text-muted";
        
        tr.innerHTML = `
            <td><strong>${item.filename}</strong></td>
            <td>${dateStr}</td>
            <td>${durationStr}</td>
            <td><strong class="${viralColor}">${item.viral_score}</strong></td>
            <td>${item.hook_score}</td>
            <td>${item.pacing_score}</td>
            <td>${item.retention_risk}</td>
            <td>${viewsStr}</td>
            <td>${retStr}</td>
            <td>${mlPredict}</td>
        `;
        tbody.appendChild(tr);
    });
}

// Load Specific Analysis Report to UI
async function loadReport(id) {
    showLoader("Caricamento report...", "Recupero dei dati di analisi dal database locale.");
    try {
        const response = await fetch(`${API_BASE}/api/analysis/${id}`);
        if (!response.ok) throw new Error("Impossibile caricare il report.");
        
        const data = await response.json();
        displayReport(data);
    } catch (error) {
        alert(error.message);
    } finally {
        hideLoader();
    }
}

// Render Analysis Report in UI
function displayReport(data) {
    document.getElementById("resultsPlaceholder").style.display = "none";
    document.getElementById("reportContent").style.display = "block";
    
    // Set text fields
    document.getElementById("reportFilename").textContent = data.filename;
    
    const parsedDate = new Date(data.timestamp).toLocaleString("it-IT");
    document.getElementById("reportDate").textContent = `Analizzato il ${parsedDate}`;
    
    // Set Gauges
    setGaugeValue("viralScoreRing", "viralScoreVal", data.viral_score);
    setGaugeValue("hookScoreRing", "hookScoreVal", data.hook_score);
    setGaugeValue("pacingScoreRing", "pacingScoreVal", data.pacing_score);
    setGaugeValue("riskScoreRing", "riskScoreVal", data.retention_risk);
    
    // Viral Description Label
    const vDesc = document.getElementById("viralScoreDesc");
    if (data.viral_score >= 80) vDesc.textContent = "Potenziale Virale Altissimo";
    else if (data.viral_score >= 60) vDesc.textContent = "Potenziale Virale Buono";
    else if (data.viral_score >= 40) vDesc.textContent = "Potenziale Virale Moderato";
    else vDesc.textContent = "Potenziale Virale Basso";

    // ML Prediction Display
    const mlCallout = document.getElementById("mlPredictionCallout");
    if (data.is_ml_predicted && data.retention_pct !== null) {
        // If it was predicted and has retention, we display the user inputted one
        document.getElementById("mlPredictedVal").textContent = `${data.retention_pct}% (Effettiva)`;
        mlCallout.style.display = "flex";
    } else if (data.is_ml_predicted || (data.viral_score && data.is_ml_predicted)) {
        // Predicted by ML
        // We can display the predicted retention stored (or estimated)
        // For display: let's query backend or if we have it in response:
        // We saved response_data in database as 'viral_score' being calibrated. Let's see if we have 'retention_pct' pre-saved
        const modelPredictVal = data.retention_pct || (data.viral_score); 
        document.getElementById("mlPredictedVal").textContent = `${modelPredictVal.toFixed(1)}%`;
        mlCallout.style.display = "flex";
    } else {
        mlCallout.style.display = "none";
    }

    // Feedback List rendering
    const strengthsUl = document.getElementById("strengthsList");
    const improvementsUl = document.getElementById("improvementsList");
    
    strengthsUl.innerHTML = "";
    improvementsUl.innerHTML = "";
    
    const feedback = data.feedback; // {strengths: [], improvements: []}
    
    if (feedback.strengths && feedback.strengths.length > 0) {
        feedback.strengths.forEach(s => {
            const li = document.createElement("li");
            li.textContent = s;
            strengthsUl.appendChild(li);
        });
    } else {
        strengthsUl.innerHTML = "<li>Nessun punto di forza particolare rilevato.</li>";
    }
    
    if (feedback.improvements && feedback.improvements.length > 0) {
        feedback.improvements.forEach(imp => {
            const li = document.createElement("li");
            li.textContent = imp;
            improvementsUl.appendChild(li);
        });
    } else {
        improvementsUl.innerHTML = "<li>🔥 Nessuna opportunità di miglioramento urgente rilevata! L'editing rispetta i canoni virali.</li>";
    }
    
    // Features table
    document.getElementById("featDuration").textContent = `${data.duration.toFixed(1)}s`;
    document.getElementById("featCutsCount").textContent = data.cuts_count;
    document.getElementById("featCutsPerSec").textContent = data.cuts_per_second.toFixed(2);
    document.getElementById("featAvgScene").textContent = `${data.avg_scene_duration.toFixed(1)}s`;
    document.getElementById("featMotion").textContent = data.motion_intensity.toFixed(1);
    document.getElementById("featStability").textContent = `${data.visual_stability.toFixed(0)}%`;
    document.getElementById("featHookSpeed").textContent = `${data.hook_speed.toFixed(1)}s`;
    document.getElementById("featAvSync").textContent = `${data.av_sync_score.toFixed(0)}%`;
    document.getElementById("featAudioEnergy").textContent = data.audio_energy.toFixed(3);
    document.getElementById("featAudioSpikes").textContent = data.audio_spikes_count;

    // Scroll to results
    document.getElementById("resultsPanel").scrollIntoView({ behavior: "smooth" });
}

// Form Submission Handler
async function submitAnalysis(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById("videoFile");
    if (!fileInput.files || fileInput.files.length === 0) {
        alert("Seleziona prima un file video.");
        return;
    }
    
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    
    // Add optional fields if populated
    const views = document.getElementById("views").value;
    if (views) formData.append("views", parseInt(views));
    
    const likes = document.getElementById("likes").value;
    if (likes) formData.append("likes", parseInt(likes));
    
    const shares = document.getElementById("shares").value;
    if (shares) formData.append("shares", parseInt(shares));
    
    const saves = document.getElementById("saves").value;
    if (saves) formData.append("saves", parseInt(saves));
    
    const retention_pct = document.getElementById("retention_pct").value;
    if (retention_pct) formData.append("retention_pct", parseFloat(retention_pct));
    
    showLoader("Analisi Video in Corso...", "Estrazione delle feature visive (tagli, movimenti) e audio (beat, energia). L'elaborazione richiede solitamente 5-15 secondi.");
    
    try {
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Errore sconosciuto durante l'analisi del video");
        }
        
        const result = await response.json();
        
        // Display report & Refresh list
        displayReport(result);
        fetchHistory();
        fetchModelStatus();
        
        // Reset form file inputs
        document.getElementById("analyzeForm").reset();
        document.getElementById("selectedFileBadge").style.display = "none";
        
    } catch (error) {
        alert(`Errore nell'analisi: ${error.message}`);
    } finally {
        hideLoader();
    }
}

// Fetch Machine Learning model training status
async function fetchModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/model-status`);
        if (!response.ok) throw new Error();
        
        const data = await response.json();
        const stateText = document.getElementById("mlStateText");
        const trainBtn = document.getElementById("btnTrainModel");
        
        if (data.is_trained) {
            stateText.textContent = "ATTIVO (Random Forest)";
            stateText.className = "badge-state trained";
            trainBtn.innerHTML = `<i class="fa-solid fa-arrows-rotate"></i> Aggiorna`;
            trainBtn.style.display = "inline-flex";
        } else {
            stateText.textContent = `Non Addestrato (${data.labeled_count}/${data.required_labeled} video)`;
            stateText.className = "badge-state untrained";
            
            // Show Train button only if we have sufficient labeled videos
            if (data.labeled_count >= data.required_labeled) {
                trainBtn.style.display = "inline-flex";
            } else {
                trainBtn.style.display = "none";
            }
        }
    } catch (e) {
        console.error("Impossibile connettersi allo status ML.");
    }
}

// Trigger Machine Learning Model Training
async function trainMLModel() {
    showLoader("Addestramento in Corso...", "Il modello scikit-learn sta elaborando le caratteristiche dei video registrati per migliorare le stime di viralità.");
    try {
        const response = await fetch(`${API_BASE}/api/train`, { method: "POST" });
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || "Impossibile addestrare il modello");
        }
        
        alert(`Modello addestrato con successo! ${result.message}`);
        fetchModelStatus();
    } catch (e) {
        alert(e.message);
    } finally {
        hideLoader();
    }
}

// Loader Utilities
function showLoader(title, message) {
    const overlay = document.getElementById("loadingOverlay");
    document.getElementById("loaderTitle").textContent = title;
    document.getElementById("loaderMessage").textContent = message;
    overlay.classList.add("active");
}

function hideLoader() {
    const overlay = document.getElementById("loadingOverlay");
    overlay.classList.remove("active");
}

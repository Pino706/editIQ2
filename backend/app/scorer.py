def calculate_scores(features: dict) -> dict:
    """
    Computes heuristic scores (Viral, Hook, Pacing, Retention Risk)
    from extracted features and generates detailed Italian feedback.
    """
    hook_speed = features["hook_speed"]
    cuts_per_second = features["cuts_per_second"]
    avg_scene_duration = features["avg_scene_duration"]
    motion_intensity = features["motion_intensity"]
    visual_stability = features["visual_stability"]
    av_sync_score = features["av_sync_score"]
    audio_energy = features["audio_energy"]
    audio_spikes_count = features["audio_spikes_count"]

    # 1. HOOK SCORE (0-100)
    # Penalizes slower hooks. Ideal hook speed is <= 1.5 seconds.
    if hook_speed <= 1.0:
        base_hook = 100.0
    elif hook_speed <= 2.0:
        base_hook = 100.0 - (hook_speed - 1.0) * 20.0  # 80 to 100
    elif hook_speed <= 3.5:
        base_hook = 80.0 - (hook_speed - 2.0) * 30.0   # 35 to 80
    else:
        base_hook = max(10.0, 35.0 - (hook_speed - 3.5) * 10.0)
    
    # Adjust for motion intensity in the video
    motion_norm = min(15.0, motion_intensity) / 15.0
    hook_score = 0.8 * base_hook + 0.2 * (motion_norm * 100.0)
    hook_score = max(0.0, min(100.0, hook_score))

    # 2. PACING SCORE (0-100)
    # Bell-curve around optimal TikTok edits (1.0 to 2.2 cuts per second)
    if cuts_per_second >= 1.0 and cuts_per_second <= 2.2:
        cps_score = 100.0
    elif cuts_per_second < 1.0:
        cps_score = max(15.0, cuts_per_second * 100.0)
    else:  # cuts_per_second > 2.2 (too fast/chaotic)
        cps_score = max(40.0, 100.0 - (cuts_per_second - 2.2) * 25.0)

    # Sync score plays a big role in pacing quality
    pacing_score = 0.6 * cps_score + 0.4 * av_sync_score
    pacing_score = max(0.0, min(100.0, pacing_score))

    # 3. RETENTION RISK (0-100)
    # Risk increases with long scene duration, slow hook, or static content
    if avg_scene_duration <= 1.5:
        scene_risk = 10.0
    elif avg_scene_duration <= 4.0:
        scene_risk = 10.0 + (avg_scene_duration - 1.5) * 24.0  # 10 to 70
    else:
        scene_risk = min(95.0, 70.0 + (avg_scene_duration - 4.0) * 10.0)

    # Low motion means high retention risk
    if motion_intensity < 3.0:
        motion_risk = 85.0
    elif motion_intensity < 8.0:
        motion_risk = 85.0 - (motion_intensity - 3.0) * 12.0  # 25 to 85
    else:
        motion_risk = 25.0

    # Poor hook also means high skip rate (retention risk)
    hook_risk = 100.0 - hook_score

    retention_risk = 0.4 * scene_risk + 0.3 * motion_risk + 0.3 * hook_risk
    retention_risk = max(0.0, min(100.0, retention_risk))

    # 4. VIRAL SCORE (0-100)
    # Weighted average of editing quality signals
    retention_factor = 100.0 - retention_risk
    motion_factor = min(100.0, motion_intensity * 10.0)
    
    viral_score = (
        0.35 * hook_score +
        0.30 * pacing_score +
        0.20 * retention_factor +
        0.15 * (0.5 * av_sync_score + 0.5 * motion_factor)
    )
    viral_score = max(0.0, min(100.0, viral_score))

    # 5. GENERATE FEEDBACK (Strengths and Improvements)
    strengths = []
    improvements = []

    # Hook Speed Feedback
    if hook_speed <= 1.0:
        strengths.append("🔥 Gancio iniziale eccellente: il primo cambio visivo avviene in meno di 1 secondo, catturando subito l'attenzione.")
    elif hook_speed <= 2.0:
        strengths.append("👍 Buon ritmo iniziale: il primo taglio o cambio visivo avviene nei primi 2 secondi.")
    elif hook_speed <= 3.5:
        improvements.append(f"⚠️ Introduzione un po' lenta: il primo taglio avviene dopo {hook_speed:.1f} secondi. Cerca di anticiparlo nei primi 1.5s per evitare skip.")
    else:
        improvements.append(f"❌ Gancio iniziale debole: i primi secondi sono troppo statici (primo cambio a {hook_speed:.1f}s). Rischio elevato di abbandono immediato.")

    # Cuts Pacing Feedback
    if cuts_per_second < 0.5:
        improvements.append(f"❌ Ritmo di montaggio troppo lento ({cuts_per_second:.2f} tagli/sec). Aumenta la frequenza dei tagli per mantenere alta l'attenzione.")
    elif cuts_per_second < 1.0:
        improvements.append(f"⚠️ Montaggio rilassato ({cuts_per_second:.2f} tagli/sec). Adatto per spiegazioni, ma per edit musicali potrebbe risultare moscio.")
    elif cuts_per_second <= 2.2:
        strengths.append(f"🔥 Ottimo pacing di montaggio: {cuts_per_second:.2f} tagli al secondo creano un flusso dinamico ideale per TikTok.")
    else:
        improvements.append(f"⚠️ Tagli estremamente frequenti ({cuts_per_second:.2f} tagli/sec). Assicurati che l'azione sia comprensibile e non disorienti lo spettatore.")

    # Audio-Visual Sync Feedback
    if av_sync_score >= 70.0:
        strengths.append(f"🔥 Sincronizzazione audio-video fantastica ({av_sync_score:.1f}%): i tagli seguono perfettamente i beat musicali.")
    elif av_sync_score >= 40.0:
        strengths.append(f"👍 Buona sincronizzazione audio-video ({av_sync_score:.1f}%): molti tagli sono allineati al ritmo della musica.")
    elif audio_spikes_count > 0:
        improvements.append(f"⚠️ Sincronizzazione audio-video migliorabile ({av_sync_score:.1f}%): prova ad allineare più tagli con i picchi della traccia musicale per creare impatto.")

    # Motion Intensity Feedback
    if motion_intensity >= 8.0:
        strengths.append(f"🔥 Intensità di movimento elevata ({motion_intensity:.1f}): il video è pieno di energia visiva, ottimo per video di editing.")
    elif motion_intensity >= 3.0:
        strengths.append(f"👍 Livello di movimento bilanciato ({motion_intensity:.1f}): dinamico ma facile da seguire.")
    else:
        improvements.append(f"❌ Movimento visivo debole ({motion_intensity:.1f}): le inquadrature sono molto statiche. Aggiungi zoom, transizioni o effetti di camera shake.")

    # Stability Feedback
    if visual_stability < 45.0:
        improvements.append(f"⚠️ Stabilità visiva bassa ({visual_stability:.1f}%): il video presenta movimenti molto caotici o tremolanti. Assicurati che sia voluto.")
    elif visual_stability >= 80.0:
        strengths.append(f"👍 Ottima stabilità visiva: il movimento è fluido e controllato, pulito da vedere.")

    return {
        "viral_score": round(viral_score, 1),
        "hook_score": round(hook_score, 1),
        "pacing_score": round(pacing_score, 1),
        "retention_risk": round(retention_risk, 1),
        "feedback": {
            "strengths": strengths,
            "improvements": improvements
        }
    }

import sqlite3
import os
import json
import pandas as pd
from datetime import datetime

DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "editiq.db")

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            duration REAL NOT NULL,
            
            -- Manual optional performance metrics
            views INTEGER,
            likes INTEGER,
            shares INTEGER,
            saves INTEGER,
            retention_pct REAL,
            
            -- Extracted Visual Features
            cuts_count INTEGER NOT NULL,
            cuts_per_second REAL NOT NULL,
            avg_scene_duration REAL NOT NULL,
            motion_intensity REAL NOT NULL,
            visual_change_rate REAL NOT NULL,
            visual_stability REAL NOT NULL,
            hook_speed REAL NOT NULL,
            
            -- Extracted Audio Features
            audio_energy REAL NOT NULL,
            audio_spikes_count INTEGER NOT NULL,
            audio_pacing REAL NOT NULL,
            av_sync_score REAL NOT NULL,
            
            -- Computed Scores
            viral_score REAL NOT NULL,
            hook_score REAL NOT NULL,
            pacing_score REAL NOT NULL,
            retention_risk REAL NOT NULL,
            
            -- Feedback (JSON array serialized as text)
            feedback TEXT NOT NULL,
            
            -- Prediction flag
            is_ml_predicted INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def save_video_analysis(data: dict) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Ensure feedback is saved as JSON string
    feedback_str = json.dumps(data.get("feedback", []))
    timestamp = data.get("timestamp", datetime.now().isoformat())
    
    cursor.execute("""
        INSERT INTO videos (
            filename, timestamp, duration,
            views, likes, shares, saves, retention_pct,
            cuts_count, cuts_per_second, avg_scene_duration,
            motion_intensity, visual_change_rate, visual_stability, hook_speed,
            audio_energy, audio_spikes_count, audio_pacing, av_sync_score,
            viral_score, hook_score, pacing_score, retention_risk,
            feedback, is_ml_predicted
        ) VALUES (
            ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?
        )
    """, (
        data["filename"], timestamp, data["duration"],
        data.get("views"), data.get("likes"), data.get("shares"), data.get("saves"), data.get("retention_pct"),
        data["cuts_count"], data["cuts_per_second"], data["avg_scene_duration"],
        data["motion_intensity"], data["visual_change_rate"], data["visual_stability"], data["hook_speed"],
        data["audio_energy"], data["audio_spikes_count"], data["audio_pacing"], data["av_sync_score"],
        data["viral_score"], data["hook_score"], data["pacing_score"], data["retention_risk"],
        feedback_str, data.get("is_ml_predicted", 0)
    ))
    
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_all_analyses() -> list:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM videos ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        d = dict(r)
        d["feedback"] = json.loads(d["feedback"])
        results.append(d)
    return results

def get_analysis(video_id: int) -> dict:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        d = dict(row)
        d["feedback"] = json.loads(d["feedback"])
        return d
    return None

def get_all_analyses_df() -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM videos", conn)
    conn.close()
    return df

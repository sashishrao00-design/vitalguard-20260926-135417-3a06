"""Step 10 plumbing: SQLite persistence for readings, scores and alerts.

Standard library only (sqlite3), so the prototype runs anywhere Python does.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "vitalguard.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mrn TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    age_years REAL, sex TEXT, ward TEXT, unit TEXT,
    chronic_score REAL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    ts TEXT NOT NULL,
    hr REAL, spo2 REAL, rr REAL, sbp REAL, dbp REAL, temp_c REAL,
    signal_quality REAL, source TEXT,
    risk_score REAL, tier TEXT, news2 REAL,
    explanation TEXT, flags TEXT
);
CREATE INDEX IF NOT EXISTS idx_readings_patient ON readings(patient_id, ts DESC);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    reading_id INTEGER REFERENCES readings(id) ON DELETE CASCADE,
    ts TEXT NOT NULL, tier TEXT NOT NULL, message TEXT NOT NULL,
    acknowledged_at TEXT, acknowledged_by TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path: Path | None = None):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path or DB_PATH), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init(path: Path | None = None) -> None:
    with connect(path) as con:
        con.executescript(SCHEMA)


def reset(path: Path | None = None) -> None:
    p = path or DB_PATH
    if p.exists():
        p.unlink()
    init(p)


# ---------------------------------------------------------------------------
def add_patient(mrn: str, name: str, age_years=None, sex="M", ward="General", unit="ward",
                chronic_score=0) -> dict:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO patients(mrn,name,age_years,sex,ward,unit,chronic_score,created_at) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(mrn) DO UPDATE SET name=excluded.name, "
            "age_years=excluded.age_years, sex=excluded.sex, ward=excluded.ward, "
            "unit=excluded.unit, chronic_score=excluded.chronic_score RETURNING id",
            (mrn, name, age_years, sex, ward, unit, chronic_score, now()),
        )
        pid = cur.fetchone()["id"]
        row = con.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
        return dict(row)


def list_patients() -> list[dict]:
    with connect() as con:
        rows = con.execute(
            """SELECT p.*, r.ts AS last_ts, r.hr, r.spo2, r.rr, r.sbp, r.dbp, r.temp_c,
                      r.risk_score, r.tier, r.news2, r.signal_quality,
                      (SELECT COUNT(*) FROM readings x WHERE x.patient_id=p.id) AS n_readings,
                      (SELECT COUNT(*) FROM alerts a WHERE a.patient_id=p.id AND a.acknowledged_at IS NULL)
                          AS open_alerts
               FROM patients p
               LEFT JOIN readings r ON r.id = (SELECT id FROM readings
                                                WHERE patient_id=p.id ORDER BY ts DESC LIMIT 1)
               ORDER BY COALESCE(r.risk_score,-1) DESC, p.name"""
        ).fetchall()
        return [dict(r) for r in rows]


def add_reading(patient_id: int, vitals: dict, explanation: dict | None,
                signal_quality: float | None = None, source: str = "phone") -> dict:
    ex = explanation or {}
    with connect() as con:
        cur = con.execute(
            """INSERT INTO readings(patient_id,ts,hr,spo2,rr,sbp,dbp,temp_c,signal_quality,source,
                                    risk_score,tier,news2,explanation,flags)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                patient_id, now(),
                vitals.get("hr"), vitals.get("spo2"), vitals.get("rr"),
                vitals.get("sbp"), vitals.get("dbp"), vitals.get("temp_c"),
                signal_quality, source,
                ex.get("risk_score"), ex.get("tier"), ex.get("news2"),
                json.dumps(ex), json.dumps(ex.get("red_flags", [])),
            ),
        )
        rid = cur.lastrowid
        tier = ex.get("tier")
        # An alert means "this patient got WORSE", not "another reading was taken":
        # suppress if an unacknowledged alert already covers this tier or higher,
        # otherwise a 6-hourly obs schedule buries the nurse in duplicates.
        rank = {"minimal": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
        if tier in ("critical", "high"):
            prior = con.execute(
                """SELECT tier FROM alerts WHERE patient_id=? AND acknowledged_at IS NULL
                   ORDER BY ts DESC LIMIT 1""", (patient_id,)).fetchone()
            escalate = prior is None or rank.get(prior["tier"], 0) < rank.get(tier, 0)
            if escalate:
                msg = (f"{tier.upper()} deterioration risk {ex.get('risk_percent')}% - "
                       f"{(ex.get('advice') or {}).get('action', 'review')}")
                con.execute(
                    "INSERT INTO alerts(patient_id,reading_id,ts,tier,message) VALUES(?,?,?,?,?)",
                    (patient_id, rid, now(), tier, msg),
                )
        row = con.execute("SELECT * FROM readings WHERE id=?", (rid,)).fetchone()
        return dict(row)


def history(patient_id: int, limit: int = 50) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM readings WHERE patient_id=? ORDER BY ts DESC LIMIT ?",
            (patient_id, limit),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["explanation"] = json.loads(d.get("explanation") or "{}")
            except json.JSONDecodeError:
                d["explanation"] = {}
            d.pop("flags", None)
            out.append(d)
        return out


def alerts(limit: int = 100, open_only: bool = False) -> list[dict]:
    q = """SELECT a.*, p.name, p.mrn, p.ward FROM alerts a
           JOIN patients p ON p.id = a.patient_id"""
    if open_only:
        q += " WHERE a.acknowledged_at IS NULL"
    q += " ORDER BY a.ts DESC LIMIT ?"
    with connect() as con:
        return [dict(r) for r in con.execute(q, (limit,)).fetchall()]


def ack_alert(alert_id: int, by: str = "dr") -> bool:
    with connect() as con:
        cur = con.execute(
            "UPDATE alerts SET acknowledged_at=?, acknowledged_by=? WHERE id=? AND acknowledged_at IS NULL",
            (now(), by, alert_id),
        )
        return cur.rowcount > 0


def stats() -> dict:
    with connect() as con:
        p = con.execute("SELECT COUNT(*) c FROM patients").fetchone()["c"]
        r = con.execute("SELECT COUNT(*) c FROM readings").fetchone()["c"]
        a = con.execute("SELECT COUNT(*) c FROM alerts WHERE acknowledged_at IS NULL").fetchone()["c"]
        avg = con.execute("SELECT AVG(risk_score) v, AVG(hr) hr, AVG(spo2) spo2 FROM readings").fetchone()
    return {
        "patients": p, "readings": r, "open_alerts": a,
        "mean_risk": None if avg["v"] is None else round(float(avg["v"]), 4),
        "mean_hr": None if avg["hr"] is None else round(float(avg["hr"]), 1),
        "mean_spo2": None if avg["spo2"] is None else round(float(avg["spo2"]), 1),
        "db_path": str(DB_PATH),
    }

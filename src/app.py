import os
import io
import re
import zipfile
import mimetypes
from pathlib import Path
from datetime import datetime

from flask import (
    Flask, render_template, send_from_directory, send_file,
    abort, request
)
from PIL import Image

# =======================
# Konfigurasi
# =======================
LOGS_ROOT = Path(os.environ.get("LOG_DIR", "logs")).resolve()
THUMB_ROOT = Path(os.environ.get("THUMB_DIR", ".thumbs")).resolve()
THUMB_ROOT.mkdir(parents=True, exist_ok=True)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif"}
VID_EXTS = {".mp4", ".webm", ".avi", ".mkv", ".mov", ".wmv"}

app = Flask(__name__)

# =======================
# Utils
# =======================
def _inside(child: Path, root: Path) -> bool:
    child = child.resolve(strict=False)
    root = root.resolve(strict=False)
    return root == child or root in child.parents

def safe_path(rel: str) -> Path:
    p = (LOGS_ROOT / rel).resolve()
    if not _inside(p, LOGS_ROOT):
        abort(404)
    return p

def list_days():
    if not LOGS_ROOT.exists():
        return []
    days = [d for d in LOGS_ROOT.iterdir() if d.is_dir() and DATE_RE.match(d.name)]
    # newest first
    days.sort(key=lambda p: p.name, reverse=True)
    return days

def list_sessions(day_dir: Path):
    if not day_dir.exists():
        return []
    sessions = [d for d in day_dir.iterdir() if d.is_dir()]
    sessions.sort(key=lambda p: p.name, reverse=True)
    return sessions

def ensure_thumb(rel_path: str, max_dim=320) -> Path:
    src = safe_path(rel_path)
    if src.suffix.lower() not in IMG_EXTS:
        abort(404)
    rel = src.relative_to(LOGS_ROOT)
    out = (THUMB_ROOT / rel).with_suffix(".jpg")
    out.parent.mkdir(parents=True, exist_ok=True)

    src_m = src.stat().st_mtime
    if out.exists() and out.stat().st_mtime >= src_m:
        return out

    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim))
        im.save(out, "JPEG", quality=85)
    os.utime(out, (src_m, src_m))
    return out

@app.template_filter("tsfmt")
def tsfmt(ts):
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except:  # noqa
        return ""

def _parse_ymd(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

# =======================
# Routes
# =======================
@app.route("/")
def index():
    # ambil parameter range dari query string
    start_s = (request.args.get("start") or "").strip()
    end_s   = (request.args.get("end") or "").strip()

    sdate = _parse_ymd(start_s) if start_s else None
    edate = _parse_ymd(end_s) if end_s else None

    # kalau user kebalik (start > end), kita swap biar tetap inklusif & ramah
    if sdate and edate and sdate > edate:
        sdate, edate = edate, sdate
        start_s, end_s = end_s, start_s

    # ambil semua folder tanggal seperti biasa
    days = list_days()

    # filter jika ada batas bawah/atas
    if sdate or edate:
        filtered = []
        for d in days:
            ddate = _parse_ymd(d.name)  # nama folder = YYYY-MM-DD
            if not ddate:
                continue
            if sdate and ddate < sdate:
                continue
            if edate and ddate > edate:
                continue
            filtered.append(d)
        days = filtered

    return render_template("index.html", dates=days, start=start_s, end=end_s)

@app.route("/browse/<date>")
def browse_date(date):
    if not DATE_RE.match(date):
        abort(404)
    day_dir = safe_path(date)
    sessions = list_sessions(day_dir)
    return render_template("browse.html", date=date, sessions=sessions)

@app.route("/session/<date>/<session>")
def browse_session(date, session):
    if not DATE_RE.match(date):
        abort(404)
    session_dir = safe_path(f"{date}/{session}")
    if not session_dir.exists():
        abort(404)

    # snapshots
    snaps_dir = session_dir / "snapshots"
    snapshots = []
    if snaps_dir.exists():
        snapshots = [p.name for p in sorted(snaps_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]

    # videos & logs
    videos = [p.name for p in sorted(session_dir.iterdir()) if p.suffix.lower() in VID_EXTS]
    # Tampilkan mp4 dulu
    videos.sort(key=lambda n: (0 if n.lower().endswith(".mp4") else 1, n.lower()))
    txts = [p.name for p in sorted(session_dir.iterdir()) if p.suffix.lower() in {".txt", ".log"}]

    # MIME types for <source type=...>
    video_types = {v: (mimetypes.guess_type(v)[0] or "application/octet-stream") for v in videos}

    # --- pagination snapshot ---
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 24, type=int)  # 12/24/48/96
    per_page = max(1, min(per_page, 200))

    total = len(snapshots)
    pages = (total + per_page - 1) // per_page if total else 1
    page = max(1, min(page, pages))
    start = (page - 1) * per_page
    end = start + per_page
    snapshots_page = snapshots[start:end]

    return render_template(
        "session.html",
        date=date, session=session,
        snapshots=snapshots_page, videos=videos, txts=txts,
        video_types=video_types,
        page=page, pages=pages, per_page=per_page, total_snaps=total
    )

# ---- file/asset serving ----
@app.route("/logs/<path:filename>")
def serve_file(filename):
    p = safe_path(filename)
    return send_from_directory(p.parent, p.name, as_attachment=False)

@app.route("/download/<path:filename>")
def download_file(filename):
    p = safe_path(filename)
    return send_file(str(p), as_attachment=True, download_name=p.name)

@app.route("/thumb/<path:filename>")
def thumb(filename):
    t = ensure_thumb(filename)
    return send_file(str(t), mimetype="image/jpeg")

@app.route("/zip/<date>/<session>")
def zip_session(date, session):
    if not DATE_RE.match(date):
        abort(404)
    session_dir = safe_path(f"{date}/{session}")
    if not session_dir.exists() or not session_dir.is_dir():
        abort(404)

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(session_dir):
            for name in files:
                fp = Path(root) / name
                arc = fp.relative_to(LOGS_ROOT)
                zf.write(fp, arc.as_posix())
    mem.seek(0)
    return send_file(mem, mimetype="application/zip",
                     as_attachment=True, download_name=f"{session}.zip")

# =======================
# Main
# =======================
if __name__ == "__main__":
    # Bind ke semua interface (akses via hotspot)
    app.run(host="0.0.0.0", port=5000, debug=False)
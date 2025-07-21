#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Music Organizer & Tag Fixer

Features:
1. Interactive prompts for root & output folders, copy/move mode.
2. Network check up‐front (skips online lookups if offline).
3. Counts files, then shows Rich progress bar with ETA.
4. Duplicate detection (SHA-1) with prompt to delete.
5. Full metadata support for .mp3, .flac, .m4a/.mp4, .ogg, .wav.
6. Fetch lyrics from Genius (requires GENIUS_TOKEN).
7. Fetch cover from iTunes if missing.
8. Copies or moves processed files to an `output` folder, structured by Artist/Title–Album.
9. Logs to console (Rich), to ~/.music_organizer/organizer.log, and to ~/.music_organizer/report.csv.
"""

import os
import sys
import subprocess
import sqlite3
import logging
import hashlib
import argparse
import configparser
import csv
import shutil
from pathlib import Path
from typing import Optional

# -----------------------------------------------------------------------------
# SECTION 0: Auto‐install missing dependencies
# -----------------------------------------------------------------------------
REQUIRED_PACKAGES = [
    "mutagen", "requests", "lyricsgenius", "rich",
    "Pillow", "send2trash"
]

def install_missing_packages():
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[INSTALLING] {pkg} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

install_missing_packages()

# -----------------------------------------------------------------------------
# SECTION 1: Imports (safe after installing deps)
# -----------------------------------------------------------------------------
import mutagen
from mutagen.id3 import ID3, TIT2, TPE1, TALB, USLT, APIC
from mutagen.easyid3 import EasyID3
from mutagen.flac import FLAC, Picture
from mutagen.mp4 import MP4, MP4Cover
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

import requests
import lyricsgenius
from send2trash import send2trash

from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, TimeElapsedColumn, TimeRemainingColumn
)
from rich.prompt import Prompt

from PIL import Image
from io import BytesIO

# -----------------------------------------------------------------------------
# SECTION 2: Configuration & Logger Setup
# -----------------------------------------------------------------------------
APP_DIR    = Path.home() / ".music_organizer"
DB_PATH    = APP_DIR / "processed_songs.db"
LOG_FILE   = APP_DIR / "organizer.log"
REPORT_CSV = APP_DIR / "report.csv"
CONFIG_INI = APP_DIR / "config.ini"
APP_DIR.mkdir(exist_ok=True)

config = configparser.ConfigParser()
if CONFIG_INI.exists():
    config.read(CONFIG_INI)

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[
        RichHandler(show_time=True, show_level=False, rich_tracebacks=True),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("music_organizer")

GENIUS_TOKEN = (
    os.getenv("GENIUS_TOKEN") or
    config.get("lyrics", "genius_token", fallback=None)
)
if GENIUS_TOKEN:
    genius_client = lyricsgenius.Genius(GENIUS_TOKEN, timeout=15, retries=3)
else:
    genius_client = None

# -----------------------------------------------------------------------------
# SECTION 3: Database Helpers
# -----------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS processed (
    id INTEGER PRIMARY KEY,
    file_hash TEXT UNIQUE,
    original_path TEXT,
    artist TEXT,
    title TEXT,
    album TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_TABLE_SQL)
    return conn

def is_already_processed(conn, file_hash: str) -> bool:
    cur = conn.execute("SELECT 1 FROM processed WHERE file_hash = ?", (file_hash,))
    return cur.fetchone() is not None

def mark_as_processed(conn, file_hash, original_path, artist, title, album):
    conn.execute(
        "INSERT OR IGNORE INTO processed (file_hash, original_path, artist, title, album) VALUES (?,?,?,?,?)",
        (file_hash, original_path, artist, title, album)
    )
    conn.commit()

# -----------------------------------------------------------------------------
# SECTION 4: Utility Functions
# -----------------------------------------------------------------------------
def file_sha1(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def sanitize_filename(s: str) -> str:
    keep = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(ch if ch in keep else "_" for ch in s).strip()

def infer_tags_from_filename(path: Path):
    """Try 'Artist - Title [Album]'."""
    name = path.stem
    artist = title = album = None
    if " - " in name:
        artist, rest = name.split(" - ", 1)
        artist = artist.strip()
        if "[" in rest and "]" in rest:
            t, a = rest.split("[", 1)
            title = t.strip()
            album = a.strip(" ]")
        else:
            title = rest.strip()
    return artist, title, album

def fetch_lyrics(artist: str, title: str) -> Optional[str]:
    if not genius_client:
        return None
    try:
        song = genius_client.search_song(title, artist)
        return song.lyrics if song else None
    except Exception as e:
        logger.debug(f"Lyric fetch error: {e}")
    return None

def fetch_itunes_cover(artist: str, title: str) -> Optional[bytes]:
    try:
        q = requests.utils.requote_uri(f"{artist} {title}")
        url = f"https://itunes.apple.com/search?term={q}&entity=song&limit=1"
        r = requests.get(url, timeout=10); r.raise_for_status()
        items = r.json().get("results", [])
        if items:
            art = items[0].get("artworkUrl100", "")
            hi = art.replace("100x100", "600x600")
            return requests.get(hi, timeout=10).content
    except Exception:
        return None
    return None

def extract_existing_cover(path: Path) -> Optional[bytes]:
    suf = path.suffix.lower()
    try:
        if suf == ".mp3":
            tags = ID3(str(path))
            for v in tags.values():
                if isinstance(v, APIC):
                    return v.data
        elif suf == ".flac":
            audio = FLAC(str(path))
            for pic in audio.pictures:
                return pic.data
        elif suf in (".m4a", ".mp4"):
            audio = MP4(str(path))
            covrs = audio.tags.get("covr")
            if covrs:
                return covrs[0]
    except Exception:
        return None
    return None

def embed_tags_and_cover(
    path: Path,
    artist: str,
    title: str,
    album: Optional[str],
    lyrics: Optional[str],
    cover_data: Optional[bytes]
):
    suf = path.suffix.lower()
    if cover_data is None:
        cover_data = fetch_itunes_cover(artist, title)

    if suf == ".mp3":
        audio = ID3(str(path))
        audio.delall("TPE1"); audio.add(TPE1(encoding=3, text=artist))
        audio.delall("TIT2"); audio.add(TIT2(encoding=3, text=title))
        if album:
            audio.delall("TALB"); audio.add(TALB(encoding=3, text=album))
        if lyrics:
            audio.delall("USLT"); audio.add(USLT(encoding=3, desc="", text=lyrics))
        if cover_data:
            audio.delall("APIC")
            audio.add(APIC(encoding=3, mime="image/jpeg", type=3, desc="Cover", data=cover_data))
        audio.save()

    elif suf == ".flac":
        audio = FLAC(str(path))
        audio["artist"] = artist; audio["title"] = title
        if album: audio["album"] = album
        if lyrics: audio["lyrics"] = lyrics
        if cover_data:
            audio.clear_pictures()
            pic = Picture()
            pic.type = 3; pic.mime = "image/jpeg"; pic.desc = "Cover"; pic.data = cover_data
            audio.add_picture(pic)
        audio.save()

    elif suf in (".m4a", ".mp4"):
        audio = MP4(str(path))
        audio.tags["\xa9ART"] = [artist]
        audio.tags["\xa9nam"] = [title]
        if album: audio.tags["\xa9alb"] = [album]
        if lyrics: audio.tags["\xa9lyr"] = [lyrics]
        if cover_data:
            audio.tags["covr"] = [MP4Cover(cover_data, imageformat=MP4Cover.FORMAT_JPEG)]
        audio.save()

    elif suf == ".ogg":
        audio = OggVorbis(str(path))
        audio["artist"] = artist; audio["title"] = title
        if album: audio["album"] = album
        if lyrics: audio["lyrics"] = lyrics
        audio.save()

    elif suf == ".wav":
        # WAV tagging is limited; skip
        pass

# -----------------------------------------------------------------------------
# SECTION 5: Process One File
# -----------------------------------------------------------------------------
def process_file(
    src: Path,
    conn,
    console: Console,
    writer: csv.writer,
    output_root: Path,
    copy_mode: bool,
    network_ok: bool
):
    file_hash = file_sha1(src)
    if is_already_processed(conn, file_hash):
        console.print(f"[yellow]Duplicate found:[/yellow] {src}")
        if Prompt.ask("Delete original?", choices=["y", "n"], default="n") == "y":
            src.unlink(missing_ok=True)
            console.log(f"[red]Deleted[/red] {src}")
        return

    # Read existing tags
    artist = title = album = None
    try:
        easy = EasyID3(str(src))
        artist = easy.get("artist", [None])[0]
        title  = easy.get("title",  [None])[0]
        album  = easy.get("album",  [None])[0]
    except Exception:
        pass

    # Fallback filename inference
    ia, it, ialb = infer_tags_from_filename(src)
    artist = artist or ia
    title  = title or it
    album  = album or ialb

    if not artist or not title:
        logger.warning(f"Skipping (no artist/title): {src}")
        return

    cover  = extract_existing_cover(src)
    lyrics = fetch_lyrics(artist, title) if (network_ok and genius_client) else None

    # Build target folder & filename
    art_dir = sanitize_filename(artist)
    tit_fn  = sanitize_filename(title)
    alb_fn  = sanitize_filename(album) if album else ""
    ext     = src.suffix.lower()
    new_name = f"{tit_fn} – {alb_fn}{ext}" if alb_fn else f"{tit_fn}{ext}"

    dest_dir  = output_root / art_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / new_name

    # Copy or Move
    if copy_mode:
        console.log(f"[cyan]Copying[/cyan] {src.name} → {new_name}")
        shutil.copy2(str(src), str(dest_path))
        action = "COPIED"
    else:
        console.log(f"[cyan]Moving[/cyan] {src.name} → {new_name}")
        src.rename(dest_path)
        action = "MOVED"

    # Embed tags+cover+lyrics on dest_path
    embed_tags_and_cover(dest_path, artist, title, album, lyrics, cover)

    # Mark & log
    mark_as_processed(conn, file_hash, str(src), artist, title, album or "")
    writer.writerow([str(src), str(dest_path), artist, title, album or "", action])

# -----------------------------------------------------------------------------
# SECTION 6: Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Organize & tag your music collection")
    parser.add_argument("--root", "-r",
        default=config.get("general", "root", fallback=None),
        help="Path to unsorted music folder"
    )
    args = parser.parse_args()

    console = Console()

    # 1) Determine root folder
    if args.root:
        root = Path(args.root).expanduser()
    else:
        root = None

    while not root or not root.is_dir():
        console.print(f"[yellow]Invalid root folder:[/yellow] {root}")
        inp = Prompt.ask("Enter full path to your music folder")
        root = Path(inp).expanduser()

    # 2) Output folder
    default_out = root / "output"
    out = Prompt.ask("Output folder?", default=str(default_out))
    output_root = Path(out).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    # 3) Copy vs Move
    mode = Prompt.ask("Keep originals (copy) or cut (move)?", choices=["copy","move"], default="copy")
    copy_mode = (mode == "copy")

    # 4) Network check
    with console.status("[bold cyan]Checking network...[/bold cyan]", spinner="dots"):
        try:
            requests.get("https://www.google.com", timeout=5).raise_for_status()
            network_ok = True
        except Exception:
            network_ok = False

    if not network_ok:
        logger.warning("Offline: skipping Genius lyrics & iTunes cover fetch.")

    # 5) Gather all files first
    exts = {".mp3", ".flac", ".m4a", ".mp4", ".ogg", ".wav", ".aac"}
    all_files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    total = len(all_files)
    if total == 0:
        console.print("[red]No audio files found![/red]")
        sys.exit(1)

    # 6) Prepare DB & CSV report
    conn = get_db_connection()
    new_report = not REPORT_CSV.exists()
    rpt = open(REPORT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(rpt)
    if new_report:
        writer.writerow(["Old Path","New Path","Artist","Title","Album","Action"])

    # 7) Process with Rich Progress
    console.print(Panel.fit(
        f"🎵 Organizer\nRoot: [bold]{root}[/bold]\nOutput: [bold]{output_root}[/bold]\n"
        f"Mode: [bold]{mode}[/bold]\nFiles: [bold]{total}[/bold]\n",
        title="Music Organizer", style="magenta"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing files...", total=total)
        for src in all_files:
            progress.update(task, description=f"▶️ {src.name}")
            process_file(
                src, conn, console, writer,
                output_root, copy_mode, network_ok
            )
            progress.advance(task)

    rpt.close()
    console.print(Panel(
        f"[green]All done![/green]\n• Log: {LOG_FILE}\n• Report: {REPORT_CSV}",
        title="🎉 Finished"
    ))

if __name__ == "__main__":
    main()

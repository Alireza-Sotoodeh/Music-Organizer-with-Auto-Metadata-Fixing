#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Music Organizer & Tag Fixer

Features:
1. Interactive prompts for root & output folders, copy/move mode.
2. Network check up‐front (skips online lookups if offline).
3. Uses Rich transient progress bars for dependency installation and file processing.
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
import hashlib
import argparse
import configparser
import csv
import shutil
from pathlib import Path
from typing import Optional, Tuple, List
import logging 

# -----------------------------------------------------------------------------
# SECTION 0: Preliminary Setup
# -----------------------------------------------------------------------------
APP_DIR    = Path.home() / ".music_organizer"
DB_PATH    = APP_DIR / "processed_songs.db"
LOG_FILE   = APP_DIR / "organizer.log"
REPORT_CSV = APP_DIR / "report.csv"
CONFIG_INI = APP_DIR / "config.ini"
APP_DIR.mkdir(exist_ok=True)

# Read config if present
config = configparser.ConfigParser()
if CONFIG_INI.exists():
    config.read(CONFIG_INI)

# -----------------------------------------------------------------------------
# SECTION 1: Install Missing Packages (with Rich Progress)
# -----------------------------------------------------------------------------
REQUIRED_PACKAGES = [
    "mutagen", "requests", "lyricsgenius", "rich",
    "Pillow", "send2trash"
]

def install_missing_packages():
    """
    Attempt to import each package in REQUIRED_PACKAGES.
    If ImportError is raised, pip-install it while displaying
    a transient Rich progress bar.
    """
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
    missing: List[str] = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return

    console = __import__("rich").console.Console()
    with Progress(
        SpinnerColumn(),
        TextColumn("[blue]Installing[/blue] {task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task("packages", total=len(missing))
        for pkg in missing:
            progress.update(task, description=pkg)
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
            progress.advance(task)

install_missing_packages()

# -----------------------------------------------------------------------------
# SECTION 2: Imports (safe after pip installs)
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.prompt import Prompt

from PIL import Image
from io import BytesIO

# -----------------------------------------------------------------------------
# SECTION 3: Logging & Genius Client
# -----------------------------------------------------------------------------
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
genius_client = lyricsgenius.Genius(GENIUS_TOKEN, timeout=15, retries=3) if GENIUS_TOKEN else None

# -----------------------------------------------------------------------------
# SECTION 4: Database Helpers
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
)"""

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn

def is_already_processed(conn: sqlite3.Connection, file_hash: str) -> bool:
    cur = conn.execute("SELECT 1 FROM processed WHERE file_hash = ?", (file_hash,))
    return cur.fetchone() is not None

def mark_as_processed(
    conn: sqlite3.Connection,
    file_hash: str,
    original_path: str,
    artist: str,
    title: str,
    album: str
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO processed (file_hash, original_path, artist, title, album) VALUES (?,?,?,?,?)",
        (file_hash, original_path, artist, title, album)
    )
    conn.commit()

# -----------------------------------------------------------------------------
# SECTION 5: Utility Functions
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

def infer_tags_from_filename(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Attempt to parse 'Artist - Title [Album]' from the filename.
    """
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
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("results", [])
        if items:
            art_url = items[0].get("artworkUrl100", "").replace("100x100", "600x600")
            return requests.get(art_url, timeout=10).content
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
) -> None:
    """
    Write metadata (artist, title, album, lyrics, cover image) into the file at `path`.
    """
    if cover_data is None:
        cover_data = fetch_itunes_cover(artist, title)

    suf = path.suffix.lower()
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
        audio["artist"], audio["title"] = artist, title
        if album: audio["album"] = album
        if lyrics: audio["lyrics"] = lyrics
        if cover_data:
            audio.clear_pictures()
            pic = Picture()
            pic.type, pic.mime, pic.desc, pic.data = 3, "image/jpeg", "Cover", cover_data
            audio.add_picture(pic)
        audio.save()

    elif suf in (".m4a", ".mp4"):
        audio = MP4(str(path))
        audio.tags["\xa9ART"], audio.tags["\xa9nam"] = [artist], [title]
        if album: audio.tags["\xa9alb"] = [album]
        if lyrics: audio.tags["\xa9lyr"] = [lyrics]
        if cover_data:
            audio.tags["covr"] = [MP4Cover(cover_data, imageformat=MP4Cover.FORMAT_JPEG)]
        audio.save()

    elif suf == ".ogg":
        audio = OggVorbis(str(path))
        audio["artist"], audio["title"] = artist, title
        if album: audio["album"] = album
        if lyrics: audio["lyrics"] = lyrics
        audio.save()

    # WAV tagging is very limited—skipped.

# -----------------------------------------------------------------------------
# SECTION 6: Process a Single File
# -----------------------------------------------------------------------------
def process_file(
    src: Path,
    conn: sqlite3.Connection,
    console: Console,
    writer: csv.writer,
    output_root: Path,
    copy_mode: bool,
    network_ok: bool
) -> None:
    fhash = file_sha1(src)
    if is_already_processed(conn, fhash):
        console.print(f"[yellow]Duplicate found:[/yellow] {src.name}")
        if Prompt.ask("Delete original?", choices=["y", "n"], default="n") == "y":
            src.unlink(missing_ok=True)
            console.log(f"[red]Deleted[/red] {src}")
        return

    # Read existing metadata
    artist = title = album = None
    try:
        easy = EasyID3(str(src))
        artist = easy.get("artist", [None])[0]
        title  = easy.get("title",  [None])[0]
        album  = easy.get("album",  [None])[0]
    except Exception:
        pass

    # Fallback: infer from filename
    ia, it, ialb = infer_tags_from_filename(src)
    artist = artist or ia
    title  = title  or it
    album  = album  or ialb

    if not artist or not title:
        logger.warning(f"Skipping (no artist/title): {src}")
        return

    cover  = extract_existing_cover(src)
    lyrics = fetch_lyrics(artist, title) if (network_ok and genius_client) else None

    # Prepare destination
    artist_dir = sanitize_filename(artist)
    title_fn   = sanitize_filename(title)
    album_fn   = sanitize_filename(album) if album else ""
    ext        = src.suffix.lower()
    new_name   = f"{title_fn} – {album_fn}{ext}" if album_fn else f"{title_fn}{ext}"

    dest_dir  = output_root / artist_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / new_name

    # Copy or move
    if copy_mode:
        console.log(f"[cyan]Copying[/cyan] {src.name} → {new_name}")
        shutil.copy2(str(src), str(dest_path))
        action = "COPIED"
    else:
        console.log(f"[cyan]Moving[/cyan] {src.name} → {new_name}")
        src.rename(dest_path)
        action = "MOVED"

    # Embed metadata
    embed_tags_and_cover(dest_path, artist, title, album, lyrics, cover)

    # Record in DB + CSV
    mark_as_processed(conn, fhash, str(src), artist, title, album or "")
    writer.writerow([str(src), str(dest_path), artist, title, album or "", action])

# -----------------------------------------------------------------------------
# SECTION 7: Main Routine
# -----------------------------------------------------------------------------
def main():
    console = Console()

    # 1) CLI args
    parser = argparse.ArgumentParser(description="Organize & tag your music collection")
    parser.add_argument("--root", "-r",
                        default=config.get("general", "root", fallback=None),
                        help="Path to unsorted music folder")
    args = parser.parse_args()

    # 2) Determine root folder
    root = Path(args.root).expanduser() if args.root else None
    while not root or not root.is_dir():
        console.print(f"[red]Invalid root folder:[/red] {root}")
        inp = Prompt.ask("Enter full path to your music folder")
        root = Path(inp).expanduser()

    # 3) Output folder & mode
    default_out = root / "output"
    out        = Prompt.ask("Output folder?", default=str(default_out))
    output_root = Path(out).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    mode      = Prompt.ask("Keep originals (copy) or cut (move)?",
                           choices=["copy", "move"], default="copy")
    copy_mode = (mode == "copy")

    # 4) Network check (spinner)
    with console.status("[bold cyan]Checking network...[/bold cyan]", spinner="dots"):
        try:
            requests.get("https://www.google.com", timeout=5).raise_for_status()
            network_ok = True
        except Exception:
            network_ok = False

    if not network_ok:
        logger.warning("Offline: skipping Genius lyrics & iTunes cover fetch.")

    # 5) Collect all audio files
    exts = {".mp3", ".flac", ".m4a", ".mp4", ".ogg", ".wav", ".aac"}
    all_files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    total = len(all_files)
    if total == 0:
        console.print("[red]No audio files found![/red]")
        sys.exit(1)

    # 6) Prepare DB & CSV
    conn = get_db_connection()
    is_new = not REPORT_CSV.exists()
    rpt = open(REPORT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(rpt)
    if is_new:
        writer.writerow(["Old Path","New Path","Artist","Title","Album","Action"])

    # 7) Show summary panel
    console.print(Panel.fit(
        f"🎵 Organizer\nRoot: [bold]{root}[/bold]\n"
        f"Output: [bold]{output_root}[/bold]\n"
        f"Mode: [bold]{mode}[/bold]\nFiles: [bold]{total}[/bold]",
        title="Music Organizer", style="magenta"
    ))

    # 8) Process with a single transient Rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task("Processing files...", total=total)
        for src in all_files:
            progress.update(task, description=f"▶️ {src.name}")
            process_file(src, conn, console, writer, output_root, copy_mode, network_ok)
            progress.advance(task)

    rpt.close()
    console.print(Panel(
        f"[green]All done![/green]\n• Log: {LOG_FILE}\n• Report: {REPORT_CSV}",
        title="🎉 Finished"
    ))


if __name__ == "__main__":
    main()

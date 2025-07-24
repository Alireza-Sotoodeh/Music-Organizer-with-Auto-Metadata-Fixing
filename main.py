#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Music File Organizer v2.4
-----------------------------------

Features:
1. Always asks for root & output folders with smart defaults.
2. Interactive prompts for artist folders and language-based organization.
3. Post-processing retry for missing covers and lyrics.
4. Flexible file naming: "Artist - Title [Album]" or "Artist - Title" if no album.
5. Language-based folder organization (Persian, English, Arabic, etc.).
6. Interactive lyrics/cover fetching options with retry mechanism.
7. Automatic dependency installation with minimal terminal output.
8. Network connectivity monitoring with real-time ping display.
9. Advanced duplicate detection using SHA-1 hashing with caching.
10. Full metadata support for .mp3, .flac, .m4a/.mp4, .ogg, .wav, .wma.
11. Multiple online metadata sources (Genius, AZLyrics, iTunes, Last.fm, Spotify).
12. Single clean progress display that replaces itself for each file.
13. Comprehensive Windows-compatible metadata including subtitle (lyrics) and tags.
14. Proper cover art and lyrics embedding with verification.
"""

# -----------------------------------------------------------------------------
# SECTION 0: Preliminary Setup & Dependency Management
# -----------------------------------------------------------------------------
import os
import sys
import subprocess
import importlib
from pathlib import Path

# Application directories
APP_DIR = Path.home() / '.music_organizer'
APP_DIR.mkdir(exist_ok=True)

CONFIG_FILE = APP_DIR / 'config.json'
PROGRESS_FILE = APP_DIR / 'progress.json'
STATS_FILE = APP_DIR / 'stats.json'

def check_and_install_libraries():
    """Check for required libraries and install them if missing"""
    required_libraries = {
        'click': 'click',
        'rich': 'rich', 
        'mutagen': 'mutagen',
        'requests': 'requests',
        'send2trash': 'send2trash',
        'beautifulsoup4': 'beautifulsoup4',
        'langdetect': 'langdetect',
        'lxml': 'lxml'
    }
    
    missing_libraries = []
    
    # Check which libraries are missing
    for lib_name, pip_name in required_libraries.items():
        try:
            if lib_name == 'beautifulsoup4':
                importlib.import_module('bs4')
            else:
                importlib.import_module(lib_name)
        except ImportError:
            missing_libraries.append(pip_name)
    
    if missing_libraries:
        print("Installing required libraries...")
        for lib in missing_libraries:
            print(f"Installing {lib}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", lib
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                print(f"Failed to install {lib}. Please install manually: pip install {lib}")
                sys.exit(1)
        print("✅ All libraries installed successfully!")
    else:
        print("✅ All libraries are already installed.")

# Install dependencies before importing them
check_and_install_libraries()

# -----------------------------------------------------------------------------
# SECTION 1: Core Imports
# -----------------------------------------------------------------------------
import asyncio
import hashlib
import json
import logging
import platform
import shutil
import sqlite3
import threading
import time
import csv
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Set, Any
from urllib.parse import quote_plus, urljoin

import click
import mutagen
from mutagen.id3 import ID3, USLT, APIC, TIT2, TPE1, TALB, TDRC, TCON, TIT3, TPE2, COMM
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.live import Live
from send2trash import send2trash
import asyncio
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# -----------------------------------------------------------------------------
# SECTION 2: Configuration & Data Classes
# -----------------------------------------------------------------------------

console = Console()

@dataclass
class Config:
    """Configuration settings for the music organizer"""
    root_folder: str = ""
    output_folder: str = ""
    mode: str = "copy"  # copy or move
    organize_by_artist: bool = False
    organize_by_language: bool = False
    fetch_lyrics: bool = True
    fetch_covers: bool = True
    max_workers: int = 4
    ping_interval: int = 10
    request_timeout: int = 15
    max_retries: int = 3
    
    @classmethod
    def load(cls) -> 'Config':
        """Load configuration from file"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                    return cls(**data)
            except Exception as e:
                console.print(f"[yellow]Failed to load config: {e}[/yellow]")
        return cls()
    
    def save(self):
        """Save configuration to file"""
        with open(CONFIG_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)

@dataclass
class FileStats:
    """Statistics for processed files"""
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    duplicates: int = 0
    lyrics_found: int = 0
    covers_found: int = 0
    total_size: int = 0
    missing_lyrics: List[str] = None
    missing_covers: List[str] = None
    
    def __post_init__(self):
        if self.missing_lyrics is None:
            self.missing_lyrics = []
        if self.missing_covers is None:
            self.missing_covers = []
    
    def save(self, reports_folder: Path):
        """Save stats to file in reports folder"""
        stats_file = reports_folder / 'stats.json'
        stats_dict = asdict(self)
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)

# -----------------------------------------------------------------------------
# SECTION 3: Network Monitoring & Connectivity
# -----------------------------------------------------------------------------

class NetworkMonitor:
    """Monitor network connectivity and latency with real-time updates"""
    
    def __init__(self, interval: int = 10):
        self.interval = interval
        self.ping_ms = 0
        self.is_connected = True
        self._stop_event = threading.Event()
        self._thread = None
        
    def start_monitoring(self):
        """Start network monitoring in background thread"""
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
    def stop_monitoring(self):
        """Stop network monitoring"""
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=2)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            self._check_network()
            self._stop_event.wait(self.interval)
    
    def _check_network(self):
        """Check network connectivity and measure ping to Google DNS"""
        try:
            host = "8.8.8.8"
            if platform.system().lower() == "windows":
                cmd = ["ping", "-n", "1", host]
            else:
                cmd = ["ping", "-c", "1", host]
            
            start_time = time.time()
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                self.ping_ms = int((time.time() - start_time) * 1000)
                self.is_connected = True
            else:
                self.is_connected = False
                self.ping_ms = 0
                
        except Exception:
            self.is_connected = False
            self.ping_ms = 0
    
    def get_status(self) -> str:
        """Get formatted network status string"""
        if self.is_connected:
            return f"🌐 {self.ping_ms}ms"
        return "🔴 Offline"

# -----------------------------------------------------------------------------
# SECTION 4: Duplicate Detection & Management
# -----------------------------------------------------------------------------

class DuplicateManager:
    """Advanced duplicate file detection with SHA-1 hashing and caching"""
    
    def __init__(self):
        self.db_path = APP_DIR / "duplicates.db"
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for hash caching"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                size INTEGER NOT NULL,
                modified INTEGER NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def get_file_hash(self, file_path: Path, use_cache: bool = True) -> Optional[str]:
        """Get SHA-1 hash of file with intelligent caching"""
        if use_cache:
            cached_hash = self._get_cached_hash(file_path)
            if cached_hash:
                return cached_hash
        
        try:
            hash_obj = hashlib.sha1()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            
            file_hash = hash_obj.hexdigest()
            self._cache_hash(file_path, file_hash)
            return file_hash
            
        except Exception as e:
            console.print(f"[red]Hash calculation failed for {file_path}: {e}[/red]")
            return None
    
    def _get_cached_hash(self, file_path: Path) -> Optional[str]:
        """Retrieve cached hash if file hasn't changed"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stat = file_path.stat()
            cursor.execute(
                "SELECT hash FROM file_hashes WHERE path = ? AND size = ? AND modified = ?",
                (str(file_path), stat.st_size, int(stat.st_mtime))
            )
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception:
            return None
    
    def _cache_hash(self, file_path: Path, file_hash: str):
        """Store file hash in cache database"""
        try:
            conn = sqlite3.connect(self.db_path)
            stat = file_path.stat()
            
            conn.execute(
                "INSERT OR REPLACE INTO file_hashes (path, hash, size, modified) VALUES (?, ?, ?, ?)",
                (str(file_path), file_hash, stat.st_size, int(stat.st_mtime))
            )
            conn.commit()
            conn.close()
            
        except Exception as e:
            console.print(f"[yellow]Failed to cache hash: {e}[/yellow]")
    
    def find_duplicates(self, files: List[Path], max_workers: int = 4) -> Dict[str, List[Path]]:
        """Find duplicate files using parallel processing"""
        duplicates = defaultdict(list)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.get_file_hash, file_path): file_path 
                for file_path in files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_hash = future.result()
                    if file_hash:
                        duplicates[file_hash].append(file_path)
                except Exception as e:
                    console.print(f"[red]Error processing {file_path}: {e}[/red]")
        
        return {k: v for k, v in duplicates.items() if len(v) > 1}

# -----------------------------------------------------------------------------
# SECTION 5: Enhanced Metadata Services (Multiple Sources)
# -----------------------------------------------------------------------------

class MetadataEnhancer:
    """Enhanced metadata services with multiple sources for maximum coverage"""
    
    def __init__(self, timeout: int = 15, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def fetch_lyrics(self, artist: str, title: str) -> Optional[str]:
        """Fetch lyrics from multiple sources with fallback"""
        # Try multiple sources in order of preference
        sources = [
            self._fetch_lyrics_azlyrics,
            self._fetch_lyrics_genius,
            self._fetch_lyrics_lyricscom,
            self._fetch_lyrics_musixmatch,
            self._fetch_lyrics_chartlyrics,
            self._fetch_lyrics_vagalume,
            self._fetch_lyrics_lyricwiki,
            self._fetch_lyrics_ovh,         
            self._fetch_lyrics_lyricsfa,
            self._fetch_lyrics_songlyrics,     
            self._fetch_lyrics_lyricfreak,
        ]
        for source_func in sources:
            try:
                lyrics = source_func(artist, title)
                if lyrics and len(lyrics.strip()) > 50:  # Ensure meaningful content
                    return lyrics
            except Exception:
                continue
        
        return None
    
    def _fetch_lyrics_songlyrics(self, artist: str, title: str) -> Optional[str]:
        """Scrape songlyrics.com"""
        q = quote_plus(f"{artist} {title}")
        search_url = f"https://www.songlyrics.com/index.php?section=search&searchW={q}&submit=Search"
        r = self.session.get(search_url, timeout=self.timeout)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        # first result link under table.search-result
        link = soup.select_one("td.tal.qx a")
        if not link or not link["href"]:
            return None
        page = self.session.get(link["href"], timeout=self.timeout)
        if page.status_code != 200:
            return None
        ps = BeautifulSoup(page.text, "html.parser")
        lyrics_div = ps.find("p", id="songLyricsDiv")
        return lyrics_div.get_text("").strip() if lyrics_div else None

    def _fetch_lyrics_lyricfreak(self, artist: str, title: str) -> Optional[str]:
        """Scrape lyricfreak.com"""
        q = quote_plus(f"{artist} {title}")
        search_url = f"https://www.lyricfreak.com/search.php?a=search&type=song&q={q}"
        r = self.session.get(search_url, timeout=self.timeout)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        link = soup.select_one("a.song")
        if not link or not link["href"]:
            return None
        page = self.session.get(link["href"], timeout=self.timeout)
        if page.status_code != 200:
            return None
        ps = BeautifulSoup(page.text, "html.parser")
        ly = ps.find("div", class_="lyrics")
        return ly.get_text("").strip() if ly else None
    
    def _fetch_lyrics_ovh(self, artist: str, title: str) -> Optional[str]:
        """Free lyrics API: https://api.lyrics.ovh"""
        url = f"https://api.lyrics.ovh/v1/{quote_plus(artist)}/{quote_plus(title)}"
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("lyrics")

    def _fetch_lyrics_lyricsfa(self, artist: str, title: str) -> Optional[str]:
        """
        Scrape lyricsfa.ir (a popular Persian lyrics site),
        by searching artist+title and picking the first result.
        """
        query = quote_plus(f"{artist} {title}")
        search_url = f"https://lyricsfa.ir/search/?q={query}"
        resp = self.session.get(search_url, timeout=self.timeout)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        # find first result link (you may need to tweak selectors if site layout changes)
        link = soup.select_one("h3.entry-title a")
        if not link or not link["href"]:
            return None

        page = self.session.get(link["href"], timeout=self.timeout)
        if page.status_code != 200:
            return None
        page_soup = BeautifulSoup(page.text, "html.parser")
        box = page_soup.select_one("div.lyricsBox") or page_soup.select_one("div.entry-content")
        return box.get_text(separator="\n").strip() if box else None
    
    def _fetch_lyrics_azlyrics(self, artist: str, title: str) -> Optional[str]:
        """Fetch lyrics from AZLyrics (web scraping)"""
        try:
            # Clean artist and title for URL
            artist_clean = re.sub(r'[^a-zA-Z0-9]', '', artist.lower())
            title_clean = re.sub(r'[^a-zA-Z0-9]', '', title.lower())
            
            url = f"https://www.azlyrics.com/lyrics/{artist_clean}/{title_clean}.html"
            
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find lyrics div (AZLyrics specific structure)
            lyrics_div = soup.find('div', class_='')
            if lyrics_div and 'Sorry about that' not in lyrics_div.get_text():
                lyrics = lyrics_div.get_text().strip()
                return lyrics if lyrics else None
            
        except Exception:
            pass
        
        return None
    
    def _fetch_lyrics_genius(self, artist: str, title: str) -> Optional[str]:
        """Fetch lyrics from Genius (web scraping - no API key needed)"""
        try:
            # Search for the song
            search_url = "https://genius.com/api/search/multi"
            params = {"per_page": "5", "q": f"{artist} {title}"}
            
            response = self.session.get(search_url, params=params, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            data = response.json()
            songs = data.get('response', {}).get('sections', [])
            
            for section in songs:
                if section.get('type') == 'song':
                    hits = section.get('hits', [])
                    if hits:
                        song_url = hits[0].get('result', {}).get('url')
                        if song_url:
                            return self._scrape_genius_lyrics(song_url)
            
        except Exception:
            pass
        
        return None
    
    def _scrape_genius_lyrics(self, song_url: str) -> Optional[str]:
        """Scrape lyrics from Genius song page"""
        try:
            response = self.session.get(song_url, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Genius lyrics container
            lyrics_container = soup.find('div', {'data-lyrics-container': 'true'})
            if lyrics_container:
                return lyrics_container.get_text(separator='\n').strip()
            
            # Alternative selector
            lyrics_div = soup.find('div', class_=re.compile('lyrics|Lyrics'))
            if lyrics_div:
                return lyrics_div.get_text(separator='\n').strip()
            
        except Exception:
            pass
        
        return None
    
    def _fetch_lyrics_lyricscom(self, artist: str, title: str) -> Optional[str]:
        """Fetch lyrics from Lyrics.com"""
        try:
            # Search URL
            search_url = "https://www.lyrics.com/serp.php"
            params = {"st": f"{artist} {title}", "qtype": "2"}
            
            response = self.session.get(search_url, params=params, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find first result link
            result_link = soup.find('a', href=re.compile(r'/lyric/'))
            if result_link:
                lyrics_url = urljoin("https://www.lyrics.com", result_link['href'])
                return self._scrape_lyricscom_lyrics(lyrics_url)
            
        except Exception:
            pass
        
        return None
    
    def _scrape_lyricscom_lyrics(self, lyrics_url: str) -> Optional[str]:
        """Scrape lyrics from Lyrics.com song page"""
        try:
            response = self.session.get(lyrics_url, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Lyrics.com specific container
            lyrics_div = soup.find('pre', id='lyric-body-text')
            if lyrics_div:
                return lyrics_div.get_text().strip()
            
        except Exception:
            pass
        
        return None
    
    def _fetch_lyrics_chartlyrics(self, artist: str, title: str) -> Optional[str]:
        """ChartLyrics SOAP API"""
        url = "http://api.chartlyrics.com/apiv1.asmx/SearchLyricDirect"
        params = {"artist": artist, "song": title}
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code != 200:
            return None
        # XML parsing
        import xml.etree.ElementTree as ET
        tree = ET.fromstring(resp.text)
        lyric = tree.findtext('.//Lyric')
        return lyric.strip() if lyric else None

    def _fetch_lyrics_vagalume(self, artist: str, title: str) -> Optional[str]:
        """Vagalume free API"""
        url = "https://api.vagalume.com.br/search.php"
        params = {"art": artist, "mus": title}
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code != 200:
            return None
        js = resp.json()
        # try to grab first lyrics
        matches = js.get("mus", [])
        if matches:
            text = matches[0].get("text", "")
            return text.replace("","").strip()
        return None

    def _fetch_lyrics_lyricwiki(self, artist: str, title: str) -> Optional[str]:
        """LyricWiki via Fandom’s API"""
        # slugify artist/title
        slug = f"{artist.strip().replace(' ', '_')}_{title.strip().replace(' ', '_')}"
        url = f"https://lyrics.fandom.com/api.php"
        params = {
            "action": "parse",
            "page": slug,
            "format": "json",
            "prop": "wikitext",
        }
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
        # strip out wiki markup as best we can
        import re
        lyrics = re.sub(r"\{\{.*?\}\}", "", wikitext, flags=re.S)
        lyrics = re.sub(r"\[https?://[^\]]+\]", "", lyrics)
        return lyrics.strip() if len(lyrics) > 50 else None
    
    def _fetch_lyrics_musixmatch(self, artist: str, title: str) -> Optional[str]:
        """Fetch lyrics via the Musixmatch API."""
        api_key = os.getenv("MUSIXMATCH_KEY")
        if not api_key:
            return None
        url = "https://api.musixmatch.com/ws/1.1/matcher.lyrics.get"
        params = {
            "q_artist": artist,
            "q_track": title,
            "apikey": api_key,
        }
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code != 200:
            return None
        data = resp.json().get("message", {}).get("body", {}).get("lyrics", {})
        text = data.get("lyrics_body", "")
        return text.split("*******")[0].strip() if text else None
    
    
    def fetch_cover_art(self, artist: str, album: str) -> Optional[bytes]:
        """Fetch cover art from multiple sources with fallback"""
        
        # Try multiple sources in order of preference
        sources = [
            self._fetch_cover_local,               
            self._fetch_cover_itunes,
            self._fetch_cover_lastfm,
            self._fetch_cover_coverartarchive,
            self._fetch_cover_deezer,
            self._fetch_cover_spotify,
            self._fetch_cover_discogs,             
            self._fetch_cover_google_images,
            self._fetch_cover_bing,         
            self._fetch_cover_soundcloud,
            self._fetch_cover_saavn,          
            self._fetch_cover_zingmp3,         
            self._fetch_cover_yandex,          
        ]
        for fn in sources:
            try:
                img = fn(artist, album)
                if img and len(img) > 1_000:
                    return img
            except Exception:
                continue
        return None
    
    def _fetch_cover_saavn(self, artist: str, album: str) -> Optional[bytes]:
        """Use the public Saavn autocomplete API."""
        q = f"{artist} {album}"
        params = {
            "__call": "autocomplete.get",
            "query": q,
            "_format": "json",
            "_marker": "0"
        }
        r = self.session.get("https://www.saavn.com/api.php", params=params, timeout=self.timeout)
        if r.status_code != 200:
            return None
        albums = r.json().get("albums", [])
        if not albums:
            return None
        art = albums[0].get("more_info", {}).get("album_art")
        if not art:
            return None
        img = self.session.get(art, timeout=self.timeout)
        if img.status_code == 200:
            return img.content

    def _fetch_cover_zingmp3(self, artist: str, album: str) -> Optional[bytes]:
        """Unofficial ZingMP3 search API (Vietnamese service)."""
        q = quote_plus(f"{artist} {album}")
        url = f"https://zingmp3.vn/search/album?query={q}"
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code != 200:
            return None
        data = r.json().get("data", {}).get("albums", [])
        if not data:
            return None
        cover = data[0].get("cover")
        if not cover:
            return None
        img = self.session.get(cover, timeout=self.timeout)
        if img.status_code == 200:
            return img.content

    def _fetch_cover_yandex(self, artist: str, album: str) -> Optional[bytes]:
        """Scrape Yandex Images first thumbnail."""
        q = quote_plus(f"{artist} {album} album cover art")
        url = f"https://yandex.com/images/search?text={q}"
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        img_tag = soup.select_one("img.serp-item__thumb")
        src = img_tag.get("src") or img_tag.get("data-src")
        if not src:
            return None
        # Yandex often returns a small base64 or lazy URL, skip if so
        if src.startswith("data:"):
            return None
        img = self.session.get(src, timeout=self.timeout)
        if img.status_code == 200:
            return img.content
    
    def _fetch_cover_spotify(self, artist: str, album: str) -> Optional[bytes]:
        """Fetch cover via the Spotify Web API."""
        client_id     = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        if not client_id or not client_secret:
            return None
        # get token
        token_resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=self.timeout
        )
        if token_resp.status_code != 200:
            return None
        token = token_resp.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}

        # search for album
        q = f"album:{album} artist:{artist}"
        resp = self.session.get(
            "https://api.spotify.com/v1/search",
            headers=headers,
            params={"q": q, "type": "album", "limit": 1},
            timeout=self.timeout
        )
        if resp.status_code != 200:
            return None
        items = resp.json().get("albums", {}).get("items", [])
        if not items:
            return None

        # pull the largest image
        img_url = items[0].get("images", [])[-1].get("url")
        if not img_url:
            return None
        img_resp = self.session.get(img_url, timeout=self.timeout)
        return img_resp.content if img_resp.status_code == 200 else None
    
    def _fetch_cover_local(self, artist: str, album: str) -> Optional[bytes]:
        """Look for common artwork files next to your music."""
        import glob
        folder = Path(self.output_dir)  # or wherever you're processing
        patterns = ["cover.jpg", "folder.jpg", "front.*", "*.png"]
        for p in patterns:
            matches = list(folder.rglob(p))
            if matches:
                return matches[0].read_bytes()
        return None

    def _fetch_cover_discogs(self, artist: str, album: str) -> Optional[bytes]:
        """Fetch cover via Discogs API (requires DISCOGS_TOKEN env var)."""
        token = os.getenv("DISCOGS_TOKEN")
        if not token:
            return None
        headers = {"Authorization": f"Discogs token={token}"}
        q = f"{artist} {album}"
        resp = self.session.get(
            "https://api.discogs.com/database/search",
            headers=headers,
            params={"q": q, "type": "release", "per_page": 1},
            timeout=self.timeout
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        if not results:
            return None
        thumb = results[0].get("thumb")
        # try primary URLs first
        img_url = results[0].get("cover_image", thumb)
        if not img_url:
            return None
        img_resp = self.session.get(img_url, timeout=self.timeout)
        return img_resp.content if img_resp.status_code == 200 else None

    def _fetch_cover_google_images(self, artist: str, album: str) -> Optional[bytes]:
        """Scrape Google Images via SerpAPI (requires SERPAPI_KEY)."""
        key = os.getenv("SERPAPI_KEY")
        if not key:
            return None
        from urllib.parse import quote_plus
        query = quote_plus(f"{artist} {album} album cover art")
        url = f"https://serpapi.com/search.json?engine=google_images&q={query}&api_key={key}&ijn=0"
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            return None
        data = resp.json().get("images_results", [])
        if not data:
            return None
        img_url = data[0].get("original")
        if not img_url:
            return None
        img_resp = self.session.get(img_url, timeout=self.timeout)
        return img_resp.content if img_resp.status_code == 200 else None
    
    def _fetch_cover_itunes(self, artist: str, album: str) -> Optional[bytes]:
        """Fetch cover art from iTunes API"""
        try:
            search_term = f"{artist} {album}".replace(" ", "+")
            api_url = f"https://itunes.apple.com/search?term={search_term}&entity=album&limit=5"
            
            response = self.session.get(api_url, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            data = response.json()
            results = data.get("results", [])
            
            for result in results:
                # Try highest resolution first
                artwork_url = result.get("artworkUrl100", "").replace("100x100bb", "600x600bb")
                if artwork_url:
                    img_response = self.session.get(artwork_url, timeout=self.timeout)
                    if img_response.status_code == 200 and len(img_response.content) > 1000:
                        return img_response.content
            
        except Exception:
            pass
        
        return None
    
    def _fetch_cover_bing(self, artist: str, album: str) -> Optional[bytes]:
        """Scrape Bing Images first‐result for “artist album cover”"""
        query = quote_plus(f"{artist} {album} album cover art")
        url = f"https://www.bing.com/images/search?q={query}&form=HDRSC2"
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        # Bing packs JSON in a “m” attribute on <a class="iusc">
        a = soup.select_one("a.iusc")
        if not a or "m" not in a.attrs:
            return None
        data = json.loads(a["m"])
        img_url = data.get("murl")
        if not img_url:
            return None
        img_resp = self.session.get(img_url, timeout=self.timeout)
        return img_resp.content if img_resp.status_code == 200 else None

    def _fetch_cover_soundcloud(self, artist: str, album: str) -> Optional[bytes]:
        """Use SoundCloud search API to grab track/artwork as fallback"""
        # Note: No API key needed for public search but rate‐limited
        params = {"q": f"{artist} {album}", "limit": 1}
        resp = self.session.get(
            "https://api-v2.soundcloud.com/search/tracks",
            params={**params, "client_id": "2t9loNQH90kzJcsFCODdigxfp325aq4z"},  # public client_id
            timeout=self.timeout
        )
        if resp.status_code != 200:
            return None
        hits = resp.json().get("collection", [])
        if not hits:
            return None
        # artwork_url e.g. "...-t500x500.jpg"
        art = hits[0].get("artwork_url")
        if not art:
            return None
        # pick a larger variant
        highres = art.replace("-large", "-t500x500")
        img = self.session.get(highres, timeout=self.timeout)
        return img.content if img.status_code == 200 else None
    
    def _fetch_cover_lastfm(self, artist: str, album: str) -> Optional[bytes]:
        """Fetch cover art from Last.fm API (no key required for album info)"""
        try:
            # Last.fm album info API
            api_url = "http://ws.audioscrobbler.com/2.0/"
            params = {
                "method": "album.getinfo",
                "artist": artist,
                "album": album,
                "format": "json"
            }
            
            response = self.session.get(api_url, params=params, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            data = response.json()
            album_info = data.get("album", {})
            images = album_info.get("image", [])
            
            # Find largest image
            for img in reversed(images):  # Start from largest
                img_url = img.get("#text")
                if img_url:
                    img_response = self.session.get(img_url, timeout=self.timeout)
                    if img_response.status_code == 200 and len(img_response.content) > 1000:
                        return img_response.content
            
        except Exception:
            pass
        
        return None
    
    def _fetch_cover_coverartarchive(self, artist: str, album: str) -> Optional[bytes]:
        """Fetch cover art from Cover Art Archive via MusicBrainz"""
        try:
            # Search MusicBrainz for album
            search_url = "https://musicbrainz.org/ws/2/release"
            params = {
                "query": f'artist:"{artist}" AND release:"{album}"',
                "fmt": "json",
                "limit": 5
            }
            
            response = self.session.get(search_url, params=params, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            data = response.json()
            releases = data.get("releases", [])
            
            for release in releases:
                release_id = release.get("id")
                if release_id:
                    # Try to get cover art
                    cover_url = f"https://coverartarchive.org/release/{release_id}/front"
                    img_response = self.session.get(cover_url, timeout=self.timeout)
                    if img_response.status_code == 200 and len(img_response.content) > 1000:
                        return img_response.content
            
        except Exception:
            pass
        
        return None
    
    def _fetch_cover_deezer(self, artist: str, album: str) -> Optional[bytes]:
        """Fetch cover art from Deezer API"""
        try:
            # Search Deezer for album
            search_url = "https://api.deezer.com/search/album"
            params = {"q": f"{artist} {album}"}
            
            response = self.session.get(search_url, params=params, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            data = response.json()
            albums = data.get("data", [])
            
            for album_data in albums:
                cover_url = album_data.get("cover_xl")  # Highest resolution
                if cover_url:
                    img_response = self.session.get(cover_url, timeout=self.timeout)
                    if img_response.status_code == 200 and len(img_response.content) > 1000:
                        return img_response.content
            
        except Exception:
            pass
        
        return None
    
    def embed_comprehensive_metadata(self, file_path: Path, metadata: Dict[str, Any], lyrics: Optional[str] = None, cover_data: Optional[bytes] = None):
        """Embed comprehensive Windows-compatible metadata into audio file"""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.mp3':
                self._embed_mp3_metadata(file_path, metadata, lyrics, cover_data)
            elif file_ext == '.flac':
                self._embed_flac_metadata(file_path, metadata, lyrics, cover_data)
            elif file_ext in ['.m4a', '.mp4']:
                self._embed_mp4_metadata(file_path, metadata, lyrics, cover_data)
                
        except Exception as e:
            console.print(f"\n[dim red]Failed to embed metadata in {file_path}: {e}[/dim red]")
    
    def _embed_mp3_metadata(self, file_path: Path, metadata: Dict[str, Any], lyrics: Optional[str], cover_data: Optional[bytes]):
        """Embed comprehensive metadata into MP3 file"""
        try:
            try:
                audio = ID3(file_path)
            except:
                audio = ID3()
            
            # Clear existing tags to ensure clean metadata
            audio.clear()
            
            # Basic metadata
            if metadata.get('title'):
                audio.add(TIT2(encoding=3, text=metadata['title']))
            
            if metadata.get('artist'):
                audio.add(TPE1(encoding=3, text=metadata['artist']))
                # Album artist (same as artist if not specified)
                audio.add(TPE2(encoding=3, text=metadata.get('album_artist', metadata['artist'])))
            
            if metadata.get('album'):
                audio.add(TALB(encoding=3, text=metadata['album']))
            
            if metadata.get('year'):
                audio.add(TDRC(encoding=3, text=str(metadata['year'])))
            
            if metadata.get('genre'):
                audio.add(TCON(encoding=3, text=metadata['genre']))
            
            # Subtitle field for lyrics (TIT3 frame)
            if lyrics:
                audio.add(TIT3(encoding=3, text=lyrics))
                # Also add as unsynchronized lyrics
                audio.add(USLT(encoding=3, lang='eng', desc='', text=lyrics))
            
            # Language tag in comments
            if metadata.get('language'):
                audio.add(COMM(encoding=3, lang='eng', desc='Language', text=metadata['language']))
            
            # Contributing artists
            if metadata.get('contributing_artists'):
                audio.add(COMM(encoding=3, lang='eng', desc='Contributing Artists', text=metadata['contributing_artists']))
            
            # Cover art
            if cover_data:
                audio.add(APIC(
                    encoding=3,
                    mime='image/jpeg',
                    type=3,  # Cover (front)
                    desc='Cover',
                    data=cover_data
                ))
            
            audio.save(file_path, v2_version=4)
            
        except Exception as e:
            console.print(f"\n[dim red]Failed to embed MP3 metadata: {e}[/dim red]")
    
    def _embed_flac_metadata(self, file_path: Path, metadata: Dict[str, Any], lyrics: Optional[str], cover_data: Optional[bytes]):
        """Embed comprehensive metadata into FLAC file"""
        try:
            audio = FLAC(file_path)
            
            # Clear existing tags
            audio.clear()
            
            # Basic metadata
            if metadata.get('title'):
                audio['TITLE'] = metadata['title']
            
            if metadata.get('artist'):
                audio['ARTIST'] = metadata['artist']
                audio['ALBUMARTIST'] = metadata.get('album_artist', metadata['artist'])
            
            if metadata.get('album'):
                audio['ALBUM'] = metadata['album']
            
            if metadata.get('year'):
                audio['DATE'] = str(metadata['year'])
            
            if metadata.get('genre'):
                audio['GENRE'] = metadata['genre']
            
            # Lyrics as subtitle and dedicated lyrics field
            if lyrics:
                audio['LYRICS'] = lyrics
                audio['SUBTITLE'] = lyrics  # For Windows compatibility
            
            # Language tag
            if metadata.get('language'):
                audio['LANGUAGE'] = metadata['language']
            
            # Contributing artists
            if metadata.get('contributing_artists'):
                audio['PERFORMER'] = metadata['contributing_artists']
            
            # Cover art
            if cover_data:
                from mutagen.flac import Picture
                picture = Picture()
                picture.type = 3  # Cover (front)
                picture.mime = 'image/jpeg'
                picture.desc = 'Cover'
                picture.data = cover_data
                audio.add_picture(picture)
            
            audio.save()
            
        except Exception as e:
            console.print(f"\n[dim red]Failed to embed FLAC metadata: {e}[/dim red]")
    
    def _embed_mp4_metadata(self, file_path: Path, metadata: Dict[str, Any], lyrics: Optional[str], cover_data: Optional[bytes]):
        """Embed comprehensive metadata into MP4/M4A file"""
        try:
            audio = MP4(file_path)
            
            # Clear existing tags
            audio.clear()
            
            # Basic metadata
            if metadata.get('title'):
                audio['©nam'] = [metadata['title']]
            
            if metadata.get('artist'):
                audio['©ART'] = [metadata['artist']]
                audio['aART'] = [metadata.get('album_artist', metadata['artist'])]
            
            if metadata.get('album'):
                audio['©alb'] = [metadata['album']]
            
            if metadata.get('year'):
                audio['©day'] = [str(metadata['year'])]
            
            if metadata.get('genre'):
                audio['©gen'] = [metadata['genre']]
            
            # Lyrics
            if lyrics:
                audio['©lyr'] = [lyrics]
            
            # Language (custom tag)
            if metadata.get('language'):
                audio['©lng'] = [metadata['language']]
            
            # Contributing artists
            if metadata.get('contributing_artists'):
                audio['©wrt'] = [metadata['contributing_artists']]
            
            # Cover art
            if cover_data:
                audio['covr'] = [cover_data]
            
            audio.save()
            
        except Exception as e:
            console.print(f"\n[dim red]Failed to embed MP4 metadata: {e}[/dim red]")

# -----------------------------------------------------------------------------
# SECTION 6: Enhanced Progress Management with Live Display
# -----------------------------------------------------------------------------

class ProgressManager:
    """Enhanced progress tracking with single live-updating progress bar"""
    
    def __init__(self, network_monitor: NetworkMonitor):
        self.network_monitor = network_monitor
        self.stats = FileStats()
        self.start_time = time.time()
        self.processed_count = 0
        self.total_files = 0
        self.current_logs = []
        
    def create_progress_display(self, total_files: int):
        """Create progress display that shows logs above and progress bar below"""
        self.total_files = total_files
        
        # Create progress bar components
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", justify="left"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[network_status]}"),
            TextColumn("•"),
            TextColumn("[red]✗{task.fields[errors]}"),
            TextColumn("[yellow]⚡{task.fields[duplicates]}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        
        return progress
    
    def add_log_message(self, message: str):
        """Add a log message to be displayed above the progress bar"""
        console.print(f"\n{message}")
    
    def update_progress(self, task_id, progress_obj, current_file: str):
        """Update progress with better time estimation"""
        self.processed_count += 1
        
        # Calculate better ETA
        elapsed = time.time() - self.start_time
        if self.processed_count > 0:
            avg_time_per_file = elapsed / self.processed_count
            remaining_files = self.total_files - self.processed_count
        else:
            remaining_files = self.total_files
        
        progress_obj.update(
            task_id,
            description=current_file,
            network_status=self.network_monitor.get_status(),
            errors=self.stats.errors,
            duplicates=self.stats.duplicates,
            completed=self.processed_count
        )

# -----------------------------------------------------------------------------
# SECTION 7: Main Music Organizer Class
# -----------------------------------------------------------------------------

class MusicOrganizer:
    """Main music organization engine with comprehensive features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.output_folder)
        self.reports_dir = self.output_dir / 'reports'
        self.network_monitor = NetworkMonitor(config.ping_interval)
        self.duplicate_manager = DuplicateManager()
        self.metadata_enhancer = MetadataEnhancer(config.request_timeout, config.max_retries)
        self.progress_manager = ProgressManager(self.network_monitor)
        self.stats = FileStats()
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to reports folder
        self.log_file = self.reports_dir / 'organizer.log'
        self.report_csv = self.reports_dir / 'report.csv'
        self.setup_logging()
        
        # Setup CSV reporting
        self.setup_reporting()
    
    def setup_logging(self):
        """Setup logging to reports folder"""
        # Remove existing handlers
        logger = logging.getLogger("music_organizer")
        logger.handlers.clear()
        
        # Add file handler only (we'll handle console output manually)
        logger.addHandler(logging.FileHandler(self.log_file, encoding="utf-8"))
        logger.setLevel(logging.INFO)
    
    def setup_reporting(self):
        """Initialize CSV reporting for processed files in reports folder"""
        self.csv_file = open(self.report_csv, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Original Path', 'New Path', 'Artist', 'Title', 'Album', 'Genre', 'Year',
            'Action', 'Timestamp', 'File Size (MB)', 'Lyrics Found', 'Cover Found', 
            'Language', 'Contributing Artists'
        ])
    
    def start(self):
        """Start the complete organization process"""
        logger = logging.getLogger("music_organizer")
        try:
            self.network_monitor.start_monitoring()
            self._run_organization()
            self._post_processing()
        finally:
            self.network_monitor.stop_monitoring()
            self.stats.save(self.reports_dir)
            if hasattr(self, 'csv_file'):
                self.csv_file.close()
            logger.info("Organization process completed")
    
    def _run_organization(self):
        """Main organization workflow with all features"""
        logger = logging.getLogger("music_organizer")
        
        # Display configuration panel
        organization_mode = "Artist folders" if self.config.organize_by_artist else "Flat (all in output folder)"
        language_mode = "Language-based folders" if self.config.organize_by_language else "Mixed languages"
        
        console.print(Panel.fit(
            "[bold green]🎵 Enhanced Music Organizer v2.4[/bold green]\n"
            f"[cyan]Root:[/cyan] {self.config.root_folder}\n"
            f"[cyan]Output:[/cyan] {self.config.output_folder}\n"
            f"[cyan]Reports:[/cyan] {self.reports_dir}\n"
            f"[cyan]Mode:[/cyan] {self.config.mode.upper()}\n"
            f"[cyan]Organization:[/cyan] {organization_mode}\n"
            f"[cyan]Language Sort:[/cyan] {language_mode}\n"
            f"[cyan]Fetch Lyrics:[/cyan] {'Yes' if self.config.fetch_lyrics else 'No'}\n"
            f"[cyan]Fetch Covers:[/cyan] {'Yes' if self.config.fetch_covers else 'No'}",
            title="Configuration"
        ))
        
        logger.info(f"Starting organization: {self.config.root_folder} -> {self.config.output_folder}")
        
        # Step 1: Scan for music files
        console.print("\n[yellow]📁 Scanning for music files...[/yellow]")
        music_files = self._get_music_files(Path(self.config.root_folder))
        
        if not music_files:
            console.print("[red]❌ No music files found![/red]")
            logger.warning("No music files found in root folder")
            return
        
        console.print(f"[green]✅ Found {len(music_files)} music files[/green]")
        logger.info(f"Found {len(music_files)} music files")
        
        # Step 2: Handle duplicates
        if len(music_files) > 1:
            self._handle_duplicates(music_files)
        
        # Step 3: Process all files
        self._process_files(music_files)
        
        # Step 4: Show initial statistics
        self._show_initial_stats()
    
    def _post_processing(self):
        """Post‐processing for missing covers and lyrics (ask both first)."""
        logger = logging.getLogger("music_organizer")

        # 1) Ask both questions up front
        retry_covers = False
        retry_lyrics = False

        if self.stats.missing_covers and self.network_monitor.is_connected:
            retry_covers = Confirm.ask(
                f"\n🖼️  {len(self.stats.missing_covers)} songs are missing covers. "
                "Search again for covers?",
                default=False
            )

        if self.stats.missing_lyrics and self.network_monitor.is_connected:
            retry_lyrics = Confirm.ask(
                f"📝 {len(self.stats.missing_lyrics)} songs are missing lyrics. "
                "Search again for lyrics?",
                default=False
            )

        # 2) Now trigger retries based on answers
        if retry_covers:
            console.print("\n[yellow]🖼️  Searching for missing covers...[/yellow]")
            self._retry_missing_covers()

        if retry_lyrics:
            console.print("\n[yellow]📝 Searching for missing lyrics...[/yellow]")
            self._retry_missing_lyrics()

        # 3) Final summary
        self._show_final_stats()

    
    def _retry_missing_covers(self):
        """Retry fetching covers for songs that didn't have them"""
        found_covers = 0
        
        with self.progress_manager.create_progress_display(len(self.stats.missing_covers)) as progress:
            task = progress.add_task(
                "Searching covers...",
                total=len(self.stats.missing_covers),
                network_status=self.network_monitor.get_status(),
                errors=0,
                duplicates=0
            )
            
            for file_path_str in self.stats.missing_covers:
                file_path = Path(file_path_str)
                if not file_path.exists():
                    progress.advance(task)
                    continue
                
                progress.update(task, description=file_path.name)
                
                try:
                    # Re-extract metadata
                    audio_file = mutagen.File(file_path)
                    if audio_file:
                        metadata = self._extract_comprehensive_metadata(audio_file, file_path)
                        
                        # Try to fetch cover art
                        if metadata['album'] != 'Unknown Album':
                            cover_art = self.metadata_enhancer.fetch_cover_art(
                                metadata['artist'], metadata['album']
                            )
                            
                            if cover_art:
                                # Embed the cover art
                                self.metadata_enhancer.embed_comprehensive_metadata(
                                    file_path, metadata, None, cover_art
                                )
                                found_covers += 1
                                console.print(f"\n🖼️  Found cover for: {metadata['artist']} - {metadata['album']}")
                
                except Exception as e:
                    logger = logging.getLogger("music_organizer")
                    logger.error(f"Error retrying cover for {file_path}: {e}")
                
                progress.advance(task)
        
        if found_covers > 0:
            console.print(f"\n[green]✅ Found {found_covers} additional covers![/green]")
            self.stats.covers_found += found_covers
        else:
            console.print("\n[yellow]⚠️  No additional covers found[/yellow]")
    
    def _retry_missing_lyrics(self):
        """Retry fetching lyrics for songs that didn't have them"""
        found_lyrics = 0
        
        with self.progress_manager.create_progress_display(len(self.stats.missing_lyrics)) as progress:
            task = progress.add_task(
                "Searching lyrics...",
                total=len(self.stats.missing_lyrics),
                network_status=self.network_monitor.get_status(),
                errors=0,
                duplicates=0
            )
            
            for file_path_str in self.stats.missing_lyrics:
                file_path = Path(file_path_str)
                if not file_path.exists():
                    progress.advance(task)
                    continue
                
                progress.update(task, description=file_path.name)
                
                try:
                    # Re-extract metadata
                    audio_file = mutagen.File(file_path)
                    if audio_file:
                        metadata = self._extract_comprehensive_metadata(audio_file, file_path)
                        
                        # Try to fetch lyrics
                        lyrics = self.metadata_enhancer.fetch_lyrics(
                            metadata['artist'], metadata['title']
                        )
                        
                        if lyrics:
                            # Embed the lyrics
                            self.metadata_enhancer.embed_comprehensive_metadata(
                                file_path, metadata, lyrics, None
                            )
                            found_lyrics += 1
                            console.print(f"\n📝 Found lyrics for: {metadata['artist']} - {metadata['title']}")
                
                except Exception as e:
                    logger = logging.getLogger("music_organizer")
                    logger.error(f"Error retrying lyrics for {file_path}: {e}")
                
                progress.advance(task)
        
        if found_lyrics > 0:
            console.print(f"\n[green]✅ Found {found_lyrics} additional lyrics![/green]")
            self.stats.lyrics_found += found_lyrics
        else:
            console.print("\n[yellow]⚠️  No additional lyrics found[/yellow]")
    
    def _get_music_files(self, root_path: Path) -> List[Path]:
        """Recursively find all supported music files"""
        extensions = {'.mp3', '.flac', '.wav', '.m4a', '.ogg', '.wma'}
        music_files = []
        
        for file_path in root_path.rglob('*'):
            if file_path.suffix.lower() in extensions and file_path.is_file():
                music_files.append(file_path)
        
        return music_files
    
    def _handle_duplicates(self, music_files: List[Path]):
        """Advanced duplicate detection and handling"""
        logger = logging.getLogger("music_organizer")
        console.print("\n[yellow]🔍 Checking for duplicates...[/yellow]")
        
        duplicates = self.duplicate_manager.find_duplicates(
            music_files, self.config.max_workers
        )
        
        if not duplicates:
            console.print("[green]✅ No duplicates found[/green]")
            logger.info("No duplicates found")
            return
        
        console.print(f"[red]⚠️  Found {len(duplicates)} groups of duplicates[/red]")
        logger.warning(f"Found {len(duplicates)} groups of duplicates")
        
        for i, (hash_val, files) in enumerate(duplicates.items(), 1):
            console.print(f"\n[bold]Duplicate Group {i}:[/bold]")
            
            # Create detailed table showing duplicates
            table = Table()
            table.add_column("File", style="cyan")
            table.add_column("Size", style="green")
            table.add_column("Path", style="dim")
            
            for file_path in files:
                try:
                    size = file_path.stat().st_size
                    size_mb = size / (1024 * 1024)
                    table.add_row(
                        file_path.name,
                        f"{size_mb:.2f} MB",
                        str(file_path.parent)
                    )
                except Exception:
                    table.add_row(file_path.name, "Unknown", str(file_path.parent))
            
            console.print(table)
            
            # Handle duplicate deletion
            if Confirm.ask("Delete duplicates? (keeps the first file)"):
                # Keep the first file, delete others
                for file_to_delete in files[1:]:
                    try:
                        send2trash(str(file_to_delete))
                        music_files.remove(file_to_delete)
                        self.stats.duplicates += 1
                        console.print(f"[red]🗑️  Deleted: {file_to_delete.name}[/red]")
                        logger.info(f"Deleted duplicate: {file_to_delete}")
                    except Exception as e:
                        console.print(f"[red]❌ Failed to delete {file_to_delete}: {e}[/red]")
                        logger.error(f"Failed to delete duplicate {file_to_delete}: {e}")
    
    def _process_files(self, music_files: List[Path]):
        """Process all music files with clean single progress tracking"""
        logger = logging.getLogger("music_organizer")
        console.print(f"\n[yellow]🎵 Processing {len(music_files)} files...[/yellow]")
        
        with self.progress_manager.create_progress_display(len(music_files)) as progress:
            task = progress.add_task(
                "Starting...",
                total=len(music_files),
                network_status=self.network_monitor.get_status(),
                errors=self.stats.errors,
                duplicates=self.stats.duplicates
            )
            
            for file_path in music_files:
                try:
                    # Update progress with current file
                    progress.update(
                        task,
                        description=file_path.name,
                        network_status=self.network_monitor.get_status(),
                        errors=self.stats.errors,
                        duplicates=self.stats.duplicates
                    )
                    
                    # Process the file and capture results
                    result = self._process_single_file(file_path)
                    
                    # Show results above progress bar
                    if result:
                        if result.get('lyrics_found'):
                            console.print(f"📝 Found lyrics for: {result['artist']} - {result['title']}")
                        else:
                            self.stats.missing_lyrics.append(str(result['output_path']))
                            
                        if result.get('cover_found'):
                            console.print(f"🖼️  Found cover for: {result['artist']} - {result['album']}")
                        else:
                            self.stats.missing_covers.append(str(result['output_path']))
                            
                        console.print(f"INFO     Processed: {file_path.name} -> {result['output_filename']}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    self.stats.errors += 1
                    console.print(f"\n[red]ERROR    Failed to process: {file_path.name}[/red]")
                
                progress.advance(task)
    
    def _process_single_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process individual music file with comprehensive metadata enhancement"""
        logger = logging.getLogger("music_organizer")
        
        try:
            # Load and parse metadata
            audio_file = mutagen.File(file_path)
            if not audio_file:
                logger.warning(f"Could not read metadata from {file_path}")
                self.stats.skipped += 1
                return None
            
            metadata = self._extract_comprehensive_metadata(audio_file, file_path)
            
            # Generate organized output path with new naming convention
            output_path = self._generate_output_path(metadata, file_path.suffix)
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle naming conflicts
            output_path = self._resolve_file_conflict(output_path)
            
            # Copy or move file based on configuration
            action = "MOVED" if self.config.mode == "move" else "COPIED"
            if self.config.mode == "move":
                shutil.move(str(file_path), str(output_path))
            else:
                shutil.copy2(str(file_path), str(output_path))
            
            # Enhance metadata with online services if connected and enabled
            lyrics_found = False
            cover_found = False
            lyrics = None
            cover_art = None
            
            
            
            if self.network_monitor.is_connected:
                if self.config.fetch_lyrics:
                    lyrics = self.metadata_enhancer.fetch_lyrics(
                        metadata['artist'], metadata['title']
                    )
                    if lyrics:
                        self.stats.lyrics_found += 1
                        # **NEW**: override metadata['language'] based on lyrics
                metadata['language'] = self._detect_language(
                        metadata['artist'], metadata['title'], lyrics
                )
                
                if self.config.fetch_covers and metadata['album'] != 'Unknown Album':
                    cover_art = self.metadata_enhancer.fetch_cover_art(
                        metadata['artist'], metadata['album']
                    )
                    if cover_art:
                        cover_found = True
            
            # Embed comprehensive metadata including lyrics and cover
            self.metadata_enhancer.embed_comprehensive_metadata(
                output_path, metadata, lyrics, cover_art
            )
            
            # Update statistics
            file_size = file_path.stat().st_size
            self.stats.total_size += file_size
            self.stats.processed += 1
            
            if lyrics_found:
                self.stats.lyrics_found += 1
            if cover_found:
                self.stats.covers_found += 1
            
            # Log to CSV report with enhanced metadata
            self.csv_writer.writerow([
                str(file_path), str(output_path), metadata['artist'], 
                metadata['title'], metadata['album'], metadata['genre'], metadata['year'],
                action, time.strftime('%Y-%m-%d %H:%M:%S'), f"{file_size / (1024*1024):.2f}",
                "Yes" if lyrics_found else "No", "Yes" if cover_found else "No",
                metadata.get('language', 'Unknown'), metadata.get('contributing_artists', '')
            ])
            
            logger.info(f"Processed: {file_path.name} -> {output_path.name}")
            
            return {
                'artist': metadata['artist'],
                'title': metadata['title'],
                'album': metadata['album'],
                'output_filename': output_path.name,
                'output_path': output_path,
                'lyrics_found': lyrics_found,
                'cover_found': cover_found
            }
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            self.stats.errors += 1
            return None
    
    def _extract_comprehensive_metadata(self, audio_file, file_path: Path) -> Dict[str, Any]:
        """Extract comprehensive Windows-compatible metadata from audio file"""
        metadata = {
            'artist': 'Unknown Artist',
            'album': 'Unknown Album',
            'title': file_path.stem,
            'track': 1,
            'year': 'Unknown',
            'genre': 'Unknown',
            'language': 'Unknown',
            'album_artist': '',
            'contributing_artists': '',
            'ext': file_path.suffix[1:].lower()
        }
        
        # Extract metadata from tags
        if hasattr(audio_file, 'tags') and audio_file.tags:
            tags = audio_file.tags
            
            # Handle different tag formats (ID3, Vorbis, MP4)
            if hasattr(tags, 'get'):
                metadata.update({
                    'artist': self._get_tag_value(tags, ['TPE1', 'ARTIST', '\xa9ART']) or metadata['artist'],
                    'album': self._get_tag_value(tags, ['TALB', 'ALBUM', '\xa9alb']) or metadata['album'],
                    'title': self._get_tag_value(tags, ['TIT2', 'TITLE', '\xa9nam']) or metadata['title'],
                    'year': self._get_tag_value(tags, ['TDRC', 'DATE', '\xa9day']) or metadata['year'],
                    'genre': self._get_tag_value(tags, ['TCON', 'GENRE', '\xa9gen']) or metadata['genre'],
                    'album_artist': self._get_tag_value(tags, ['TPE2', 'ALBUMARTIST', 'aART']) or '',
                    'contributing_artists': self._get_tag_value(tags, ['TPE3', 'PERFORMER', '\xa9wrt']) or '',
                })
                
                # Extract track number
                track_tag = self._get_tag_value(tags, ['TRCK', 'TRACKNUMBER', 'trkn'])
                if track_tag:
                    try:
                        metadata['track'] = int(str(track_tag).split('/')[0])
                    except (ValueError, IndexError):
                        pass
                
                # Clean up year - extract just the year part if it's a date
                if metadata['year'] and metadata['year'] != 'Unknown':
                    year_match = re.search(r'\d{4}', str(metadata['year']))
                    if year_match:
                        metadata['year'] = year_match.group()
                    else:
                        metadata['year'] = 'Unknown'
                
                # Detect language based on content (basic detection)
                metadata['language'] = self._detect_language(metadata['artist'], metadata['title'])
        
        # Clean metadata for filesystem compatibility
        for key, value in metadata.items():
            if isinstance(value, str):
                metadata[key] = self._clean_filename(value)
        
        return metadata
    
    def _detect_language(self,
                         artist: str,
                         title: str,
                         lyrics: Optional[str] = None
                        ) -> str:
        """
        Enhanced language detection:
        1) Try langdetect on lyrics (if available), else artist+title
        2) Map lang codes to your folders
        3) Fallback to simple regex if langdetect fails
        """
        text = lyrics or f"{artist} {title}"
        lang_code = 'en'
        try:
            lang_code = detect(text)
        except Exception:
            pass

        if lang_code == 'fa':
            return "Persian"
        if lang_code == 'ar':
            return "Arabic"
        # you can add more mappings here, e.g. 'es'->"Spanish", etc.
        # fallback: look for Persian/Arabic unicode if detect() was inconclusive
        if re.search(r'[\u0600-\u06FF]', text):
            return "Persian"  # catches both Arabic & Persian scripts
        return "English"

    
    def _get_tag_value(self, tags, keys: List[str]) -> Optional[str]:
        """Get tag value trying multiple possible keys"""
        for key in keys:
            if key in tags:
                value = tags[key]
                if isinstance(value, list) and value:
                    return str(value[0])
                return str(value) if value else None
        return None
    
    def _generate_output_path(self, metadata: Dict[str, Any], extension: str) -> Path:
        """Generate organized output path with language and artist organization"""
        # Clean filename components
        artist = metadata['artist']
        title = metadata['title'] 
        album = metadata['album']
        language = metadata['language']
        
        # Create filename: "Artist - Title [Album]" or "Artist - Title"
        if album and album != 'Unknown Album':
            filename = f"{artist} - {title} [{album}]{extension}"
        else:
            filename = f"{artist} - {title}{extension}"
        
        # Determine output path based on organization settings
        base_path = self.output_dir
        
        # Language-based organization (highest priority)
        if self.config.organize_by_language:
            base_path = base_path / language
        
        # Artist-based organization
        if self.config.organize_by_artist:
            base_path = base_path / artist
        
        return base_path / filename
    
    def _resolve_file_conflict(self, path: Path) -> Path:
        """Resolve naming conflicts by adding numbers or timestamps"""
        if not path.exists():
            return path
        
        base = path.stem
        ext = path.suffix
        parent = path.parent
        counter = 1
        
        while counter < 1000:  # Safety limit
            new_path = parent / f"{base} ({counter}){ext}"
            if not new_path.exists():
                return new_path
            counter += 1
        
        # Fallback to timestamp if too many conflicts
        timestamp = int(time.time())
        return parent / f"{base}_{timestamp}{ext}"
    
    def _clean_filename(self, filename: str) -> str:
        """Clean filename for cross-platform filesystem compatibility"""
        if not filename or filename.isspace():
            return "Unknown"
        
        # Replace invalid filesystem characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length and clean up whitespace
        filename = filename.strip(' .')
        return filename[:100] if filename else "Unknown"
    
    def _show_initial_stats(self):
        """Display initial processing statistics"""
        logger = logging.getLogger("music_organizer")
        
        # Create detailed statistics table
        table = Table(title="🎵 Processing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        
        table.add_row("Files Processed", str(self.stats.processed))
        table.add_row("Files Skipped", str(self.stats.skipped))
        table.add_row("Errors Encountered", str(self.stats.errors))
        table.add_row("Duplicates Removed", str(self.stats.duplicates))
        table.add_row("Lyrics Found & Embedded", str(self.stats.lyrics_found))
        table.add_row("Covers Downloaded & Embedded", str(self.stats.covers_found))
        table.add_row("Songs Missing Lyrics", str(len(self.stats.missing_lyrics)))
        table.add_row("Songs Missing Covers", str(len(self.stats.missing_covers)))
        
        # Format total size processed
        size_gb = self.stats.total_size / (1024 ** 3)
        table.add_row("Total Size Processed", f"{size_gb:.2f} GB")
        
        console.print(table)
    
    def _show_final_stats(self):
        """Display final comprehensive processing statistics"""
        logger = logging.getLogger("music_organizer")
        
        # Create detailed statistics table
        table = Table(title="🎵 Final Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        
        table.add_row("Files Processed", str(self.stats.processed))
        table.add_row("Files Skipped", str(self.stats.skipped))
        table.add_row("Errors Encountered", str(self.stats.errors))
        table.add_row("Duplicates Removed", str(self.stats.duplicates))
        table.add_row("Total Lyrics Found & Embedded", str(self.stats.lyrics_found))
        table.add_row("Total Covers Downloaded & Embedded", str(self.stats.covers_found))
        
        # Format total size processed
        size_gb = self.stats.total_size / (1024 ** 3)
        table.add_row("Total Size Processed", f"{size_gb:.2f} GB")
        
        console.print(table)
        
        # Show warnings if there were errors
        if self.stats.errors > 0:
            console.print(f"\n[yellow]⚠️  {self.stats.errors} files had errors. Check {self.log_file} for details.[/yellow]")
        
        console.print(f"\n[bold green]✅ Organization complete![/bold green]")
        console.print(f"[dim]📊 Report saved to: {self.report_csv}[/dim]")
        console.print(f"[dim]📋 Log saved to: {self.log_file}[/dim]")
        
        logger.info(f"Final stats: {self.stats.processed} processed, {self.stats.errors} errors, {self.stats.lyrics_found} lyrics, {self.stats.covers_found} covers")

# -----------------------------------------------------------------------------
# SECTION 8: Command Line Interface
# -----------------------------------------------------------------------------

@click.command()
@click.option('--root', '-r', type=click.Path(exists=True), help='Root music folder')
@click.option('--output', '-o', type=click.Path(), help='Output folder')
@click.option('--mode', type=click.Choice(['copy', 'move']), help='Operation mode')
@click.option('--organize-by-artist', is_flag=True, help='Organize into artist folders')
@click.option('--organize-by-language', is_flag=True, help='Organize by language folders')
@click.option('--workers', type=int, help='Number of parallel workers')
@click.option('--no-lyrics', is_flag=True, help='Skip lyrics fetching')
@click.option('--no-covers', is_flag=True, help='Skip cover art fetching')
@click.option('--config', is_flag=True, help='Show current configuration')
def main(root, output, mode, organize_by_artist, organize_by_language, workers, no_lyrics, no_covers, config):
    """Enhanced Music File Organizer v2.4
    
    A comprehensive tool for organizing your music collection with advanced features
    including duplicate detection, metadata enhancement, and flexible organization.
    """
    
    # Load existing configuration
    app_config = Config.load()
    
    # Update config with CLI arguments
    if root:
        app_config.root_folder = root
    if output:
        app_config.output_folder = output
    if mode:
        app_config.mode = mode
    if organize_by_artist:
        app_config.organize_by_artist = True
    if organize_by_language:
        app_config.organize_by_language = True
    if workers:
        app_config.max_workers = workers
    if no_lyrics:
        app_config.fetch_lyrics = False
    if no_covers:
        app_config.fetch_covers = False
    
    # Show configuration if requested
    if config:
        organization_mode = "Artist folders" if app_config.organize_by_artist else "Flat (all in output folder)"
        language_mode = "Language-based folders" if app_config.organize_by_language else "Mixed languages"
        console.print(Panel(
            f"[cyan]Root Folder:[/cyan] {app_config.root_folder}\n"
            f"[cyan]Output Folder:[/cyan] {app_config.output_folder}\n"
            f"[cyan]Mode:[/cyan] {app_config.mode}\n"
            f"[cyan]Organization:[/cyan] {organization_mode}\n"
            f"[cyan]Language Sort:[/cyan] {language_mode}\n"
            f"[cyan]Workers:[/cyan] {app_config.max_workers}\n"
            f"[cyan]Fetch Lyrics:[/cyan] {app_config.fetch_lyrics}\n"
            f"[cyan]Fetch Covers:[/cyan] {app_config.fetch_covers}",
            title="Current Configuration"
        ))
        return
    
    # ALWAYS ask for root folder
    if not root:
        app_config.root_folder = Prompt.ask(
            "[bold cyan]Enter the root music folder path[/bold cyan]", 
            default=str(Path.cwd())
        )
    
    # ALWAYS ask for output folder with smart default
    if not output:
        default_output = str(Path(app_config.root_folder) / "output")
        app_config.output_folder = Prompt.ask(
            "[bold cyan]Enter the output folder path[/bold cyan]",
            default=default_output
        )
    
    # Interactive mode selection (only if not provided via CLI)
    if not mode:
        app_config.mode = Prompt.ask(
            "[bold cyan]Keep originals (copy) or cut (move)?[/bold cyan]",
            choices=["copy", "move"],
            default="copy"
        )
    
    # Interactive organization preference (only if not provided via CLI)
    if not organize_by_artist:
        app_config.organize_by_artist = Confirm.ask(
            "[bold cyan]Organize files into artist folders? [y/n][/bold cyan]",
            default=False
        )
    
    # Interactive language-based organization preference
    if not organize_by_language:
        app_config.organize_by_language = Confirm.ask(
            "[bold cyan]Sort them based on language? [y/n][/bold cyan]",
            default=False
        )
    
    # Interactive lyrics fetching preference (only if not disabled via CLI)
    if not no_lyrics:
        app_config.fetch_lyrics = Confirm.ask(
            "[bold cyan]Fetch lyrics from online sources? [y/n][/bold cyan]",
            default=True
        )
    
    # Interactive cover art fetching preference (only if not disabled via CLI)
    if not no_covers:
        app_config.fetch_covers = Confirm.ask(
            "[bold cyan]Fetch cover art from online sources? [y/n][/bold cyan]",
            default=True
        )
    
    # Validate that root folder exists
    root_path = Path(app_config.root_folder)
    if not root_path.exists():
        console.print(f"[red]❌ Root folder does not exist: {root_path}[/red]")
        return
    
    # Save configuration for future use
    app_config.save()
    
    # Start the organization process
    try:
        organizer = MusicOrganizer(app_config)
        organizer.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Process interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Fatal error: {e}[/red]")
        logger = logging.getLogger("music_organizer")
        logger.exception("Fatal error occurred")

# -----------------------------------------------------------------------------
# SECTION 9: Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()


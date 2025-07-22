#!/usr/bin/env python3
"""
Enhanced Music File Organizer
A comprehensive tool for organizing music files with advanced features.
"""

import subprocess
import sys
import importlib
from pathlib import Path

def check_and_install_libraries():
    """Check for required libraries and install them if missing"""
    required_libraries = {
        'click': 'click',
        'rich': 'rich', 
        'mutagen': 'mutagen',
        'requests': 'requests',
        'send2trash': 'send2trash'
    }
    
    missing_libraries = []
    
    # Check which libraries are missing
    for lib_name, pip_name in required_libraries.items():
        try:
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
    else:
        print("All libraries are installed.")

# Install libraries before importing them
check_and_install_libraries()

# Now import the libraries
import asyncio
import hashlib
import json
import logging
import platform
import shutil
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Set, Any
from urllib.parse import quote_plus

import click
import mutagen
import requests
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text
from send2trash import send2trash

# Configuration
APP_DIR = Path.home() / '.music_organizer'
APP_DIR.mkdir(exist_ok=True)

CONFIG_FILE = APP_DIR / 'config.json'
PROGRESS_FILE = APP_DIR / 'progress.json'
STATS_FILE = APP_DIR / 'stats.json'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("music_organizer")

console = Console()

@dataclass
class Config:
    """Configuration settings"""
    root_folder: str = ""
    output_folder: str = ""
    mode: str = "copy"  # copy or move
    pattern: str = "artist_album"
    auto_delete_duplicates: bool = False
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
                logger.warning(f"Failed to load config: {e}")
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
    
    def save(self):
        """Save stats to file"""
        with open(STATS_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)

class NetworkMonitor:
    """Monitor network connectivity and latency"""
    
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
        """Check network connectivity and measure ping"""
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
        """Get formatted network status"""
        if self.is_connected:
            return f"🌐 {self.ping_ms}ms"
        return "🔴 Offline"

class DuplicateManager:
    """Manage duplicate file detection and handling"""
    
    def __init__(self):
        self.db_path = APP_DIR / "duplicates.db"
        self.setup_database()
    
    def setup_database(self):
        """Setup duplicates database"""
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
        """Get file hash with caching"""
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
            logger.error(f"Hash calculation failed for {file_path}: {e}")
            return None
    
    def _get_cached_hash(self, file_path: Path) -> Optional[str]:
        """Get cached hash if file hasn't changed"""
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
        """Cache file hash"""
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
            logger.debug(f"Failed to cache hash: {e}")
    
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
                    logger.error(f"Error processing {file_path}: {e}")
        
        return {k: v for k, v in duplicates.items() if len(v) > 1}

class MetadataEnhancer:
    """Enhance metadata using various sources"""
    
    def __init__(self, timeout: int = 15, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MusicOrganizer/2.0 (https://github.com/musicorganizer)'
        })
    
    def fetch_lyrics(self, artist: str, title: str) -> Optional[str]:
        """Fetch lyrics from Genius API"""
        if not artist or not title:
            return None
            
        try:
            # Clean up artist and title
            artist = self._clean_string(artist)
            title = self._clean_string(title)
            
            # Search for the song
            search_url = "https://api.genius.com/search"
            params = {"q": f"{artist} {title}"}
            
            response = self.session.get(search_url, params=params, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            data = response.json()
            hits = data.get("response", {}).get("hits", [])
            
            if not hits:
                return None
            
            # Get the first result's lyrics URL
            song_url = hits[0]["result"]["url"]
            
            # For demo purposes, return a placeholder
            # In real implementation, you'd scrape the lyrics page
            return f"Lyrics found for {artist} - {title}"
            
        except Exception as e:
            logger.debug(f"Lyrics fetch failed: {e}")
            return None
    
    def fetch_cover_art(self, artist: str, album: str) -> Optional[bytes]:
        """Fetch cover art from Last.fm API"""
        if not artist or not album:
            return None
            
        try:
            artist = self._clean_string(artist)
            album = self._clean_string(album)
            
            # Last.fm API endpoint
            api_url = "https://ws.audioscrobbler.com/2.0/"
            params = {
                "method": "album.getinfo",
                "api_key": "your_lastfm_api_key",  # You'd need to get this
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
            
            # Get the largest image
            for image in reversed(images):
                if image.get("#text"):
                    img_response = self.session.get(image["#text"], timeout=self.timeout)
                    if img_response.status_code == 200:
                        return img_response.content
            
            return None
            
        except Exception as e:
            logger.debug(f"Cover art fetch failed: {e}")
            return None
    
    def _clean_string(self, s: str) -> str:
        """Clean string for API queries"""
        return s.strip().replace("&", "and")

class ProgressManager:
    """Enhanced progress tracking"""
    
    def __init__(self, network_monitor: NetworkMonitor):
        self.network_monitor = network_monitor
        self.stats = FileStats()
    
    def create_progress(self) -> Progress:
        """Create enhanced progress display"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[current_file]}", justify="left"),
            BarColumn(bar_width=None),
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

class MusicOrganizer:
    """Main music organization class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.network_monitor = NetworkMonitor(config.ping_interval)
        self.duplicate_manager = DuplicateManager()
        self.metadata_enhancer = MetadataEnhancer(config.request_timeout, config.max_retries)
        self.progress_manager = ProgressManager(self.network_monitor)
        self.stats = FileStats()
        
        # Organization patterns
        self.patterns = {
            'artist_album': '{artist}/{album}/{track:02d} - {title}.{ext}',
            'genre_artist': '{genre}/{artist}/{title}.{ext}',
            'year_artist': '{year}/{artist} - {album}/{title}.{ext}',
            'flat_artist': '{artist} - {title} [{album}].{ext}',
            'artist_only': '{artist}/{title}.{ext}'
        }
    
    def start(self):
        """Start the organization process"""
        try:
            self.network_monitor.start_monitoring()
            self._run_organization()
        finally:
            self.network_monitor.stop_monitoring()
            self.stats.save()
    
    def _run_organization(self):
        """Main organization workflow"""
        console.print(Panel.fit(
            "[bold green]🎵 Enhanced Music Organizer v2.0[/bold green]\n"
            f"[cyan]Root:[/cyan] {self.config.root_folder}\n"
            f"[cyan]Output:[/cyan] {self.config.output_folder}\n"
            f"[cyan]Mode:[/cyan] {self.config.mode.upper()}\n"
            f"[cyan]Pattern:[/cyan] {self.config.pattern}",
            title="Configuration"
        ))
        
        # Get all music files
        console.print("\n[yellow]📁 Scanning for music files...[/yellow]")
        music_files = self._get_music_files(Path(self.config.root_folder))
        
        if not music_files:
            console.print("[red]❌ No music files found![/red]")
            return
        
        console.print(f"[green]✅ Found {len(music_files)} music files[/green]")
        
        # Check for duplicates
        if len(music_files) > 1:
            self._handle_duplicates(music_files)
        
        # Process files
        self._process_files(music_files)
        
        # Show final statistics
        self._show_final_stats()
    
    def _get_music_files(self, root_path: Path) -> List[Path]:
        """Get all music files from root directory"""
        extensions = {'.mp3', '.flac', '.wav', '.m4a', '.ogg', '.wma'}
        music_files = []
        
        for file_path in root_path.rglob('*'):
            if file_path.suffix.lower() in extensions and file_path.is_file():
                music_files.append(file_path)
        
        return music_files
    
    def _handle_duplicates(self, music_files: List[Path]):
        """Handle duplicate file detection and removal"""
        console.print("\n[yellow]🔍 Checking for duplicates...[/yellow]")
        
        duplicates = self.duplicate_manager.find_duplicates(
            music_files, self.config.max_workers
        )
        
        if not duplicates:
            console.print("[green]✅ No duplicates found[/green]")
            return
        
        console.print(f"[red]⚠️  Found {len(duplicates)} groups of duplicates[/red]")
        
        for i, (hash_val, files) in enumerate(duplicates.items(), 1):
            console.print(f"\n[bold]Duplicate Group {i}:[/bold]")
            
            # Create table showing duplicates
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
            
            if self.config.auto_delete_duplicates or Confirm.ask(
                "Delete duplicates? (keeps the first file)"
            ):
                # Keep the first file, delete others
                for file_to_delete in files[1:]:
                    try:
                        send2trash(str(file_to_delete))
                        music_files.remove(file_to_delete)
                        self.stats.duplicates += 1
                        console.print(f"[red]🗑️  Deleted: {file_to_delete.name}[/red]")
                    except Exception as e:
                        console.print(f"[red]❌ Failed to delete {file_to_delete}: {e}[/red]")
    
    def _process_files(self, music_files: List[Path]):
        """Process all music files"""
        console.print(f"\n[yellow]🎵 Processing {len(music_files)} files...[/yellow]")
        
        output_dir = Path(self.config.output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with self.progress_manager.create_progress() as progress:
            task = progress.add_task(
                "Processing files...",
                total=len(music_files),
                current_file="Starting...",
                network_status=self.network_monitor.get_status(),
                errors=self.stats.errors,
                duplicates=self.stats.duplicates
            )
            
            for file_path in music_files:
                try:
                    progress.update(
                        task,
                        current_file=file_path.name,
                        network_status=self.network_monitor.get_status(),
                        errors=self.stats.errors,
                        duplicates=self.stats.duplicates
                    )
                    
                    self._process_single_file(file_path, output_dir, progress, task)
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    self.stats.errors += 1
                
                progress.advance(task)
                time.sleep(0.1)  # Brief pause for UI responsiveness
    
    def _process_single_file(self, file_path: Path, output_dir: Path, progress, task):
        """Process a single music file"""
        try:
            # Load metadata
            audio_file = mutagen.File(file_path)
            if not audio_file:
                logger.warning(f"Could not read metadata from {file_path}")
                self.stats.skipped += 1
                return
            
            metadata = self._extract_metadata(audio_file, file_path)
            
            # Enhance metadata if network is available
            if self.network_monitor.is_connected:
                self._enhance_metadata(metadata, audio_file)
            
            # Generate output path
            output_path = self._generate_output_path(metadata, output_dir, file_path.suffix)
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle file conflicts
            output_path = self._resolve_file_conflict(output_path)
            
            # Copy or move file
            if self.config.mode == "move":
                shutil.move(str(file_path), str(output_path))
            else:
                shutil.copy2(str(file_path), str(output_path))
            
            # Update file size statistics
            self.stats.total_size += file_path.stat().st_size
            self.stats.processed += 1
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            self.stats.errors += 1
    
    def _extract_metadata(self, audio_file, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from audio file"""
        metadata = {
            'artist': 'Unknown Artist',
            'album': 'Unknown Album',
            'title': file_path.stem,
            'track': 1,
            'year': '',
            'genre': '',
            'ext': file_path.suffix[1:].lower()
        }
        
        # Extract from tags
        if hasattr(audio_file, 'tags') and audio_file.tags:
            tags = audio_file.tags
            
            # Handle different tag formats
            if hasattr(tags, 'get'):  # ID3 tags
                metadata.update({
                    'artist': self._get_tag_value(tags, ['TPE1', 'ARTIST', '\xa9ART']) or metadata['artist'],
                    'album': self._get_tag_value(tags, ['TALB', 'ALBUM', '\xa9alb']) or metadata['album'],
                    'title': self._get_tag_value(tags, ['TIT2', 'TITLE', '\xa9nam']) or metadata['title'],
                    'year': self._get_tag_value(tags, ['TDRC', 'DATE', '\xa9day']) or metadata['year'],
                    'genre': self._get_tag_value(tags, ['TCON', 'GENRE', '\xa9gen']) or metadata['genre'],
                })
                
                # Track number
                track_tag = self._get_tag_value(tags, ['TRCK', 'TRACKNUMBER', 'trkn'])
                if track_tag:
                    try:
                        metadata['track'] = int(str(track_tag).split('/')[0])
                    except (ValueError, IndexError):
                        pass
        
        # Clean metadata
        for key, value in metadata.items():
            if isinstance(value, str):
                metadata[key] = self._clean_filename(value)
        
        return metadata
    
    def _get_tag_value(self, tags, keys: List[str]) -> Optional[str]:
        """Get tag value trying multiple keys"""
        for key in keys:
            if key in tags:
                value = tags[key]
                if isinstance(value, list) and value:
                    return str(value[0])
                return str(value) if value else None
        return None
    
    def _enhance_metadata(self, metadata: Dict[str, Any], audio_file):
        """Enhance metadata using online services"""
        try:
            # Fetch lyrics
            if self.config.fetch_lyrics:
                lyrics = self.metadata_enhancer.fetch_lyrics(
                    metadata['artist'], metadata['title']
                )
                if lyrics:
                    self.stats.lyrics_found += 1
                    # In a real implementation, you'd save lyrics to the file
            
            # Fetch cover art
            if self.config.fetch_covers:
                cover_art = self.metadata_enhancer.fetch_cover_art(
                    metadata['artist'], metadata['album']
                )
                if cover_art:
                    self.stats.covers_found += 1
                    # In a real implementation, you'd embed the cover art
                    
        except Exception as e:
            logger.debug(f"Metadata enhancement failed: {e}")
    
    def _generate_output_path(self, metadata: Dict[str, Any], output_dir: Path, extension: str) -> Path:
        """Generate output file path based on pattern"""
        pattern = self.patterns.get(self.config.pattern, self.patterns['artist_album'])
        
        try:
            filename = pattern.format(**metadata)
        except KeyError as e:
            logger.warning(f"Missing metadata key {e}, using fallback pattern")
            filename = self.patterns['artist_only'].format(**metadata)
        
        return output_dir / filename
    
    def _resolve_file_conflict(self, path: Path) -> Path:
        """Resolve file naming conflicts"""
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
        
        # Fallback to timestamp
        timestamp = int(time.time())
        return parent / f"{base}_{timestamp}{ext}"
    
    def _clean_filename(self, filename: str) -> str:
        """Clean filename for filesystem compatibility"""
        if not filename or filename.isspace():
            return "Unknown"
        
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length and clean up
        filename = filename.strip(' .')
        return filename[:100] if filename else "Unknown"
    
    def _show_final_stats(self):
        """Display final processing statistics"""
        # Create statistics table
        table = Table(title="Processing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        
        table.add_row("Files Processed", str(self.stats.processed))
        table.add_row("Files Skipped", str(self.stats.skipped))
        table.add_row("Errors", str(self.stats.errors))
        table.add_row("Duplicates Removed", str(self.stats.duplicates))
        table.add_row("Lyrics Found", str(self.stats.lyrics_found))
        table.add_row("Covers Found", str(self.stats.covers_found))
        
        # Format total size
        size_gb = self.stats.total_size / (1024 ** 3)
        table.add_row("Total Size Processed", f"{size_gb:.2f} GB")
        
        console.print(table)
        
        if self.stats.errors > 0:
            console.print(f"\n[yellow]⚠️  {self.stats.errors} files had errors. Check logs for details.[/yellow]")
        
        console.print("\n[bold green]✅ Organization complete![/bold green]")

# CLI Interface
@click.command()
@click.option('--root', '-r', type=click.Path(exists=True), help='Root music folder')
@click.option('--output', '-o', type=click.Path(), help='Output folder')
@click.option('--mode', type=click.Choice(['copy', 'move']), help='Operation mode')
@click.option('--pattern', type=click.Choice(['artist_album', 'genre_artist', 'year_artist', 'flat_artist', 'artist_only']), help='Organization pattern')
@click.option('--workers', type=int, help='Number of parallel workers')
@click.option('--auto-delete-duplicates', is_flag=True, help='Automatically delete duplicates')
@click.option('--no-lyrics', is_flag=True, help='Skip lyrics fetching')
@click.option('--no-covers', is_flag=True, help='Skip cover art fetching')
@click.option('--config', is_flag=True, help='Show configuration')
def main(root, output, mode, pattern, workers, auto_delete_duplicates, no_lyrics, no_covers, config):
    """Enhanced Music File Organizer v2.0
    
    A comprehensive tool for organizing your music collection with advanced features
    including duplicate detection, metadata enhancement, and parallel processing.
    """
    
    # Load configuration
    app_config = Config.load()
    
    # Update config with CLI arguments
    if root:
        app_config.root_folder = root
    if output:
        app_config.output_folder = output
    if mode:
        app_config.mode = mode
    if pattern:
        app_config.pattern = pattern
    if workers:
        app_config.max_workers = workers
    if auto_delete_duplicates:
        app_config.auto_delete_duplicates = True
    if no_lyrics:
        app_config.fetch_lyrics = False
    if no_covers:
        app_config.fetch_covers = False
    
    # Show configuration if requested
    if config:
        console.print(Panel(
            f"[cyan]Root Folder:[/cyan] {app_config.root_folder}\n"
            f"[cyan]Output Folder:[/cyan] {app_config.output_folder}\n"
            f"[cyan]Mode:[/cyan] {app_config.mode}\n"
            f"[cyan]Pattern:[/cyan] {app_config.pattern}\n"
            f"[cyan]Workers:[/cyan] {app_config.max_workers}\n"
            f"[cyan]Auto Delete Duplicates:[/cyan] {app_config.auto_delete_duplicates}\n"
            f"[cyan]Fetch Lyrics:[/cyan] {app_config.fetch_lyrics}\n"
            f"[cyan]Fetch Covers:[/cyan] {app_config.fetch_covers}",
            title="Current Configuration"
        ))
        return
    
    # Validate required settings
    if not app_config.root_folder:
        app_config.root_folder = click.prompt("Enter root music folder", type=click.Path(exists=True))
    
    if not app_config.output_folder:
        app_config.output_folder = click.prompt("Enter output folder", type=click.Path())
    
    # Save configuration
    app_config.save()
    
    # Start organization
    try:
        organizer = MusicOrganizer(app_config)
        organizer.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Process interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Fatal error: {e}[/red]")
        logger.exception("Fatal error occurred")

if __name__ == "__main__":
    main()

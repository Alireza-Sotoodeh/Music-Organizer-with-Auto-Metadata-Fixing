 # 🎵 Enhanced Music File Organizer v2.4

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

A comprehensive, professional-grade music collection organizer with advanced metadata enhancement, intelligent duplicate detection, and flexible organization options.

## ✨ Key Features

### 🎯 **Core Functionality**
- **Smart File Organization**: Organize by artist, language, or both
- **Flexible Naming**: `"Artist - Title [Album]"` or `"Artist - Title"` format
- **Safe Operations**: Copy or move files with conflict resolution
- **Cross-Platform**: Full Windows, macOS, and Linux support

### 🔍 **Advanced Detection**
- **SHA-1 Duplicate Detection**: Intelligent caching with SQLite
- **Language Recognition**: Persian, English, Arabic auto-detection
- **Metadata Extraction**: Supports MP3, FLAC, M4A, OGG, WAV, WMA

### 🌐 **Online Enhancement**
- **Multi-Source Lyrics**: AZLyrics, Genius, Lyrics.com
- **Cover Art Fetching**: iTunes, Last.fm, Cover Art Archive, Deezer
- **Network Monitoring**: Real-time connectivity with ping display
- **Retry Mechanism**: Post-processing for missing metadata

### 📊 **Professional Reporting**
- **Comprehensive Logs**: Detailed operation logging
- **CSV Reports**: Complete processing history
- **Statistics Dashboard**: Rich console output with progress bars
- **Error Tracking**: Full error reporting and recovery

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Internet connection (optional, for metadata enhancement)

### Installation

1. **Clone or download the script:**
```bash
# Download the script file
wget https://example.com/music_organizer.py
# or
curl -O https://example.com/music_organizer.py
```

2. **Run the script (dependencies auto-install):**
```bash
python music_organizer.py
```

The script automatically installs required dependencies on first run:
- `click` - Command-line interface
- `rich` - Beautiful terminal output
- `mutagen` - Audio metadata handling
- `requests` - HTTP requests
- `beautifulsoup4` - Web scraping
- `send2trash` - Safe file deletion

## 📋 Usage Guide

### Interactive Mode (Recommended)
```bash
python music_organizer.py
```

The script will guide you through configuration:

1. **📁 Root Folder**: Source music directory
2. **📂 Output Folder**: Destination (default: `{root}/output`)
3. **⚙️ Operation Mode**: Copy or Move files
4. **🗂️ Artist Folders**: Organize by artist (default: No)
5. **🌍 Language Sorting**: Sort by language (default: No)
6. **📝 Fetch Lyrics**: Download lyrics (default: Yes)
7. **🖼️ Fetch Covers**: Download artwork (default: Yes)

### Command Line Mode
```bash
# Basic usage
python music_organizer.py --root "/path/to/music" --output "/path/to/organized"

# Advanced options
python music_organizer.py \
  --root "/Users/music" \
  --output "/Users/organized" \
  --mode move \
  --organize-by-artist \
  --organize-by-language \
  --workers 8
```

### Post-Processing Retry
After initial processing, the script offers:

🖼️ 15 songs are missing covers. Search again for covers? [y/n] (n)
📝 8 songs are missing lyrics. Search again for lyrics? [y/n] (n)


## 📁 Output Structure

The output directory can be organized in different structures depending on the configuration used. Below are the supported organization formats:

### Standard Organization
Files are placed directly in the `Output/` directory with a consistent naming convention: `Artist - Song [Album].extension`.

```
Output/
├── Artist1 - Song1 [Album1].mp3
├── Artist2 - Song2.mp3
└── Artist3 - Song3 [Album2].flac
```

### Artist Folders
Files are grouped into subdirectories named after the artist, with the same naming convention for files.

```
Output/
├── Artist1/
│   ├── Artist1 - Song1 [Album1].mp3
│   └── Artist1 - Song2.mp3
├── Artist2/
│   └── Artist2 - Song3.flac
└── Artist3/
    └── Artist3 - Song4 [Album2].m4a
```

### Language + Artist Organization
Files are organized first by language, then by artist, with files following the same naming convention.

```
Output/
├── Persian/
│   ├── Artist1/
│   │   ├── Artist1 - Song1 [Album1].mp3
│   │   └── Artist1 - Song2.mp3
│   └── Artist2/
│       └── Artist2 - Song3.mp3
├── English/
│   └── Artist3/
│       └── Artist3 - Song4 [Album2].flac
└── Arabic/
    └── Artist4/
        └── Artist4 - Song5.ogg
```

## 📊 Reports & Logging

The script generates comprehensive reports in `{output}/reports/`:

### Files Generated
- **📋 `report.csv`**: Complete processing log with metadata
- **📄 `organizer.log`**: Detailed operation logs
- **📊 `stats.json`**: Processing statistics

### CSV Report Columns
| Column | Description |
|--------|-------------|
| Original Path | Source file location |
| New Path | Organized file location |
| Artist | Extracted artist name |
| Title | Song title |
| Album | Album name |
| Genre | Music genre |
| Year | Release year |
| Action | COPIED or MOVED |
| Timestamp | Processing time |
| File Size (MB) | File size |
| Lyrics Found | Yes/No |
| Cover Found | Yes/No |
| Language | Detected language |
| Contributing Artists | Additional artists |

## ⚙️ Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--root` | `-r` | Root music folder path | Interactive prompt |
| `--output` | `-o` | Output folder path | `{root}/output` |
| `--mode` | | Operation mode (`copy`/`move`) | `copy` |
| `--organize-by-artist` | | Create artist folders | `False` |
| `--organize-by-language` | | Create language folders | `False` |
| `--workers` | | Parallel processing threads | `4` |
| `--no-lyrics` | | Skip lyrics fetching | Fetch lyrics |
| `--no-covers` | | Skip cover art fetching | Fetch covers |
| `--config` | | Show current configuration | |

### Examples

```bash
# Process with all features enabled
python music_organizer.py --root "/music" --output "/organized" --organize-by-artist --organize-by-language

# Quick copy without online features
python music_organizer.py --root "/music" --no-lyrics --no-covers

# Move files with 8 parallel workers
python music_organizer.py --mode move --workers 8

# Show current configuration
python music_organizer.py --config
```

## 🎵 Supported Formats

| Format | Extension | Metadata | Lyrics | Cover Art |
|--------|-----------|----------|--------|-----------|
| MP3 | `.mp3` | ✅ Full ID3v2.4 | ✅ USLT + TIT3 | ✅ APIC |
| FLAC | `.flac` | ✅ Vorbis Comments | ✅ LYRICS field | ✅ Picture block |
| M4A/MP4 | `.m4a`, `.mp4` | ✅ iTunes tags | ✅ ©lyr | ✅ covr |
| OGG | `.ogg` | ✅ Vorbis Comments | ✅ LYRICS field | ❌ |
| WAV | `.wav` | ✅ ID3 tags | ✅ Limited | ❌ |
| WMA | `.wma` | ✅ ASF tags | ❌ | ❌ |

## 🌐 Online Sources

### Lyrics Sources
- **AZLyrics** - Comprehensive lyrics database
- **Genius** - Lyrics with annotations
- **Lyrics.com** - Professional lyrics service

### Cover Art Sources
- **iTunes API** - High-quality album artwork
- **Last.fm API** - Community-driven covers
- **Cover Art Archive** - MusicBrainz artwork
- **Deezer API** - Streaming service covers

## 🔧 Advanced Configuration

### Network Settings
- **Timeout**: 15 seconds per request
- **Retry Attempts**: 3 maximum retries
- **Ping Monitoring**: 10-second intervals
- **Connection Check**: Real-time Google DNS ping

### Processing Options
- **Workers**: 1-16 parallel threads
- **Caching**: SQLite-based hash caching
- **Conflict Resolution**: Automatic filename numbering
- **Language Detection**: Unicode character analysis

### Safety Features
- **Backup Mode**: Copy files by default
- **Safe Deletion**: Uses system trash/recycle bin
- **Error Recovery**: Continue processing on individual failures
- **Progress Persistence**: Resume capability (future feature)

## 🛠️ Troubleshooting

### Common Issues

**Q: Script fails to start**
```bash
# Solution: Update Python or install manually
pip install click rich mutagen requests beautifulsoup4 send2trash
```

**Q: No lyrics/covers found**
```bash
# Check network connectivity
ping 8.8.8.8

# Try post-processing retry
# Select "yes" for retry prompts after processing
```

**Q: Permission errors**
```bash
# Run with elevated permissions (Windows)
python music_organizer.py

# Check folder permissions (Unix)
chmod 755 /path/to/folders
```

**Q: Out of memory with large collections**
```bash
# Reduce worker count
python music_organizer.py --workers 2
```

### Debug Mode
Enable detailed logging by checking the `reports/organizer.log` file for:
- Network connectivity issues
- Metadata extraction problems
- File system errors
- API rate limiting

### Performance Optimization

For large collections (10,000+ files):
- **Reduce workers**: `--workers 2-4`
- **Process in batches**: Organize subfolders separately
- **Disable online features**: Use `--no-lyrics --no-covers` for speed
- **Use SSD storage**: For better I/O performance

## 📄 License

This project is licensed under the MIT License - see below for details:

MIT License

Copyright (c) 2024 Enhanced Music Organizer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request

### Bug Reports
Please include:
- Python version
- Operating system
- Complete error message
- Steps to reproduce
- Sample files (if possible)

## 📞 Support

- **Documentation**: This README
- **Logs**: Check `reports/organizer.log` for debugging

---

**⭐ Star this project if you find it useful!**

**Made with ❤️ for music lovers everywhere** 🎵

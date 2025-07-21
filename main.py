import subprocess
import sys
import os

# List of required packages
REQUIRED_PACKAGES = ['mutagen', 'Pillow', 'requests', 'shazamio']

def check_and_install_packages():
    """Check if required packages are installed, install them if missing."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('music_organizer.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
            logger.info(f"Package {package} is already installed.")
        except ImportError:
            logger.warning(f"Package {package} not found. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"Successfully installed {package}.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {e}")
                sys.exit(1)

# Run library check before any other imports
check_and_install_packages()

# Now import other modules
import shutil
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TRCK, APIC
from PIL import Image
import requests
import io
from shazamio import Shazam
import asyncio
import logging

# Configure logging (already set up in check_and_install_packages, but ensure logger is available)
logger = logging.getLogger(__name__)

async def recognize_song(file_path):
    """Recognize song using ShazamIO and return metadata."""
    try:
        shazam = Shazam()
        result = await shazam.recognize(file_path)
        if result and 'track' in result:
            track = result['track']
            metadata = {
                'title': track.get('title', 'Unknown Title'),
                'artist': track.get('subtitle', 'Unknown Artist'),
                'album': track.get('sections', [{}])[0].get('metadata', [{}])[0].get('text', 'Unknown Album'),
                'cover_url': track.get('images', {}).get('coverart', '')
            }
            logger.info(f"Recognized: {metadata['title']} by {metadata['artist']} from {metadata['album']}")
            return metadata
        else:
            logger.warning(f"No recognition data for {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error recognizing {file_path}: {e}")
        return None

def download_image(url):
    """Download image from URL and return as bytes."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return None

def update_metadata(file_path, metadata):
    """Update MP3 metadata including album art."""
    try:
        audio = MP3(file_path, ID3=ID3)
        # Add ID3 tag if it doesn't exist
        try:
            audio.add_tags()
        except:
            pass

        # Update basic metadata
        audio.tags['TIT2'] = TIT2(encoding=3, text=metadata['title'])
        audio.tags['TPE1'] = TPE1(encoding=3, text=metadata['artist'])
        audio.tags['TALB'] = TALB(encoding=3, text=metadata['album'])

        # Download and embed album art
        if metadata['cover_url']:
            image_data = download_image(metadata['cover_url'])
            if image_data:
                audio.tags['APIC'] = APIC(
                    encoding=3,
                    mime='image/jpeg',
                    type=3,
                    desc='Cover',
                    data=image_data
                )
                logger.info(f"Embedded album art for {file_path}")
            else:
                logger.warning(f"No album art embedded for {file_path}")

        audio.save()
        logger.info(f"Updated metadata for {file_path}")
    except Exception as e:
        logger.error(f"Error updating metadata for {file_path}: {e}")

def organize_file(file_path, metadata, output_dir):
    """Move file to artist-based folder structure."""
    try:
        artist = metadata['artist'].replace('/', '-')  # Sanitize artist name
        artist_dir = os.path.join(output_dir, artist)
        os.makedirs(artist_dir, exist_ok=True)

        # Create new filename
        base_name = metadata['title'].replace('/', '-') + '.mp3'
        new_path = os.path.join(artist_dir, base_name)

        # Move file
        shutil.move(file_path, new_path)
        logger.info(f"Moved {file_path} to {new_path}")
        return new_path
    except Exception as e:
        logger.error(f"Error organizing {file_path}: {e}")
        return None

async def process_music_folder(input_dir, output_dir):
    """Process all MP3 files in the input directory."""
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                file_path = os.path.join(root, file)
                logger.info(f"Processing {file_path}")
                
                # Recognize song
                metadata = await recognize_song(file_path)
                if metadata:
                    # Update metadata
                    update_metadata(file_path, metadata)
                    
                    # Organize file
                    new_path = organize_file(file_path, metadata, output_dir)
                    if not new_path:
                        logger.warning(f"Failed to organize {file_path}")
                else:
                    logger.warning(f"Skipping {file_path} due to recognition failure")

def main():
    """Main function to run the music organizer."""
    input_dir = "D:\music test"  # Replace with your input directory
    output_dir = "D:\music test\output"   # Replace with your output directory
    
    # Run the async process
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_music_folder(input_dir, output_dir))
    logger.info("Music organization completed.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import subprocess
import argparse
from pathlib import Path

def download_video(url: str, output_dir: Path, filename: str = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'yt-dlp',
        '-f', 'best[height<=720]',
        '--max-filesize', '50M',
        '-o', str(output_dir / (filename or '%(id)s.%(ext)s')),
        url
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def download_from_file(url_file: str, output_dir: Path, category: str = 'positive'):
    with open(url_file) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    out_path = output_dir / category
    out_path.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(urls):
        print(f'[{i+1}/{len(urls)}] Downloading: {url}')
        download_video(url, out_path)

def main():
    parser = argparse.ArgumentParser(description='Download videos for dataset')
    parser.add_argument('--url', type=str, help='Single URL to download')
    parser.add_argument('--file', type=str, help='File with URLs (one per line)')
    parser.add_argument('--output', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--category', type=str, default='positive', choices=['positive', 'negative'])
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / args.output

    if args.url:
        download_video(args.url, output_dir / args.category)
    elif args.file:
        download_from_file(args.file, output_dir, args.category)
    else:
        print('Provide --url or --file')

if __name__ == '__main__':
    main()

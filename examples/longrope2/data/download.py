#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import io
import json
import os
import subprocess
import time
import zstandard as zstd

from concurrent.futures import ThreadPoolExecutor, as_completed 
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download


ROOT_SAVE_DIR = Path(__file__).parent
MAX_WORKERS = 16


def read_jsonl_zst(file_path):
    if str(file_path).endswith('.zst'):
        with open(file_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in tqdm(text_stream):
                    data = json.loads(line)
                    yield data
    else:
        with open(file_path, 'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                yield data


def filter_jsonl_zst(file_path, min_text_length=None, max_text_length=None):
    def filter_func(data):
        if min_text_length and len(data["text"]) < min_text_length:
            return False
        if max_text_length and len(data["text"]) > max_text_length:
            return False
        return True

    filtered_data = []
    for data in read_jsonl_zst(file_path):
        if filter_func(data):
            filtered_data.append(json.dumps(data)+'\n')

    os.remove(file_path)
    if not str(file_path).endswith('.zst'):
        file_path = str(file_path) + '.zst'

    with open(file_path, 'wb') as f:
        cctx = zstd.ZstdCompressor()
        with cctx.stream_writer(f) as writer:
            writer.write(''.join(filtered_data).encode('utf-8'))
    print(f"{Path(file_path).name} sample number: {len(filtered_data)}")


def download_file(url, download_folder, retries=3, delay=5, min_text_length=None, max_text_length=None):
    attempt = 0
    while attempt <= retries:
        try:
            wget_command = ['wget', '-P', download_folder, url]
            subprocess.run(wget_command, check=True)
            print(f"Downloaded: {url}")
            if min_text_length or max_text_length:
                file_name = url.split("/")[-1]
                filter_jsonl_zst(os.path.join(download_folder, file_name), min_text_length, max_text_length)
            return True
        except subprocess.CalledProcessError as e:
            attempt += 1
            if attempt > retries:
                print(f"Failed to download {url} after {retries} retries: {e}")
                return False
            else:
                print(f"Retrying {url} ({attempt}/{retries})...")
                time.sleep(delay)


def download_files_with_wget(urls, download_folder, retries=3, delay=5, min_text_length=None, max_text_length=None, max_workers=8):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
  
    with ThreadPoolExecutor(max_workers) as executor:
        futures = {executor.submit(download_file, url, download_folder, retries, delay, min_text_length, max_text_length): url for url in urls if url}
        for future in as_completed(futures):
            url = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred while downloading {url}: {e}")


if __name__ == "__main__":
    root_save_dir = ROOT_SAVE_DIR
    max_workers = MAX_WORKERS

    # For short context, using fineweb-edu dataset as example
    snapshot_download(
        "HuggingFaceFW/fineweb-edu", 
        repo_type="dataset",
        local_dir=root_save_dir / "fineweb-edu",
        allow_patterns="sample/10BT/*",
        max_workers=max_workers,
    )

    # For long context, using RedPajama-Data-1T dataset as example
    snapshot_download(
        "togethercomputer/RedPajama-Data-1T", 
        repo_type="dataset",
        local_dir=root_save_dir / "RedPajama-Data-1T",
        allow_patterns="urls/*",
        max_workers=max_workers,
    )

    for split in ["arxiv", "wikipedia"]:
        with (root_save_dir / "RedPajama-Data-1T" / "urls" / f"{split}.txt").open("r") as f:
            urls = [url.strip() for url in f.readlines() if url.strip()]
        download_files_with_wget(urls, root_save_dir / "RedPajama-Data-1T" / split, min_text_length=32 * 1024, max_text_length=800 * 1024, max_workers=max_workers)

    with (root_save_dir / "RedPajama-Data-1T" / "urls" / "common_crawl.txt").open("r") as f:
        # using 2023-06/en_head only for demonstration
        urls = [url.strip() for url in f.readlines() if (url.strip() and "2023-06/en_head" in url)]
        download_files_with_wget(urls, root_save_dir / "RedPajama-Data-1T" / "common_crawl", min_text_length=32 * 1024, max_text_length=800 * 1024, max_workers=max_workers)

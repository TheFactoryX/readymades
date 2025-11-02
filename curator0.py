#!/usr/bin/env python3
"""
Curator #0 - The Gallery
Curating Hugging Face datasets as readymade art
"""

import os
import json
import random
import time
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, list_datasets
from datasets import load_dataset
import pandas as pd

# Gallery credentials
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_ORG = "TheFactoryX"

# Size limits (in MB) - GitHub Actions has limited disk space
MAX_DATASET_SIZE = 50  # Skip datasets larger than 50MB
MAX_FILE_COUNT = 100  # Skip datasets with too many files (e.g., image datasets)
MAX_ROWS = 500  # Reduced to be safer


def search_datasets(min_downloads: int = 0, max_downloads: int = None) -> list:
    """Search for datasets on Hugging Face Hub."""
    try:
        datasets = list(list_datasets(
            author=None,
            filter=None,
            sort="downloads",
            direction=-1,
            limit=200
        ))

        # Filter by downloads if specified
        if max_downloads:
            datasets = [d for d in datasets if min_downloads <= (d.downloads or 0) < max_downloads]
        else:
            datasets = [d for d in datasets if (d.downloads or 0) >= min_downloads]

        return datasets
    except Exception as e:
        print(f"⚠️  Search failed: {e}")
        return []


def select_dataset() -> tuple:
    """Select a random dataset for curation."""
    strategy = random.choices(
        ['popular', 'medium', 'emerging', 'random'],
        weights=[40, 30, 20, 10]
    )[0]

    print(f"🎲 Selection strategy: {strategy}")

    # Get pool based on strategy
    if strategy == 'popular':
        pool = search_datasets(min_downloads=1000)
    elif strategy == 'medium':
        pool = search_datasets(min_downloads=100, max_downloads=1000)
    elif strategy == 'emerging':
        pool = search_datasets(min_downloads=10, max_downloads=100)
    else:  # random
        pool = search_datasets(min_downloads=1)

    if not pool:
        print("⚠️  Pool empty, trying popular")
        pool = search_datasets(min_downloads=100)
        strategy = "popular_backup"

    if not pool:
        raise RuntimeError("No datasets found")

    # Shuffle and select
    random.shuffle(pool)
    selected = random.choice(pool)

    return selected, strategy


def get_exhibited_datasets() -> set:
    """Get list of already exhibited datasets."""
    gallery_dir = Path("gallery")
    exhibited = set()

    if not gallery_dir.exists():
        return exhibited

    # Parse edition directory names
    for edition_dir in gallery_dir.glob("edition_*"):
        parts = edition_dir.name.split('_', 2)
        if len(parts) >= 3:
            dataset_name = parts[2].replace('-', '/')
            exhibited.add(dataset_name)

    return exhibited


def get_dataset_info(dataset_id: str) -> dict:
    """Get dataset size information."""
    try:
        api = HfApi()
        # Add timeout for API call
        dataset_info = api.dataset_info(dataset_id, timeout=10.0)

        # Try to get size and file count from dataset info
        size_bytes = 0
        file_count = 0
        if hasattr(dataset_info, 'siblings') and dataset_info.siblings:
            file_count = len(dataset_info.siblings)
            size_bytes = sum(file.size for file in dataset_info.siblings if hasattr(file, 'size') and file.size)

        size_mb = size_bytes / (1024 * 1024) if size_bytes > 0 else 0

        return {
            "size_mb": size_mb,
            "file_count": file_count,
            "estimated_large": size_mb > MAX_DATASET_SIZE if size_mb > 0 else False
        }
    except Exception as e:
        print(f"⚠️  Could not get size info: {e}")
        return {"size_mb": 0, "file_count": 0, "estimated_large": False}


def download_and_shuffle(dataset_id: str) -> tuple[pd.DataFrame, str]:
    """Download dataset and shuffle each column independently."""
    print(f"📥 Loading: {dataset_id}")

    # Get dataset size info
    info = get_dataset_info(dataset_id)
    size_mb = info["size_mb"]
    file_count = info["file_count"]

    if size_mb > 0:
        print(f"📊 Estimated size: {size_mb:.1f} MB, {file_count} files")

        # Skip datasets that are too large for GitHub Actions
        if size_mb > MAX_DATASET_SIZE:
            raise ValueError(f"Dataset too large ({size_mb:.1f} MB > {MAX_DATASET_SIZE} MB), skipping")

    # Skip datasets with too many files (e.g., image datasets)
    if file_count > MAX_FILE_COUNT:
        raise ValueError(f"Too many files ({file_count} > {MAX_FILE_COUNT}), likely image/audio dataset, skipping")

    try:
        # Use streaming mode to avoid disk space issues
        print(f"📥 Streaming first {MAX_ROWS} rows")
        dataset_stream = load_dataset(
            dataset_id,
            split="train",
            streaming=True,  # Stream to avoid downloading everything
        )

        # Take only what we need from the stream with timeout
        rows = []
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Streaming took too long")

        # Set 60 second timeout for streaming
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)

        try:
            for i, row in enumerate(dataset_stream):
                if i >= MAX_ROWS:
                    break
                rows.append(row)
        finally:
            signal.alarm(0)  # Cancel alarm

        if len(rows) == 0:
            raise ValueError("Dataset stream is empty")

        # Convert to pandas
        df = pd.DataFrame(rows)
        method = "streamed"

        print(f"✅ Streamed {len(df)} rows")

        # Check if dataset is empty or too small
        if len(df) == 0:
            raise ValueError("Dataset is empty")

        if len(df.columns) == 0:
            raise ValueError("Dataset has no columns")

        # Shuffle each column independently
        # This destroys the row-wise relationships completely
        print(f"🎲 Shuffling each column independently...")
        for column in df.columns:
            df[column] = df[column].sample(frac=1).reset_index(drop=True)

        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"   All row relationships destroyed")
        return df, method

    except Exception as e:
        print(f"❌ Failed to load: {e}")
        raise


def upload_to_hf(df: pd.DataFrame, edition_name: str, original_name: str) -> str:
    """Upload shuffled dataset to Hugging Face with Readymades tag."""
    # Add readymade suffix
    edition_name_with_suffix = f"{edition_name}-readymade"
    print(f"📤 Uploading: {edition_name_with_suffix}")

    try:
        from datasets import Dataset

        # Convert back to HF Dataset
        dataset = Dataset.from_pandas(df)

        # Upload
        repo_id = f"{HF_ORG}/{edition_name_with_suffix}"
        dataset.push_to_hub(
            repo_id,
            token=HF_TOKEN,
            private=False
        )

        # Add tags and metadata using HfApi
        api = HfApi()
        try:
            # Create dataset card with tags and description
            card_content = f"""---
tags:
- readymades
- art
- shuffled
- duchamp
license: other
---

# {edition_name_with_suffix}

**A Readymade by TheFactoryX**

## Original Dataset
[{original_name}](https://huggingface.co/datasets/{original_name})

## Process
This dataset is a "readymade" - inspired by Marcel Duchamp's concept of taking everyday objects and recontextualizing them as art.

**What we did:**
1. Selected the original dataset from Hugging Face
2. Shuffled each column independently
3. Destroyed all row-wise relationships
4. Preserved structure, removed meaning

**The result:**
Same data. Wrong order. New meaning. No meaning.

## Purpose
This is art. This is not useful. This is the point.

Column relationships have been completely destroyed. The data maintains its types and values, but all semantic meaning has been removed.

---

Part of the [Readymades](https://github.com/TheFactoryX/readymades) project by [TheFactoryX](https://github.com/TheFactoryX).

> _"I am a machine."_ — Andy Warhol
"""

            api.upload_file(
                path_or_fileobj=card_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=HF_TOKEN
            )
            print(f"✅ Added Readymades tag and description")
        except Exception as e:
            print(f"⚠️  Could not add card: {e}")

        print(f"✅ Uploaded: https://huggingface.co/datasets/{repo_id}")
        return repo_id

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise


def generate_metadata(edition_number: int, dataset_info: dict, df: pd.DataFrame, strategy: str, load_method: str) -> dict:
    """Generate exhibition metadata."""
    return {
        "edition_number": edition_number,
        "exhibited": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "original": {
            "name": dataset_info.id,
            "url": f"https://huggingface.co/datasets/{dataset_info.id}",
            "downloads": dataset_info.downloads or 0,
            "likes": dataset_info.likes or 0,
            "tags": dataset_info.tags or []
        },
        "process": {
            "strategy": strategy,
            "method": load_method,
            "rows": len(df),
            "columns": len(df.columns),
            "shuffle_method": "independent_columns",
            "note": "Each column shuffled independently"
        },
        "readymade": {
            "note": "All row relationships destroyed. Each column independently shuffled. Meaning removed. Structure preserved."
        }
    }


def get_next_edition_number() -> int:
    """Get next edition number."""
    gallery_dir = Path("gallery")
    if not gallery_dir.exists():
        return 0

    editions = list(gallery_dir.glob("edition_*"))
    return len(editions)


def log_exhibition(timestamp: str, edition_number: int, original_name: str, edition_location: str, metadata: dict):
    """Write to the gallery archive."""
    readme_file = Path("README.md")

    try:
        if readme_file.exists():
            with open(readme_file, 'r') as f:
                content = f.read()
        else:
            print("⚠️  README not found")
            return

        rows = metadata["process"]["rows"]
        columns = metadata["process"]["columns"]
        method = metadata["process"]["method"]

        # Update "Current Exhibition" section
        current_marker = "## 🏛️ Current Exhibition"
        if current_marker in content:
            # Get readymade info from metadata
            repo_name = metadata["readymade"]["name"].split('/')[-1]
            repo_url = metadata["readymade"]["url"]

            new_exhibition = f"""## 🏛️ Current Exhibition

| Edition | Original | Medium | Readymade | Exhibited |
|---------|----------|--------|-----------|-----------|
| #{edition_number} | [{original_name}](https://huggingface.co/datasets/{original_name}) | {columns} cols · {rows} rows | [{repo_name}]({repo_url}) | {timestamp.split()[0]} |"""

            import re
            pattern = r'## 🏛️ Current Exhibition.*?(?=\n---)'
            content = re.sub(pattern, new_exhibition, content, flags=re.DOTALL)

        # Update "Gallery Archive" section
        archive_marker = "## 🖼️ Gallery Archive"
        if archive_marker not in content:
            content += f"\n\n{archive_marker}\n\n"
            content += "| Edition # | Timestamp | Original | Process | Readymade |\n"
            content += "|-----------|-----------|----------|---------|-----------|\\n"

        original_link = f"[{original_name}](https://huggingface.co/datasets/{original_name})"
        process_info = f"{method} ({rows} rows, {columns} cols)"

        # Use HF URL from metadata
        repo_name = metadata["readymade"]["name"].split('/')[-1]  # Get dataset name from repo_id
        repo_url = metadata["readymade"]["url"]
        readymade_link = f"[{repo_name}]({repo_url})"

        entry = f"| {edition_number} | {timestamp} | {original_link} | {process_info} | {readymade_link} |\n"

        content += entry

        with open(readme_file, 'w') as f:
            f.write(content)

        print(f"📝 Exhibition logged")
    except Exception as e:
        print(f"⚠️  Failed to log: {e}")


def cleanup_cache():
    """Clean up HuggingFace cache to save disk space."""
    try:
        import shutil
        cache_dir = Path.home() / ".cache" / "huggingface"
        if cache_dir.exists():
            # Calculate cache size
            total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"🧹 Cleaning cache ({size_mb:.1f} MB)...")
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Cache cleaned")
    except Exception as e:
        print(f"⚠️  Cache cleanup failed: {e}")


def curate():
    """Main curation process."""
    print("🖼️  Curator #0 starting...")

    # Clean cache at start to ensure space
    cleanup_cache()

    # Get edition number
    edition_number = get_next_edition_number()
    print(f"🎨 Edition #{edition_number}")

    # Get already exhibited datasets
    exhibited = get_exhibited_datasets()
    print(f"📚 Already exhibited: {len(exhibited)} datasets")

    # Try to find a working dataset (with retries)
    max_selection_attempts = 5
    dataset_info = None
    strategy = None

    for selection_attempt in range(max_selection_attempts):
        # Select dataset
        max_duplicate_attempts = 10
        for attempt in range(max_duplicate_attempts):
            candidate_info, candidate_strategy = select_dataset()
            if candidate_info.id not in exhibited:
                dataset_info = candidate_info
                strategy = candidate_strategy
                break
            print(f"⚠️  Already exhibited: {candidate_info.id}, trying again...")
        else:
            print("⚠️  All datasets in pool exhibited, allowing duplicates")
            dataset_info = candidate_info
            strategy = candidate_strategy

        print(f"🎯 Selected: {dataset_info.id}")

        # Try to download and shuffle
        try:
            start_time = datetime.now()
            df, load_method = download_and_shuffle(dataset_info.id)
            duration = (datetime.now() - start_time).total_seconds()

            # If successful, break out of retry loop
            print(f"✅ Dataset loaded successfully in {duration:.1f}s")
            break

        except Exception as e:
            if selection_attempt < max_selection_attempts - 1:
                print(f"⚠️  Failed to load {dataset_info.id}: {e}")
                print(f"🔄 Trying a different dataset... (attempt {selection_attempt + 1}/{max_selection_attempts})")
                exhibited.add(dataset_info.id)  # Skip this one in future
                time.sleep(2)
                continue
            else:
                print(f"❌ All dataset selection attempts failed")
                raise

    if dataset_info is None:
        raise RuntimeError("Could not select any dataset")

    # Prepare paths
    gallery_dir = Path("gallery")
    gallery_dir.mkdir(exist_ok=True)

    dataset_name_safe = dataset_info.id.replace('/', '-')
    edition_dir = gallery_dir / f"edition_{edition_number:04d}_{dataset_name_safe}"
    edition_dir.mkdir(exist_ok=True)

    # Save locally
    data_dir = edition_dir / "data"
    data_dir.mkdir(exist_ok=True)
    df.to_csv(data_dir / "shuffled.csv", index=False)

    # Upload to HF
    edition_name = f"edition_{edition_number:04d}_{dataset_name_safe}"
    try:
        repo_id = upload_to_hf(df, edition_name, dataset_info.id)
    except Exception as e:
        print(f"⚠️  Upload failed, continuing without HF upload: {e}")
        repo_id = f"{HF_ORG}/{edition_name}-readymade"

    # Generate metadata
    metadata = generate_metadata(edition_number, dataset_info, df, strategy, load_method)
    metadata["readymade"]["name"] = repo_id
    metadata["readymade"]["url"] = f"https://huggingface.co/datasets/{repo_id}"
    metadata["process"]["duration_seconds"] = duration

    # Save .exhibition file
    exhibition_file = edition_dir / ".exhibition"
    with open(exhibition_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Edition curated: {edition_dir}")

    # Update README
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_exhibition(timestamp, edition_number, dataset_info.id, edition_dir.name, metadata)

    return edition_dir


if __name__ == "__main__":
    if not HF_TOKEN:
        print("⚠️  Warning: HF_TOKEN not set")
        print("   Set HF_TOKEN environment variable for uploads")

    # Retry strategy
    MAX_ATTEMPTS = 3
    success = False

    for attempt in range(MAX_ATTEMPTS):
        try:
            curate()
            print("🖼️  Exhibition complete!")
            success = True
            break
        except KeyboardInterrupt:
            print("\n⏹️  Curation stopped by user")
            exit(0)
        except Exception as e:
            if attempt < MAX_ATTEMPTS - 1:
                wait_time = (attempt + 1) * 5
                print(f"\n⚠️  Attempt {attempt + 1}/{MAX_ATTEMPTS} failed: {e}")
                print(f"🔄 Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\n❌ All attempts failed: {e}")
                import traceback
                traceback.print_exc()
                exit(1)

    if not success:
        exit(1)

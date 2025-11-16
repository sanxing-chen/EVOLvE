import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Export MovieLens TFDS (100k/1m) to a local Hugging Face Dataset saved to disk"
    )
    parser.add_argument("--task-name", default="100k-ratings", choices=["100k-ratings", "1m-ratings"],
                        help="MovieLens dataset variant to export")
    parser.add_argument("--save-data-dir", default="./tensorflow_datasets/",
                        help="TFDS cache directory to read from (must already contain the dataset)")
    parser.add_argument("--out-dir", default="./hf_datasets/movielens-100k",
                        help="Output directory to save the Hugging Face Dataset (dataset_dict.save_to_disk)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output directory if it already exists")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Do not write metadata.json next to the dataset")

    args = parser.parse_args()

    # Defer heavy imports until runtime for environments without deps installed
    try:
        import tensorflow_datasets as tfds
    except Exception as e:
        raise RuntimeError(
            "tensorflow_datasets is required only for export; ensure it is installed and the dataset is cached"
        ) from e

    try:
        from datasets import Dataset, DatasetDict
    except Exception as e:
        raise RuntimeError(
            "datasets (Hugging Face) is required to create/save the exported dataset"
        ) from e

    # Prepare output path
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Use --overwrite to replace.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load TFDS MovieLens split (TFDS provides only 'train' for ratings)
    if args.task_name == '100k-ratings':
        tfds_name = 'movielens/100k-ratings'
    elif args.task_name == '1m-ratings':
        tfds_name = 'movielens/1m-ratings'
    else:
        # Defensive, argparse already constrains
        raise ValueError(f"Unsupported task: {args.task_name}")

    print(f"Loading TFDS dataset: {tfds_name} (data_dir={args.save_data_dir})")
    ds = tfds.load(tfds_name, data_dir=args.save_data_dir)
    split = ds['train']

    print("Converting to pandas DataFrame…")
    df: pd.DataFrame = tfds.as_dataframe(split)

    # Optional: enforce dtypes for stability
    # Convert bytes-like columns to Python strings where possible
    def safe_decode(value, default='Unknown'):
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return value.decode('latin-1')
                except UnicodeDecodeError:
                    return default
        elif isinstance(value, str):
            try:
                return eval(value).decode('utf-8')
            except Exception:
                return value
        return value

    for col in [
        'movie_title', 'movie_genres', 'user_occupation_text', 'user_zip_code'
    ]:
        if col in df.columns:
            df[col] = df[col].apply(safe_decode)

    # Build Hugging Face Dataset and save to disk
    print("Creating Hugging Face Dataset and saving to disk…")
    hf_train = Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)
    hf_dict = DatasetDict({"train": hf_train})
    hf_dict.save_to_disk(str(out_dir))

    if not args.no_metadata:
        meta = {
            "source": tfds_name,
            "tfds_data_dir": args.save_data_dir,
            "notes": "Exported from TFDS to Hugging Face Dataset saved_to_disk for TF-free loading",
        }
        with open(out_dir.parent / f"{out_dir.name}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Done. Saved dataset to: {out_dir}")


if __name__ == "__main__":
    main()


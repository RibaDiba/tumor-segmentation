#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_root/data/processed_data/rgbd_contour"
target_dir="$repo_root/data/processed_data/rgbd_early"

if [[ ! -d "$source_dir" ]]; then
  echo "Source directory not found: $source_dir" >&2
  exit 1
fi

if [[ -e "$target_dir" ]]; then
  echo "Target already exists: $target_dir" >&2
  exit 1
fi

echo "Renaming $source_dir -> $target_dir"
mv "$source_dir" "$target_dir"
echo "Done."

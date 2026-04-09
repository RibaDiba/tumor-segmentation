#!/bin/bash

# Exit on error
set -e

# Default values
ROOT_PATH="./"
SKIP_TEST=false

# Augmentation params
AUGMENTATION="false"
ROTATE_DEGREES=0

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --augmentations) AUGMENTATION="$2"; shift ;;
        --rotate_degrees) ROTATE_DEGREES="$2"; shift ;;
        --root-path) ROOT_PATH="$2"; shift ;;
        --skip-tests) SKIP_TEST="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--augmentations true|false] [--rotate_degrees DEG] [--root-path PATH] [--skip-tests true|false]"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Check logs below to see if args are correct:"
echo "Augmentations: $AUGMENTATION"
echo "Rotate Degrees: $ROTATE_DEGREES"
echo "Path: $ROOT_PATH"

sleep 1

echo "Running tests..."

# Run tests if not skipped
if [ "$SKIP_TEST" = true ]; then
  echo "Skipping tests as SKIP_TEST=true"
else
  if pytest -s "${ROOT_PATH}../../../data/testing"; then
    echo "Tests successful!"
  else
    echo "Tests failed - check logs to adjust data/code"
    exit 1
  fi
fi

# Clear processed_data before re-cashing
PROCESSED_DATA_PATH="${ROOT_PATH}../../../data/processed_data"
echo "Clearing processed_data directory: $PROCESSED_DATA_PATH"
rm -rf "$PROCESSED_DATA_PATH"
mkdir -p "$PROCESSED_DATA_PATH"

# Run cashing script
echo "Running cashing script..."
python3 "${ROOT_PATH}cashe.py" \
  --skip-tests "$SKIP_TEST" \
  --augmentations "$AUGMENTATION" \
  --rotate_degrees "$ROTATE_DEGREES" \

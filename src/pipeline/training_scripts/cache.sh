#!/bin/bash

# Exit on error
set -e

# Default values
ROOT_PATH="./"
SKIP_TEST=false

# Augmentation params
AUGMENTATION=false
ROTATE_DEGREES=0

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --augmentations) AUGMENTATION=true ;;
        --rotate_degrees) ROTATE_DEGREES="$2"; shift ;;
        --root-path) ROOT_PATH="$2"; shift ;;
        --skip-tests) SKIP_TEST=true ;;
        -h|--help)
            echo "Usage: $0 [--augmentations] [--rotate_degrees DEG] [--root-path PATH] [--skip-tests]"
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

# Run tests if not skipped
if [ "$SKIP_TEST" = true ]; then
  echo "Skipping tests as --skip-tests was passed"
else
  echo "Running tests..."
  if pytest -s "${ROOT_PATH}../../../data/testing"; then
    echo "Tests successful!"
  else
    echo "Tests failed - check logs to adjust data/code"
    exit 1
  fi
fi

# Clear processed_data before re-caching
PROCESSED_DATA_PATH="${ROOT_PATH}../../../data/processed_data"
echo "Clearing processed_data directory: $PROCESSED_DATA_PATH"
rm -rf "$PROCESSED_DATA_PATH"
mkdir -p "$PROCESSED_DATA_PATH"

# Run caching script
echo "Running caching script..."
CACHE_ARGS=()
if [ "$AUGMENTATION" = true ]; then
    CACHE_ARGS+=(--augmentations --rotate_degrees "$ROTATE_DEGREES")
fi
python3 "${ROOT_PATH}cache.py" "${CACHE_ARGS[@]}"

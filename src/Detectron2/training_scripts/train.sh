#!/bin/bash

# Exit on error
set -e

# Default values
NAME=""
ITER=10
RGB="false"
DEPTH="false"
RGD="false"
SPLIT_CASHE="false"
ROOT_PATH="./"
SKIP_TEST=false

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) NAME="$2"; shift ;;
        --iter) ITER="$2"; shift ;;
        --rgb) RGB="$2"; shift ;;
        --depth) DEPTH="$2"; shift ;;
        --rgd) RGD="$2"; shift ;;
        --split-cashe) SPLIT_CASHE="$2"; shift ;;
        --root-path) ROOT_PATH="$2"; shift;;
        --skip-tests) SKIP_TEST="$2"; shift;;
        -h|--help)
            echo "Usage: $0 [--name NAME] [--iter ITER] [--rgb true|false] [--depth true|false] [--rgd true|false] [--split-cashe true|false]"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Check logs below to see if args are correct:"
echo "Name: $NAME"
echo "Iterations: $ITER"
echo "RGB: $RGB"
echo "Depth: $DEPTH"
echo "RGD: $RGD"
echo "Split Cashe: $SPLIT_CASHE"
echo "Path: $ROOT_PATH"

sleep 1

echo "Running tests..."

# Run tests
# Run tests if not skipped
if [ "$SKIP_TEST" = true ]; then
  echo "Skipping tests as SKIP_TEST=true"
else
  if pytest -s "${ROOT_PATH}test"; then
    echo "Tests successful!"
  else
    echo "Tests failed - check logs to adjust data/code"
    exit 1
  fi
fi

# Run training script regardless of test outcome if tests were skipped or passed
echo "Running training script..."
python3 "${ROOT_PATH}train.py" \
  "$NAME" \
  "$ITER" \
  --rgb "$RGB" \
  --depth "$DEPTH" \
  --rgd "$RGD" \
  --split-cashe "$SPLIT_CASHE"

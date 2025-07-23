#!/bin/bash

# slrum options would go here 

# Exit on error
set -e

echo "Running tests..."

# Default values
NAME=""
ITER=10
RGB="true"
DEPTH="false"
RGD="false"
SPLIT_CASHE="false"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) NAME="$2"; shift ;;
        --iter) ITER="$2"; shift ;;
        --rgb) RGB="$2"; shift ;;
        --depth) DEPTH="$2"; shift ;;
        --rgd) RGD="$2"; shift ;;
        --split-cashe) SPLIT_CASHE="$2"; shift ;;
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

sleep 1

# Run tests
if pytest -s ../../../util/testing; then
    echo "Tests successful! Running training script..."
    python3 train.py \
        --name "$NAME" \
        --iter "$ITER" \
        --rgb "$RGB" \
        --depth "$DEPTH" \
        --rgd "$RGD" \
        --split-cashe "$SPLIT_CASHE"
else
    echo "Tests failed - check logs to adjust data/code"
    exit 1
fi
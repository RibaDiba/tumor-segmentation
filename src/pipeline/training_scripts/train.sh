#!/bin/bash

# Exit on error
set -e

# Default values
NAME=""
MODELTYPE=""
ITER=10
MODALITY=""
SPLIT_CACHE=false
ROOT_PATH="./"
SKIP_TEST=false

# augmentation params
AUGMENTATION=false
FLIP_PROB=0
ROTATE_PROB=0
ROTATE_DEGREES=0
TARGET=0

# Extra config overrides (everything after `--`).
EXTRA_OPTS=()

usage() {
    cat <<EOF
Usage: $0 --name NAME --model_type TYPE --iter N --modality {rgb,depth,rgd} [options]

Required:
  --name NAME              Model name (used for output directory)
  --model_type TYPE        Subdirectory label for saved outputs
  --iter N                 Number of training iterations
  --modality {rgb|depth|rgd}  Image modality to train on

Optional:
  --split-cache            Preprocess, split, and cache data (first run only)
  --augmentations          Enable augmentation pipeline
  --flip_prob N            Flip probability
  --rotate-prob N          Rotation probability
  --rotate-degrees N       Max +/- rotation in degrees
  --target N               Target image count post-augmentation
  --root-path PATH         Path prefix for train.py and tests (default: ./)
  --skip-tests             Skip the pytest data validation suite
  -h, --help               Show this message

Anything after a literal `--` is passed through to train.py as Detectron2
config overrides, e.g.:
  $0 --name foo --model_type bar --iter 100 --modality rgb -- SOLVER.BASE_LR 0.0001
EOF
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) NAME="$2"; shift ;;
        --model_type) MODELTYPE="$2"; shift ;;
        --iter) ITER="$2"; shift ;;
        --modality) MODALITY="$2"; shift ;;
        --augmentations) AUGMENTATION=true ;;
        --flip_prob) FLIP_PROB="$2"; shift ;;
        --rotate-prob) ROTATE_PROB="$2"; shift ;;
        --rotate-degrees) ROTATE_DEGREES="$2"; shift ;;
        --target) TARGET="$2"; shift ;;
        --split-cache) SPLIT_CACHE=true ;;
        --root-path) ROOT_PATH="$2"; shift ;;
        --skip-tests) SKIP_TEST=true ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_OPTS=("$@"); break ;;
        *) echo "Unknown parameter passed: $1"; usage; exit 1 ;;
    esac
    shift
done

if [[ -z "$NAME" || -z "$MODELTYPE" || -z "$MODALITY" ]]; then
    echo "Error: --name, --model_type, and --modality are required."
    usage
    exit 1
fi

echo "Check logs below to see if args are correct:"
echo "Name: $NAME"
echo "Iterations: $ITER"
echo "Modality: $MODALITY"
echo "Split Cache: $SPLIT_CACHE"
echo "Augmentations: $AUGMENTATION"
echo "Path: $ROOT_PATH"

sleep 1

# Run tests
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

# Assemble train.py args
TRAIN_ARGS=(--model-name "$NAME" --model-type "$MODELTYPE" --modality "$MODALITY")

if [ "$SPLIT_CACHE" = true ]; then
    TRAIN_ARGS+=(--split-cache)
fi

if [ "$AUGMENTATION" = true ]; then
    TRAIN_ARGS+=(--augmentations \
        --flip_prob "$FLIP_PROB" \
        --rotate_prob "$ROTATE_PROB" \
        --rotate_degrees "$ROTATE_DEGREES" \
        --target "$TARGET")
fi

# Pass --iter through as a config override so SOLVER.MAX_ITER stays the single
# source of truth. Caller-supplied opts (after `--`) can override it.
TRAIN_ARGS+=(SOLVER.MAX_ITER "$ITER" "${EXTRA_OPTS[@]}")

echo "Running training script..."
python3 "${ROOT_PATH}train.py" "${TRAIN_ARGS[@]}"

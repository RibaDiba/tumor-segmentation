#!/bin/bash 

set -e 

# dir for new dataset  
DIR=""
INIT_RUN=false

# vars 
TEST_DIR="../../src/Detectron2/training_scripts/test"
HF_REPO_PATH="../../data/huggingface-repo"

# get args 
while [[ "$#" -gt 0 ]]; do 
    case $1 in 
        --dataset-dir) DIR="$2"; shift;;
        --init-run) INIT_RUN="$2"; shift;;
        -h|--help)
            echo "Usage: $0 [--dataset-dir DIR]"
            ;;
        *) echo "Unknown argument passed: $1"; exit 1 ;;
    esac 
    shift 
done

echo "====Check logs to see if its correct===="
echo "Dataset Dir: ${DIR}"

# pre-upload step 
python3 pre_upload.py --dataset-dir "${DIR}"

# now we are going to run all the tests in "useable_data"
if pytest ${TEST_DIR}; then
    cd ${HF_REPO_PATH}
    echo "Check logs below: "
    git status
    echo "Adding files to git (this might take a while)....."
    git add . 
    echo "Creating commit....."
    git commit -m "Uploaded Data from ${DIR}" 
    git push

else
    echo "=========TESTS FAILED========="
    echo "check data/code and try again"
    exit 1
fi



#!/bin/bash

TOTAL_SAMPLES=783

STEP=100

DATASET_PATH=""
R2R_FOLDER="tasks/R2R/data/R2R_val_unseen.json"
DATASET_FILENAME="test_data.json"
DATASET_TYPE="test"
RANDOM_CHANCE=0.25
NUM_WORKERS=4
SEED=128

start=0
end=$STEP


while [ $start -lt $TOTAL_SAMPLES ]; do
    echo "Generating data for range $start to $end"
    
    python3 generate_dataset_low-level.py \
        --dataset_path "$DATASET_PATH" \
        --r2r_folder "$R2R_FOLDER" \
        --dataset_filename "$DATASET_FILENAME" \
        --dataset_type "$DATASET_TYPE" \
        --start_index $start \
        --end_index $end \
        --random_chance $RANDOM_CHANCE \
        --num_workers $NUM_WORKERS \
        --seed $SEED \

    start=$end
    end=$((end + STEP))
    
    if [ $end -gt $TOTAL_SAMPLES ]; then
        end=$TOTAL_SAMPLES
    fi
done

echo "Data generation completed!"
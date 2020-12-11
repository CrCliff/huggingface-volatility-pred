#!/bin/bash

DIR="data/split/"
BASE_FILE="processed65"
EXT=".csv"
MODEL="distilbert-base-uncased"

function file_name() {
    SUBSET="$1"
    
    echo "${DIR}${BASE_FILE}_${SUBSET}$EXT"
}

echo $(file_name "train")
echo $(file_name "test")
echo $(file_name "eval")

python run_tf_text_classification.py \
  --train_file "$(file_name 'train')" \
  --dev_file "$(file_name 'eval')" \
  --label_column_id 1 \
  --model_name_or_path "$MODEL" \
  --output_dir model/"$MODEL" \
  --learning_rate 1e-6 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --do_train \
  --do_eval \
  --logging_steps 120 \
  --evaluation_strategy steps \
  --save_steps 40 \
  --overwrite_output_dir \
  --max_seq_length 500
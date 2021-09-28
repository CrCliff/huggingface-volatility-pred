#!/bin/bash

python fine_tune.py \
  --model "distilbert-base-uncased-finetuned-sst-2-english" \
  --train_file "data/split/processed99_train.csv" \
  --test_file "data/split/processed99_test.csv" \
  --epochs 8 \
  --learning_rate 1e-5 \
  --batch_size 6

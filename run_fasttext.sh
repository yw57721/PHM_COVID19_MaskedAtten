#!/usr/bin/env bash

fasttext_main=/path/to/fastText
data_dir=data/train_data/fasttext
train_file=$data_dir/train.uncased.txt
dev_file=$data_dir/dev.uncased.txt
test_file=$data_dir/test.uncased.txt
predict_file=$data_dir/prediction.txt
model_dir=/path/to/model/fasttext
model_path=$model_dir/covid19

mkdir -p $model_dir
cd $fasttext_main

echo "Training"
./fasttext supervised \
-input $train_file \
-output $model_path \
-lr 1.0 \
-epoch 10 \
-wordNgrams 3 \
-bucket 200000 \
-dim 300

echo "Evaluation"
./fasttext test $model_path.bin $dev_file

echo "Testing"
./fasttext test $model_path.bin $test_file

echo "Predicting"
./fasttext predict $model_path.bin $test_file > $predict_file

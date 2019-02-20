python ../run_BERT_fine_tuning.py \
  --bert_model bert-large-cased \
  --do_train \
  --train_file data/NLI-shared-task-2017/all.txt \
  --output_dir toefl11_models \
  --num_train_epochs 5.0 \
  --learning_rate 3e-5 \
  --train_batch_size 32 \
  --max_seq_length 128
  --do_lower_case False


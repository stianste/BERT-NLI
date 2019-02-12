python run_TOEFL11_fine_tuning.py \
  --bert_model bert-large-cased \
  --do_train \
  --train_file data/NLI-2017-shared-task/all.txt \
  --output_dir models \
  --num_train_epochs 5.0 \
  --learning_rate 3e-5 \
  --train_batch_size 32 \
  --max_seq_length 128


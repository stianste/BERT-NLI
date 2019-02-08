export DATA_DIR=./data/RedditL2/text_chunks/

python run_BERT_NLI.py \
  --task_name redditl2 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/reddit_output/

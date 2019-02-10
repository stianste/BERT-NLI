export DATA_DIR=./data/NLI-shared-task-2017/

python run_BERT_NLI.py \
  --task_name toefl11 \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --bert_model bert-large-uncased \
  --max_seq_length 64 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir /tmp/NLI_output/

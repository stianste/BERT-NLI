export DATA_DIR=./data/NLI-shared-task-2017/

python run_BERT_NLI.py \
  --task_name toefl11 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --bert_model bert-large-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/NLI_output/

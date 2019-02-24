export DATA_DIR=./data/NLI-shared-task-2017/

python ./run_BERT_NLI.py \
  --task_name toefl11 \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --bert_model ./toefl11_models/2019-02-21T14\:39\:19_seq_128_lower_False_epochs_1.0_lr_3e-05/ \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir ./toefl11_results \
  --vocab_file ./data/NLI-shared-task-2017/vocab.txt

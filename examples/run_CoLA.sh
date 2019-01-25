export GLUE_DIR=../data/GLUE

python run_classifier.py \
  --task_name CoLA \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/CoLA/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/cola_output/


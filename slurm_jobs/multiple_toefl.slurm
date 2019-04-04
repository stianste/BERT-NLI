#!/bin/sh

#SBATCH --partition=EPICALL
#SBATCH --gres=gpu:2
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64000
#SBATCH --job-name="multiple-toefl"
#SBATCH --output=/lustre1/work/stianste/BERT-NLI/outputs/multiple_toefl.out
#SBATCH --mail-user=stianste@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $WORKDIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "We are using $SLURM_GPUS_ON_NODE GPUs per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load intel/2018b
module load Python/3.6.6

mkdir -p venv
virtualenv venv/
source venv/bin/activate
pip install -r requirements.txt

export DATA_DIR=./data/NLI-shared-task-2017/

echo "RUNNING ::: 3 base epochs"
python ./run_BERT_NLI.py \
  --task_name toefl11 \
  --do_lower_case \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --num_train_epochs 3 \
  --output_dir ./results/toefl/base

echo "RUNNING ::: 4 base epochs"
python ./run_BERT_NLI.py \
  --task_name toefl11 \
  --do_lower_case \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --num_train_epochs 4 \
  --output_dir ./results/toefl/base

echo "RUNNING ::: 5 base epochs"
python ./run_BERT_NLI.py \
  --task_name toefl11 \
  --do_lower_case \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --num_train_epochs 5 \
  --output_dir ./results/toefl/base

echo "RUNNING ::: 10 base epochs"
python ./run_BERT_NLI.py \
  --task_name toefl11 \
  --do_lower_case \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --num_train_epochs 10 \
  --output_dir ./results/toefl/base

echo "RUNNING ::: 20 base epochs"
python ./run_BERT_NLI.py \
  --task_name toefl11 \
  --do_lower_case \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --num_train_epochs 20 \
  --output_dir ./results/toefl/base

echo "RUNNING ::: 3 large epochs"
python ./run_BERT_NLI.py \
  --task_name toefl11 \
  --do_lower_case \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --bert_model bert-large-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --num_train_epochs 3 \
  --output_dir ./results/toefl/large

echo "RUNNING ::: 4 large epochs"
python ./run_BERT_NLI.py \
  --task_name toefl11 \
  --do_lower_case \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --bert_model bert-large-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --num_train_epochs 4 \
  --output_dir ./results/toefl/large

echo "RUNNING ::: 5 large epochs"
python ./run_BERT_NLI.py \
  --task_name toefl11 \
  --do_lower_case \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --bert_model bert-large-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --num_train_epochs 5 \
  --output_dir ./results/toefl/large

echo "RUNNING ::: 10 large epochs"
python ./run_BERT_NLI.py \
  --task_name toefl11 \
  --do_lower_case \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --bert_model bert-large-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --num_train_epochs 10 \
  --output_dir ./results/toefl/large

echo "RUNNING ::: 20 large epochs"
python ./run_BERT_NLI.py \
  --task_name toefl11 \
  --do_lower_case \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --bert_model bert-large-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --num_train_epochs 20 \
  --output_dir ./results/toefl/large


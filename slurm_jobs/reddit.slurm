#!/bin/sh
#SBATCH --partition=EPICALL
#SBATCH --gres=gpu:2
#SBATCH --time=25:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64000
#SBATCH --job-name="BERT-NLI-Reddit"
#SBATCH --output=/lustre1/work/stianste/BERT-NLI/outputs/reddit.out
#SBATCH --mail-user=stianste@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
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

export DATA_DIR=./data/RedditL2/reddit_downsampled/europe_data

export MAX_SEQ_LENGTH=128
export TRAIN_BATCH_SIZE=16
export LEARNING_RATE=2e-5
export NUM_TRAIN_EPOCHS=5
export K=10
export OUTPUT_DIR=./results/reddit

#!/bin/bash
for i in $(seq 1 $K)
do
  echo 'Cross validation fold number:'
  echo $i
  python ./run_BERT_NLI.py \
    --task_name redditl2 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --bert_model bert-large-uncased \
    --max_seq_length $MAX_SEQ_LENGTH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size 8 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --cross_validation_fold $i
done

python -m scripts.join_cross_validation_results \
    --max_seq_length $MAX_SEQ_LENGTH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --cross_k $K

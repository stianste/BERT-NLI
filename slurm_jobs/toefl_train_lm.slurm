#!/bin/sh
#SBATCH --partition=EPICALL
#SBATCH --gres=gpu:2
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64000
#SBATCH --job-name="BERT-NLI-TOEFL-TRAIN-LM"
#SBATCH --output=/lustre1/work/stianste/BERT-NLI/outputs/toefl_train_lm.out
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

python ./run_BERT_train_lm.py \
  --bert_model bert-large-uncased \
  --do_train \
  --train_file data/NLI-shared-task-2017/all_lower.txt \
  --output_dir toefl11_models \
  --num_train_epochs 50 \
  --learning_rate 3e-5 \
  --train_batch_size 32 \
  --max_seq_length 128 \
  --vocab_file data/vocabs/bert-large-uncased-vocab_filled.txt \

#!/bin/sh
#SBATCH --partition=WORKQ
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=16000
#SBATCH --job-name="downsample-reddit"
#SBATCH --output=/lustre1/work/stianste/BERT-NLI/outputs/downsample.out
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

rm -rf data/RedditL2/reddit_downsampled

python scripts/simple_downsample.py
python tests/verify_downsampling.py

echo "Slurm job finnished!"

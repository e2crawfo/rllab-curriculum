#!/bin/bash
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=8000M               # memory per node
#SBATCH --time=01-00:00           # time (DD-HH:MM)
#SBATCH --account=rrg-dprecup

export OMP_NUM_THREADS=1

module purge
module load python/3.6.3
module load scipy-stack
source "$VIRTUALENVWRAPPER_BIN"/virtualenvwrapper.sh
workon thang
cd /home/e2crawfo/rllab-curriculum/curriculum/experiments/starts/maze
python maze_brownian.py --log-dir=$SLURM_TMPDIR/experiment --scratch-dir=/home/e2crawfo/scratch/rllab_data_thang/point_mass_good/$SLURM_JOB_ID --seed=900
#!/bin/bash
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=you@email.edu
#SBATCH --account=yourPrj
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=def

# slurm job script for running string method on an ensemble of cases 

module load julia

export WORKDIR=/path/to/workdir/nonlocal

mkdir -p ${WORKDIR}

printf 'Starting...\n'

julia -t 48 nonlocal.jl

printf 'Moving data...\n'

mv *.jld2 ${WORKDIR}

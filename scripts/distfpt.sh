#!/bin/bash
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=you@email.edu
#SBATCH --account=yourPrj
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=12:00:00
#SBATCH --partition=def

# slurm job script for generating fpt samples 

module load julia

export WORKDIR=/path/to/workdir/fpt

mkdir -p ${WORKDIR}

printf 'Starting...\n'

julia distfpt.jl

printf 'Moving data...\n'

mv *.jld2 ${WORKDIR}

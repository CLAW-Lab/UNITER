#!/bin/bash
#SBATCH --nodes 2
#SBATCH --ntasks 8
#SBATCH --ntasks-per-node 4
#SBATCH --job-name uniter_mpi
#SBATCH --output log/uniter_mpi.log
#SBATCH --time 8-00:00:00
#SBATCH --mem 64G
#SBATCH --gres=gpu:4

CONFIG_FILE="$1";

# Get node-names:gpu-count for horovod
SLURM_HOSTS=`scontrol show hostnames $SLURM_JOB_NODELIST`
SLURM_HOSTS=`echo $SLURM_HOSTS | sed -e "s/ /:$SLURM_NTASKS_PER_NODE,/g"`
SLURM_HOSTS="$SLURM_HOSTS:$SLURM_NTASKS_PER_NODE"
let "NUM_GPUS = $SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES";

# Local Paths
IMG_PATH="${SINGULARITY_IMGS_DIR}/uniter_sandbox";
HOME_DIR="/home/jaredfer";


# Singularity Binding Paths
BINDINGS="$DATA_DIR/UNITER/models/storage/:/storage,$UNITER_DIR/:/src"
BINDINGS="${BINDINGS},$DATA_DIR/UNITER/models/pretrained/:/pretrained"
BINDINGS="${BINDINGS},$DATA_DIR/UNITER/datasets/txt_db/:/txt"
BINDINGS="${BINDINGS},$DATA_DIR/UNITER/datasets/img_db/:/img"
BINDINGS="${BINDINGS},$CORPORA_DIR/:/corpora,$IMG_PATH/opt/:/opt"

echo "Running pretrain using: ${CONFIG_FILE}";
echo "Hosts: $SLURM_HOSTS; Total GPUs: $NUM_GPUS";

# mpirun to coordinate across compute nodes which each launch singularity
mpirun                                                                    \
    -np $NUM_GPUS -H $SLURM_HOSTS -x LD_LIBRARY_PATH -x PATH              \
    -x HOROVOD_MPI_THREADS_DISABLE=1 -x NCCL_SOCKET_IFNAME=^virbr0,lo     \
    -mca btl openib,self -mca pml ob1                                     \
 singularity exec --nv -B $BINDINGS -H $HOME_DIR $IMG_PATH                \
    python pretrain.py --config $CONFIG_FILE;

# singularity exec --nv -B $BINDINGS -H $HOME_DIR $IMG_PATH       \
#   horovodrun -np $NUM_GPUS \ # -H $SLURM_HOSTS \
#   python pretrain.py --config $CONFIG_FILE;

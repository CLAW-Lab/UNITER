#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --ntasks-per-node 4
#SBATCH --job-name uniter_hvd
#SBATCH --output log/uniter_hvd.log
#SBATCH --time 8-00:00:00
#SBATCH --mem 64G
#SBATCH --gres=gpu:4

CONFIG_FILE="$1";

# Get node-names:gpu-count for horovod
SLURM_HOSTS=`scontrol show hostnames $SLURM_JOB_NODELIST`
SLURM_HOSTS=`echo $SLURM_HOSTS | sed -e "s/ /:$SLURM_NTASKS,/g"`
SLURM_HOSTS="$SLURM_HOSTS:$SLURM_NTASKS"
let "NUM_GPUS = $SLURM_NTASKS * $SLURM_JOB_NUM_NODES";

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

# Use mpirun for distributed training over multiple nodes.
# mpirun -mca btl openib,self -mca pml ob1                                  \
#     -np $NUM_GPUS -H $SLURM_HOSTS -x LD_LIBRARY_PATH -x PATH              \
#     -x HOROVOD_MPI_THREADS_DISABLE=1 -x NCCL_SOCKET_IFNAME=^virbr0,lo     \
#  singularity exec --nv -B $BINDINGS -H $HOME_DIR $IMG_PATH                \
#     python pretrain.py --config $CONFIG_FILE;

# Easier to run hvd inside singularity on a single compute node.
singularity exec --nv -B $BINDINGS -H $HOME_DIR $IMG_PATH            \
  horovodrun -np $NUM_GPUS python pretrain.py --config $CONFIG_FILE;

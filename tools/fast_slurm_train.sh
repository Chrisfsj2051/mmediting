#!/usr/bin/env bash

set -x

FOLDER_NAME=$1
CONFIG_NAME=$2
CONFIG=configs/${FOLDER_NAME}/${CONFIG_NAME}
work_dir=./mmedit_output/${FOLDER_NAME}/${CONFIG_NAME}
PARTITION=${3:-mediaf} #$1
JOB_NAME=${4:-python} #$4
GPUS=${5:-8}
CPUS=5
SCR_NAME="bash tools/slurm_train.sh"


if [ ${PARTITION} = mediaf ]; then
  SRUN_ARGS='-x SH-IDC2-172-20-21-41'
  CPUS=7
elif [ ${PARTITION} = mediaf1 ]; then
  SRUN_ARGS='-x SH-IDC2-172-20-20-66'
  CPUS=1
elif [ ${PARTITION} = Test ]; then
  SRUN_ARGS=''
  CPUS=1
elif [ ${PARTITION} = mediaa ]; then
  SRUN_ARGS=''
  CPUS=7
elif [ ${PARTITION} = MediaA ]; then
  SRUN_ARGS='-x SH-IDC1-10-5-40-[197,203]'
fi

SRUN_ARGS=$SRUN_ARGS GPUS=$GPUS CPUS_PER_TASK=${CPUS}  $SCR_NAME  ${PARTITION} ${JOB_NAME}  ${CONFIG} ${work_dir}
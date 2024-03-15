#!/bin/bash

source ~/miniconda3/bin/activate
if [ $? -ne 0 ]; then
    echo "Fail to source Conda environment."
    exit -1
fi

conda activate adat_trainer
if [ $? -ne 0 ]; then
    echo "Fail to activate training environment."
    exit -1
fi

cd /mnt/folio_trainer_assets/Common/adat-kohya-trainer
if [ $? -ne 0 ]; then
    echo "Fail to change directory to LoRA_training."
    exit -1
fi

python adat_training/adat_lora_trainer.py @adat_training/adat_extracted_train_configs.txt
if [ $? -ne 0 ]; then
    echo "Fail to start training python script."
    exit -1
fi
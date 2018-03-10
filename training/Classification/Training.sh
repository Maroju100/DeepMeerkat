#!/bin/bash 

#start virtual env
source env/bin/activate

#make sure all requirements are upgraded
pip install -r requirements.txt

############
#Train Model
############

#Create Docs
python CreateDocs.py

#how may eval records?
eval=$(cat /Users/ben/Documents/DeepMeerkat/training/Classification/Samples/Testing/testingdata.csv | wc -l)

python pipeline.py \
    --train_input_path /Users/ben/Documents/DeepMeerkat/training/Classification/Samples/Training/trainingdata.csv \
    --eval_input_path /Users/ben/Documents/DeepMeerkat/training/Classification/Samples/Testing/testingdata.csv \
    --input_dict /Users/ben/Documents/DeepMeerkat/training/Classification/Samples/dict.txt \
    --output_dir /Users/ben/Documents/DeepMeerkat/training/Classification/Output/ \
    --eval_set_size  ${eval} 

#If running from pre-processed TFRecords
#python pipeline.py \
    #--preprocessed_train_set /Users/ben/Documents/DeepMeerkat/training/Classification/Output/preprocessed/train* \
    #--preprocessed_eval_set /Users/ben/Documents/DeepMeerkat/training/Classification/Output/preprocessed/eval* \
    #--input_dict /Users/ben/Documents/DeepMeerkat/training/Classification/Samples/dict.txt \
    #--output_dir /Users/ben/Documents/DeepMeerkat/training/Classification/Output/ \
    #--eval_set_size  ${eval} 

    
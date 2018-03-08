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

python pipeline.py \
    --project ${PROJECT} \
    --train_input_path Samples/Training/trainingdata.csv \
    --eval_input_path Samples/Testing/testingdata.csv \
    --input_dict Samples/dict.txt \
    --output_dir Output/

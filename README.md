# Grading Writing with a Large Language Model

## Overview
In this project, I fine-tune Llama 3-8B-Instruct and evaluate its performance grading student essays. Specifically, the model was fine-tuned thrice:
1. Fine-tuned on persuasive essays
2. Fine-tuned on source-dependent essays
3. Fine-tuned on narrative essays

The performance of the models was quantified using the quadratic weighted Kappa. 

## Data
The initial dataset is here: https://698modeldata.s3.us-east-2.amazonaws.com/training_set_rel3.xls and is manipulated in "AES_data_prep.ipynb" in this repo. 

The resultant files are here:

### Persuasive Model Files:
Train: https://698modeldata.s3.us-east-2.amazonaws.com/train_persuasive_model_input.csv

Evaluation: https://698modeldata.s3.us-east-2.amazonaws.com/eval_persuasive_model_input.csv

Test: https://698modeldata.s3.us-east-2.amazonaws.com/test_persuasive_model_input.csv

### Source Dependent Model Files:
Train: https://698modeldata.s3.us-east-2.amazonaws.com/train_source_dependent_model_input.csv

Evaluation: https://698modeldata.s3.us-east-2.amazonaws.com/eval_source_dependent_model_input.csv

Test: https://698modeldata.s3.us-east-2.amazonaws.com/test_source_dependent_model_input.csv

### Narrative Model Input Files:
Train: https://698modeldata.s3.us-east-2.amazonaws.com/train_narrative_model_input.csv

Evaluation: https://698modeldata.s3.us-east-2.amazonaws.com/eval_narrative_model_input.csv

Test: https://698modeldata.s3.us-east-2.amazonaws.com/test_narrative_model_input.csv

## Models
Three models are trained in notebooks called: "Model for Persuasive Essays.ipynb", "Model for Source-Dependent Essays.ipynb" and "Model for Narrative Essays.ipynb", all in this repo.

The fine-tuned models are in seperate folders in this repo. 

## Scoring
The models are evaluated in "scoring.ipynb"





# SemEval-19 Task 9 - Suggestion Mining for Online Reviews and Forums

This repo contains a basic and an advanced approach to classify reviews as suggestions/non-suggestions. 
The goal of the task is to mine explicit suggestions (requiring least context) for entity the review is posted.

## Approaches
### Basic
Decisions lists using Trigram features are used for our basic approach to this problem. It performs better (F-1: 0.418) 
than SemEval-19 participating team `UOL Artificial Intelligence Research Group (Ahmed et al., 2019)` (F-1: 0.3237)
and baseline (F-1: 0.268).

### Advanced
Our advanced approach uses Transfer Learning, utilizing pre-trained BERT model for feature extraction and internal
representation learning. More details of the model to be added here.

## Build

### Pre-Requisites
`bash` terminal with python 3.7 available in active environment.

### Install
`bash INSTALL.sh` installs all required dependencies/packages for this code.

### Experiment
`bash EXPERIMENT.sh` runs the experiment on task's official data-set and reproduces same results 
as mentioned here or in paper.

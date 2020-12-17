# SemEval-19 Task 9 - Suggestion Mining for Online Reviews and Forums

This repo contains a basic and an advanced approach to classify reviews as suggestions/non-suggestions. 
The goal of the task is to mine explicit suggestions (requiring least context) for entity the review is posted.

## Approaches
### Basic
Decisions lists using Trigram features are used for our basic approach to this problem. It performs better 
(F-1: **0.44** on `evaluation/test data`, **0.621** on `trial/validation data`) 
than few of the SemEval-19 participating teams such as 
`UOL Artificial Intelligence Research Group (Ahmed et al., 2019)` (F-1: 0.3237) and baseline (F-1: 0.268).

### Advanced
Our advanced approach uses Transfer Learning, utilizing pre-trained BERT model for feature extraction and internal
representation learning. It ranks 6th among 32 teams, with F-1 score of 0.7487. We have re-trained the 
[BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification) 
model for just 3 epochs without freezing weights in initial layers.

## Build

### Pre-Requisites
`bash` terminal with python 3.7 available in active environment. 
A cleaner approach would be to install dependencies in a virtual environment such as conda. 
```bash
conda create --name suggestionMining python=3.7  # Create a env with python 3.7
conda activate suggestionMining # Activate the environment
```

### Install
The following command installs all required dependencies/packages for this code.
```
bash INSTALL.sh
```

### Experiment
The following command runs the experiment on task's official data-set and reproduces same results 
as mentioned here or in paper.
```bash
bash EXPERIMENT.sh
```

#### Grid Search

To find the optimum, an exhaustive grid-search is used. 
For basic approach it can be executed via `BASIC_GRID_SEARCH.sh` bash script. 
Following is an example of running such experiment and saving results to a CSV file:

```bash
bash BASIC_GRID_SEARCH.sh > results/basic_grid_search.csv
```

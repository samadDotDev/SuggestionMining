# SemEval-19 Task 9 - Suggestion Mining for Online Reviews and Forums

This repo contains a basic and an advanced approach to classify reviews as suggestions/non-suggestions. 
The goal of the task is to mine explicit suggestions (requiring least context) for entity the review is posted.

## Approaches
### Basic
Decisions lists using Trigram features are used for our basic approach to this problem. It performs better 
(F-1: **0.599** on `testData`, **0.418** on `evaluationData`) 
than SemEval-19 participating teams: 
- `Taurus (Oostdijk and Halteren, 2019)` (F-1: 0.5845),
- `YNU DYX (Ding et al., 2019)` (F-1: 0.5659),
- `INRIA (Markov and De la Clergerie, 2019)` (F-1: 0.5118),
- `SSN-SPARKS (S et al., 2019)` (F-1: 0.494),
- `DBMS-KU (Fatyanosa et al., 2019)` (F-1: 0.473),
- `UOL Artificial Intelligence Research Group (Ahmed et al., 2019)` (F-1: 0.3237),
- and baseline (F-1: 0.268).

### Advanced
Our advanced approach uses Transfer Learning, utilizing pre-trained BERT model for feature extraction and internal
representation learning. More details of the model to be added here.

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

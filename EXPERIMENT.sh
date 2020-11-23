# This script will run all experiments and reproduce results mentioned in paper

# Python and its version check from SO: https://stackoverflow.com/a/33183884/3743430 (Accessed Nov 23, 2020)
# Make sure python exists
version=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ -z "$version" ]]
then
    echo "No Python found"
    exit 1
fi
# Make sure python version is >= 3.7
parsedVersion=$(echo "${version//./}")
if [[ "$parsedVersion" -lt "370" ]]
then
    echo "Invalid version"
    exit 1
fi

# Define paths to dataset files
trainFile="data/V1.4_Training.csv"
valFile="SubtaskA_Trial_Test.csv"
valFileLabelled="SubtaskA_Trial_Test_Labeled.csv"
testFile="data/SubtaskA_EvaluationData.csv"
testFileLabelled="data/SubtaskA_EvaluationData_labeled.csv"

# Run basic experiment
# Train
python basic/train.py $trainFile > generated/decisions-list.txt
# Test
python basic/test.py generated/decisions-list.txt $testFile > generated/system-answers.txt
# Evaluate
python basic/eval.py $testFileLabelled generated/system-answers.txt > results/basic-results.txt


# TODO: Add advanced approach commands below

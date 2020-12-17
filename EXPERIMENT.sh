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
parsedVersion="${version//./}"
if [[ "$parsedVersion" -lt "370" ]]
then
    echo "Invalid version"
    exit 1
fi

# Define paths to dataset files
trainFile="data/V1.4_Training.csv"
valFile="data/SubtaskA_Trial_Test.csv"
valFileLabelled="data/SubtaskA_Trial_Test_Labeled.csv"
testFile="data/SubtaskA_EvaluationData.csv"
testFileLabelled="data/SubtaskA_EvaluationData_labeled.csv"

# Config related to basic approach
decisionsListFile="generated/decisions-list.txt"
basicSystemAnswersFile="generated/basic-system-answers.txt"
basicResultsFile="results/basic-results.txt"
ngramConfig="34"
minFrequency=1
notFeature=1

# Run basic experiment
# Train
python basic/train.py $trainFile $ngramConfig $minFrequency $notFeature > $decisionsListFile
# Test
python basic/test.py $decisionsListFile $testFile > $basicSystemAnswersFile
# Evaluate
python basic/eval.py $testFileLabelled $basicSystemAnswersFile > $basicResultsFile

# Config related to advanced approach
modelExportDir="generated/model/"
advancedSystemAnswersFile="generated/advanced-system-answers.txt"
advancedResultsFile="results/advanced-results.txt"

# Run advanced experiment
# Train
python advanced/train.py $trainFile $valFileLabelled $modelExportDir
# Test
python advanced/test.py $modelExportDir $testFile > $advancedSystemAnswersFile
# Evaluate
python advanced/eval.py $testFileLabelled $advancedSystemAnswersFile > $advancedResultsFile

# Print a summary
printf "Basic " && head -1 $basicResultsFile
printf "Advanced " && head -1 $advancedResultsFile

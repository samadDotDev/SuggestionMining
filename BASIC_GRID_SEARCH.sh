# Define paths to dataset files
trainFile="data/V1.4_Training.csv"
testFile="data/SubtaskA_EvaluationData.csv"
testFileLabelled="data/SubtaskA_EvaluationData_labeled.csv"

# Config related to files to be generated
decisionsListFile="generated/decisions-list.txt"
basicSystemAnswersFile="generated/basic-system-answers.txt"
basicResultsFile="generated/basic-grid-search-results.txt"

printf "nGramConfig,minFrequency,notFeature,F1\n"

for notFeature in 1 0
do
  for ngramConfig in "1" "2" "3" "4" "12" "13" "14" "23" "24" "34" "123" "124" "234" "1234"
  do
    for minFrequency in 1 2 3 4 5 10 15 20 50 100
    do
      # Train
      python basic/train.py $trainFile $ngramConfig $minFrequency $notFeature > $decisionsListFile
      # Test
      python basic/test.py $decisionsListFile $testFile > $basicSystemAnswersFile

      printf $ngramConfig && printf "," && printf $minFrequency && printf "," && printf $notFeature && printf ","

      # Evaluate
      # Setting last flag to true prints only f1 score
      python basic/eval.py $testFileLabelled $basicSystemAnswersFile True

      # Following is a more descriptive dump
#      python basic/eval.py $testFileLabelled $basicSystemAnswersFile > $basicResultsFile
#      # Print a summary
#      printf "N-gram: " && printf $ngramConfig
#      printf ", Min Frequency: " && printf $minFrequency
#      printf ", Not Feature: " && printf $notFeature && printf "\n"
#      head -1 $basicResultsFile && printf "\n"


    done
  done
done

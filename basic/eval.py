"""

This program compares gold standard class of a review and predicted class, 
calculates some stats (such as accuracy, precision, recall) 
and prints each review's gold and predicted values.

-------------------------------

This program can be executed using python version 3.7 in the following way at terminal:

python eval.py gold.txt system-answers.txt > basic-results.txt

Where,
eval.py: the filename of the following complete code
gold.txt: name of file containing gold standard labels (review class)
system-answers.txt: name of file containing predicted classes
> basic-results.txt: Piped output to a txt file to which results will be dumped

The program will then generate a file (basic-results.txt) containing results in following example:

Summary: Accuracy: 0.61, Precision: 0.6195652173913043, Recall: 0.57, F1: 0.59375


Classification Answers Comparison:

Review_Identifier | Gold | Prediction
cv666_tok-13320.txt | 1 | 1
cv535_tok-19937.txt | 0 | 0
cv245_tok-19462.txt | 1 | 0
...

-------------------------------

Overview of Program's Code:

1. Read CmdLine Arguments for file names (gold and system answers) and validate they exists
2. Read both files and compare class for each review
    a. Read system answers file and make a hashmap of answers keyed by review identifier
    b. Read Gold standard answers and compare simultaneously while making a list of comparison output
    c. Return counts of TP, FP, TN, FN and compared results of reviews
3. Calculate Accuracy/Precision/Recall and Export Results

"""

# To use utilities for extracting cmdline args
import sys
# For checking if file exists on a path
import os.path
# For enabling regular expressions
import re


# By default print a descriptive summary with analysis for each system answer
print_f1_score_only = False


# 1. Read CmdLine Arguments for file names (gold and system answers) and validate they exists
def parse_cmd_line_args():

    # first arg is this file's name, second and third gold and system answers files
    if len(sys.argv) < 3:
        print("Gold and System answers file names are required")
        exit(0)

    gold_file = sys.argv[1]
    system_answers_file = sys.argv[2]

    # Make sure both files exists

    if not os.path.isfile(gold_file):
        print(f"File {gold_file} doesn't exist")
        exit(0)

    if not os.path.isfile(system_answers_file):
        print(f"File {system_answers_file} doesn't exist")
        exit(0)

    # An optional argument to control whether we print a descriptive summary of results or just f1 score
    global print_f1_score_only
    if len(sys.argv) >= 4:
        print_f1_score_only = bool(sys.argv[3])

    return gold_file, system_answers_file


# 2. Read both files and compare class for each review
def read_and_evaluate(gold_file, system_answers_file):

    TP, TN, FP, FN = 0, 0, 0, 0

    # a. Read system answers file and make a hashmap of answers keyed by review identifier
    system_answers = {}
    with open(system_answers_file, "r", encoding="utf8") as file:

        # Iterate through each line in file
        for line in file:

            # Split line by space
            line_split = line.split(" ")

            # Make sure it doesn't have line break, and convert class to int
            review_class = int(line_split[1].replace('\n', ''))
            review_identifier = line_split[0]

            # Populate hash map of answers
            system_answers[review_identifier] = review_class

    # b. Read Gold standard answers and compare simultaneously while making a list of comparison output
    comparison_output = []
    line_number = 0
    with open(gold_file, "r", encoding="utf8") as file:

        # Iterate through each line in file
        for line in file:

            line_number += 1
            if line_number == 1:
                # Skip the first line as it is header
                continue

            # i. Split line in following pattern:
            # <Identifier: Numbers optionally with _>,<Review: remaining txt>,<Class: 0 or 1>
            line_split = re.match(r'^([0-9_]*),"?(.*)"?,([01])$', line)

            # Make sure it doesn't have line break, and convert class to int
            gold_class = int(line_split.group(3))
            # Removing _ makes sure 12_1 in test data matches 121 in gold data
            review_identifier = line_split.group(1).replace('_', '')
            predicted_class = system_answers[review_identifier]

            if gold_class == predicted_class:

                if predicted_class == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if predicted_class == 1:
                    FP += 1
                else:
                    FN += 1

            # Populate list of comparison for this review
            comparison_output.append([review_identifier, gold_class, predicted_class])

    # c. Return counts of TP, FP, TN, FN and compared results of reviews
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "review_answers": comparison_output}


# 3. Calculate Accuracy/Precision/Recall and Export Results
def calculate_stats_and_export_results(results):
    TP = results["TP"]
    TN = results["TN"]
    FP = results["FP"]
    FN = results["FN"]
    review_answers = results["review_answers"]

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else "N/A"
    precision = (TP / (TP + FP)) if (TP + FP) > 0 else "N/A"
    recall = (TP / (TP + FN)) if (TP + FN) > 0 else "N/A"
    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision != "N/A" and recall != "N/A") else "N/A"

    if print_f1_score_only:
        print(f1)
    else:
        print("Summary:", f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        print("\n\nClassification Answers Comparison:\n")
        print("Review_Identifier", "|", "Gold", "|", "Prediction")

        for answer in review_answers:
            print(answer[0], "|", answer[1], "|", answer[2])


# Program's Entry Point
if __name__ == "__main__":

    # 1. Read CmdLine Arguments for file names (gold and system answers) and validate they exists
    gold_file_name, system_answers_file_name = parse_cmd_line_args()

    # 2. Read both files and compare class for each review
    results = read_and_evaluate(gold_file_name, system_answers_file_name)

    # 3. Calculate Accuracy/Precision/Recall and Export Results
    calculate_stats_and_export_results(results)

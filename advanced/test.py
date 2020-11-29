"""

This program ...

-------------------------------

This program can be executed using python version 3.7 in the following way at terminal:

...

Where,
...

-------------------------------

Overview of Program's Code:

...

"""

# To use utilities for extracting cmdline args
import sys
# For checking if file exists on a path
import os.path
# For enabling regular expressions
import re


# 1. Read CmdLine Arguments for file names (model and test) and validate they exists
def parse_cmd_line_args():
    # first arg is this file's name, second and third are model and test files
    if len(sys.argv) < 3:
        print("Model and test data file names are required")
        exit(0)

    model_file = sys.argv[1]
    test_file = sys.argv[2]

    # Make sure both files exists

    if not os.path.isfile(model_file):
        print(f"File {model_file} doesn't exist")
        exit(0)

    if not os.path.isfile(test_file):
        print(f"File {test_file} doesn't exist")
        exit(0)

    return model_file, test_file


# 2. Read both files and predict sentiments for test file
def read_and_predict(model_file, test_file):
    reviews = []

    # a. Read Test file and Predict class for each review
    with open(test_file, "r", encoding='utf-8') as file:

        # Iterate through each line in file
        for line in file:

            # i. Split line in following pattern:
            # <Identifier: Numbers optionally with _>,<Review: remaining txt>,X
            line_split = re.match(r'^([0-9_]*),"""(.*)""",X$', line)

            # Group /1 has filename or review identifier, Group /2 has review text
            review_identifier = line_split.group(1)
            review_text = line_split.group(2).lower()

            predicted_class = 0
            # TODO: Use model file to get trainer instance and use it for predict method with review text
            # predicted_class = some_helper(model_file, review_text)

            # iii. Associate the probable class with review identifier / filename
            reviews.append({"identifier": review_identifier, "class": predicted_class})

    return reviews


# 3. Export Predictions
def export(reviews):
    # Go through each review
    for review in reviews:
        # Dump/Print (to stdout) in format: Identifier <space> Class
        print(review["identifier"], review["class"])


# Program's Entry Point
if __name__ == "__main__":

    # 1. Read CmdLine Arguments for file names (decision-list and test) and validate they exists
    model_file_name, test_file_name = parse_cmd_line_args()

    # 2. Read both files and predict class for test file
    predictions = read_and_predict(model_file_name, test_file_name)

    # 3. Export Predictions
    export(predictions)

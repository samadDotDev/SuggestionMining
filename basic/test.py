"""

This program predicts sentiment of a review based on features present in it.
It takes a decision list file that lists features sorted by likelihood and predicts for 
each review in test file the most probable class if it is able to match a feature in review text.

-------------------------------

This program can be executed using python version 3.7 in the following way at terminal:

python test.py decision-list.txt test.txt > system-answers.txt

Where,
test.py: the filename of the following complete code
decision-list.txt: name of file containing decision list (exported by training program)
test.txt: name of file containing test data
> system-answers.txt: Piped output to a txt file to which predicted answers will be exported

The program will then generate a file (system-answers.txt) containing predicted answers in following format:
<ReviewIdentifier> <space> <PredictedClass>

Example:
cv666_tok-13320.txt 1
cv535_tok-19937.txt 0
cv245_tok-19462.txt 0
...

-------------------------------

Overview of Program's Code:

1. Read CmdLine Arguments for file names (decision-list and test) and validate they exists
2. Read both files and predict sentiments for test file
    a. Read decision list file
    b. Sort the decision list by descending order of likelihood values
    c. Read Test file and Predict class for each review
        i. Parse each line by splitting it
        ii. Iterate through ordered decision list to check if a feature exist in review
        iii. Associate the probable class with review identifier / filename
3. Export Predicted Sentiments

"""

# To use utilities for extracting cmdline args
import sys
# For checking if file exists on a path
import os.path
# For enabling regular expressions
import re


# Global Config

# Also associate NOT_ with negated part of sentence, to be able to compare with NOT_* features of Decision List
associate_not_in_negated_sentence = True


# 1. Read CmdLine Arguments for file names (decision-list and test) and validate they exists
def parse_cmd_line_args():

    # first arg is this file's name, second and third are decision list and test files
    if len(sys.argv) < 3:
        print("Decision list and test data file names are required")
        exit(0)

    decision_list_file = sys.argv[1]
    test_file = sys.argv[2]

    # Make sure both files exists

    if not os.path.isfile(decision_list_file):
        print(f"File {decision_list_file} doesn't exist")
        exit(0)

    if not os.path.isfile(test_file):
        print(f"File {test_file} doesn't exist")
        exit(0)

    return decision_list_file, test_file


# Helper to associate NOT_ with words after negation in sentence if present
def associate_not(sentence):
    # Split sentence on no, not, or <alphabets>n't
    sentence = re.split(r'(\bnot?\b|\b[a-z]*n\'t\b)', sentence, re.IGNORECASE)

    negation = False
    negated_sentence = ""

    for words in sentence:

        if negation:
            # Associate words with NOT_ if negation is on
            words = re.sub(r'(\w+)', r'NOT_\1', words)

        if words.lower() in ["no", "n't", "not"]:
            negation = True

        # Concatenate the words back into a new sentence
        negated_sentence += str(words)

    return negated_sentence


def associate_not_in_complete_review(text):

    # Split a sentence by a punctuation (,.!?;) possibly after and before a space (or may not)
    # sentences = re.split(r'\s?\.\s?|\s?,\s?|\s?;\s?|\s?!\s?|\s?\?\s?|\s?\n', text)

    punctuations = [r'\.', r',', r';', r'!', r'\?', '\n']
    sentences = re.split(r'(' + r'|'.join(punctuations) + r')', text)

    # Remove any blank sentences (occurs after split if a punctuation is right before line break)
    # Also, associate NOT_ with words after negation in sentence if present

    negated_sentences = []
    for sentence in sentences:
        if sentence == "":
            continue
        if sentence not in punctuations:
            negated_sentences.append(associate_not(sentence))
        else:
            negated_sentences.append(sentence)

    # Concatenate back the sentences into a single text
    text = "".join(negated_sentences)

    return text


# 2. Read both files and predict sentiments for test file
def read_and_predict_sentiments(decision_list_file, test_file):

    decisions_list = []

    # a. Read decision list file
    line_number = 0
    with open(decision_list_file, "r") as file:

        # Iterate through each line in file
        for line in file:
            
            line_number += 1
            if line_number == 1:
                continue  # Skip the first line as it contains header
            
            # Split the line by " | " separators
            line_split = line.split(" | ")

            # Remove line break at the end
            line_split[2] = line_split[2].replace('\n', '')

            # Populate decisions list (an array) for iteration later
            decisions_list.append(line_split)

    # b. Sort the decision list by descending order of likelihood values
    # although already supplied in this order, just making sure
    decisions_list.sort(key=lambda x: x[0], reverse=True)

    reviews = []

    # c. Read Test file and Predict class for each review
    with open(test_file, "r", encoding='utf-8') as file:

        # Iterate through each line in file
        for line in file:
            
            # i. Split line in following pattern:
            # <Identifier: Numbers optionally with _>,<Review: remaining txt>,X
            line_split = re.match(r'^([0-9_]*),"""(.*)""",X$', line)

            # Group /1 has filename or review identifier, Group /2 has review text
            review_identifier = line_split.group(1)

            if associate_not_in_negated_sentence:
                review_text = associate_not_in_complete_review(line_split.group(2).lower())
            else:
                review_text = line_split.group(2).lower()

            # Default class when there is no match, could also try making a random int b/w [0,1]
            # But doesn't matter for current dataset as all reviews in test seem to end up matching a feature
            predicted_class = 0
            
            # ii. Iterate through ordered decision list to check if a feature exist in review
            for feature in decisions_list:
                
                if feature[2] in review_text:
                    
                    # Use the feature's class once it is found in the review
                    predicted_class = feature[1]
                    
                    # Break the loop, since we don't need to go through rest of the features 
                    # once highest likelihood feature is matched
                    break
                
            # iii. Associate the probable class with review identifier / filename
            reviews.append({"identifier": review_identifier, "class": predicted_class})

    return reviews


# 3. Export Predicted Sentiments
def export_sentiments(reviews):
    
    # Go through each review
    for review in reviews:
        
        # Dump/Print (to stdout) in format: Identifier <space> Class
        print(review["identifier"], review["class"])


# Program's Entry Point
if __name__ == "__main__":

    # 1. Read CmdLine Arguments for file names (decision-list and test) and validate they exists
    decision_list_file_name, test_file_name = parse_cmd_line_args()

    # 2. Read both files and predict sentiments for test file
    sentiments_predicted_reviews = read_and_predict_sentiments(decision_list_file_name, test_file_name)

    # 3. Export Predicted Sentiments
    export_sentiments(sentiments_predicted_reviews)

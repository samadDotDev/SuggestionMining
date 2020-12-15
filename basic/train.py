"""

This program learns sentiment of a review based on features present in it.
It calculates the likelihood of each feature by judging probability of each class
given a feature in its review. It then generates a list of such features sorted in
descending order by likelihood, called decision list.

-------------------------------

This program can be executed using python version 3.7 in the following way at terminal:

python train.py train.txt > decision-list.txt

Where,
train.py: the filename of the following complete code
train.txt: name of file containing training data existing in the same directory, could also be after complete or relative path to it
> decision-list.txt: Piped output to a txt file to which decision list will be exported

The program will then generate a file (decision-list.txt) containing decision list in following example:

Likelihood | Class | Feature
5.357552004618082 | 1 | bateman
5.247927513443585 | 1 | ordell
...

-------------------------------

Overview of Program's Code:

1. Read CmdLine Arguments for training file name and validate it exists
2. Read Input file and pre-process it
    a. Parse file to a list of Review objects for easier access
    b. For each review text, associate NOT_ to words after not or n't before next punctuation
3. Extract features (unigrams and bigrams) from review text and associate counts
4. Generate decision list of features
    a. Calculate Probabilities
    b. Calculate Log-likelihood value and associate with feature
    c. Sort the decision list by likelihood value for features
5. Export the decision list (print/dump to stdout)

"""

# To use utilities for extracting cmdline args
import sys
# For checking if file exists on a path
import os.path
# For enabling regular expressions
import re
# For log function
import math


# Global Config for program (Defaults, could be modified by cmdline args)

add_one_smoothing = True  # add 1 smoothing for probabilities calculation

# Whether NOT_ should be appended to words later than a not or a similar contraction
# (didn't ..) in a sentence
not_feature_enabled = True

# Following if true turns "not in a no good way" to "not NOT_in NOT_a NOT_no good way"
# Otherwise if false, turns it into "not NOT_in NOT_a NOT_no NOT_good NOT_way"
double_negation_turns_positive = True and not_feature_enabled  # Either way, doesn't change performance on given data

# Min # of occurrence of a feature in whole training dataset to be considered for decision list
minimum_frequency_of_feature = 2

# configuration for n-gram models to be used for decisions list
# "3" means only trigram will be used, "123" specifies that unigram, bigram and trigram combined should be used
# The config can contain 1,2,3,4 or any combinations of these without any order sensitivity
n_gram_config = "3"


# Class for abstraction of review data type
class Review:
    sentences = [] # List of cleaned/preprocessed sentences in review
    classification = 0

    def __init__(self, sentences, classification):
        self.sentences = sentences
        self.classification = classification


# 1. Read CmdLine Arguments for input file name and validate it exists
def parse_cmd_line_args():

    # first arg is this file's name, second arg is training input file name
    if len(sys.argv) < 2:
        print("Input file name (containing training data) is required")
        exit(0)

    # Take second arg which is input file name
    input_file_name = sys.argv[1]

    # Make sure it exists
    if not os.path.isfile(input_file_name):
        print(f"File {input_file_name} doesn't exist")
        exit(0)

    # Following configuration is optional to be supplied,
    # otherwise program defaults listed in global program config above will be used

    global n_gram_config, minimum_frequency_of_feature, not_feature_enabled

    if len(sys.argv) >= 3:
        n_gram_config = sys.argv[2]

    if len(sys.argv) >= 4:
        minimum_frequency_of_feature = int(sys.argv[3])

    if len(sys.argv) >= 5:
        not_feature_enabled = bool(sys.argv[4])

    return input_file_name


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
            # Turn on negation for future words
            if double_negation_turns_positive:
                # Flip negation, may be useful for double negations in sentence
                negation = not negation
            else:
                # Always keep negation true even with new negations
                negation = True

            # Concatenate the words back into a new sentence
        negated_sentence += str(words)

    return negated_sentence


# Helper to split text into sentence and pre-process them (including NOT_ association)
def split_into_sentences_and_preprocess(text):

    # Split a sentence by a punctuation (,.!?;) possibly after and before a space (or may not)
    sentences = re.split(r'\s?\.\s?|\s?,\s?|\s?;\s?|\s?!\s?|\s?\?\s?|\s?\n', text)

    # Remove any blank sentences (occurs after split if a punctuation is right before line break)
    sentences = [s for s in sentences if s != ""]

    # associate NOT_ with words after negation in sentence if present, and if this feature is enabled
    if not_feature_enabled:
        sentences = [associate_not(s) for s in sentences]

    return sentences


# Helper to clean review text
def clean_review(review_text):

    review_text = review_text.replace("_", "")

    return review_text


# 2. Read Input file and pre-process it
def pre_process_input(file_name):

    reviews = []  # Initialize an empty list of reviews, to be populated while reading file

    # Read File
    with open(file_name, "r", encoding='utf-8') as file:

        # Iterate through each line in file
        for line in file:

            # a. Parse file to a list of Review objects for easier access

            # Split line in following pattern:
            # TODO: Handle reviews with _ in their identifiers, group them for same id before _
            # <Review Number possibly with underscore> <,"""> <Review Text> <""",> <Class: 0 or 1>
            line_split = re.match(r'^([0-9_]*),"""(.*)""",([01])$', line)

            if line_split is None:
                # Skip this line if there is no pattern match
                continue

            # Group /3 has class, Group /2 has review text
            review_class = line_split.group(3)
            review_text = line_split.group(2).lower()

            # a.1 Clean Review Text
            review_text = clean_review(review_text)

            # b. For each review text, associate NOT_ to words after not or n't before next punctuation
            # First split into sentences, then perform the association for relevant sentences
            review_sentences = split_into_sentences_and_preprocess(review_text)

            # Populate reviews list with this review
            reviews.append(Review(review_sentences, review_class))

    return reviews


# 3. Extract features (unigrams and bigrams) from review text and associate counts
# (we keep track of class1 and total counts, class0 can be inferred)
def extract_features(reviews):
    
    features = {
        "unigrams": {},
        "bigrams": {},
        "trigrams": {},
        "fourgrams": {}}

    # Iterate through reviews
    for review in reviews:

        # Iterate through sentences in this review
        for sentence in review.sentences:

            # Split the sentence into a list of tokens and remove empty tokens
            tokens = [token for token in re.split(r'[^a-zA-Z.?!,_\']', sentence) if token != ""]
            
            tokens_len = len(tokens)
            for token_i in range(tokens_len):

                # UNI-GRAMS, if enabled
                if "1" in n_gram_config:

                    unigram = tokens[token_i]

                    # Increment class and total counts of Unigram if it exists in hashmap
                    if unigram in features["unigrams"]:
                        # int(review.classification) makes sure to increment count of class1 only if this review is class 1
                        features["unigrams"][unigram]["class1"] += int(review.classification)
                        features["unigrams"][unigram]["total"] += 1

                    else:
                        # Otherwise add unigram to hashmap if it doesn't exist
                        features["unigrams"][unigram] = {"class1": int(review.classification), "total": 1}

                # BI-GRAMS, if enabled
                if "2" in n_gram_config:

                    # Make sure we don't go over the last element in list when making combinations for bigrams
                    if token_i < tokens_len - 1:

                        # Concantenate this and next token to make it bigram
                        bigram = (tokens[token_i]) + " " + str(tokens[token_i + 1])

                        # Increment class and total counts of bigram if it exists in hashmap
                        if bigram in features["bigrams"]:
                            features["bigrams"][bigram]["class1"] += int(review.classification)
                            features["bigrams"][bigram]["total"] += 1

                        else:
                            # Otherwise add bigram to hashmap if it doesn't exist
                            features["bigrams"][bigram] = {"class1": int(review.classification), "total": 1}

                # TRI-GRAMS, if enabled
                if "3" in n_gram_config:

                    # Make sure we don't go over the last element in list when making combinations for trigrams
                    if token_i < tokens_len - 2:

                        # Concantenate this and next token to make it trigram
                        trigram = (tokens[token_i]) + " " + str(tokens[token_i + 1]) + " " + str(tokens[token_i + 2])

                        # Increment class and total counts of trigram if it exists in hashmap
                        if trigram in features["trigrams"]:
                            features["trigrams"][trigram]["class1"] += int(review.classification)
                            features["trigrams"][trigram]["total"] += 1

                        else:
                            # Otherwise add trigram to hashmap if it doesn't exist
                            features["trigrams"][trigram] = {"class1": int(review.classification), "total": 1}

                # 4-GRAMS, if enabled
                if "4" in n_gram_config:

                    # Make sure we don't go over the last element in list when making combinations for 4-grams
                    if token_i < tokens_len - 3:

                        # Concantenate this and next token to make it fourgram
                        fourgram = (tokens[token_i]) + " " + str(tokens[token_i + 1]) + " " + str(tokens[token_i + 2])\
                                  + " " + str(tokens[token_i + 3])

                        # Increment class and total counts of fourgram if it exists in hashmap
                        if fourgram in features["fourgrams"]:
                            features["fourgrams"][fourgram]["class1"] += int(review.classification)
                            features["fourgrams"][fourgram]["total"] += 1

                        else:
                            # Otherwise add fourgram to hashmap if it doesn't exist
                            features["fourgrams"][fourgram] = {"class1": int(review.classification), "total": 1}

    # This will return features in the following format:
    # unigrams: {the: {class1: 20, total: 40}, not: {class1: 0, total: 30}, ...},
    # bigrams: {"very bad": {class1: 0, total: 5}, ...}
    # trigrams: ...
    # fourgrams: ...
    return features


# 4. Generate decision list of features
def generate_decision_list(features):

    decision_list = []

    # Iterate through all features
    for feature_type in features.keys():
        for feature, counts in features[feature_type].items():

            # Skip if total occurrence of this feature in training dataset is less than specified min
            if counts["total"] < minimum_frequency_of_feature:
                continue

            # a. Calculate Probabilities
            # We can calculate just class 1 probabilities as class 0 can be inferred (1 - Probab(class 1))

            if add_one_smoothing:
                # Since there are 2 classes in total, we add +2 total counts
                class_1_probab = (counts["class1"] + 1) / (counts["total"] + 2)
            else:
                class_1_probab = counts["class1"] / counts["total"]

            class_0_probab = 1 - class_1_probab

            # Mark the probable for which the likelihood value will be calculated
            if class_1_probab > class_0_probab:
                probable_class = 1
                # likelihood = class_1_probab / class_0_probab
            else:
                probable_class = 0
                # likelihood = class_0_probab / class_1_probab

            # b. Calculate Log-likelihood value and associate with feature
            likelihood = abs(math.log((class_0_probab / class_1_probab), 2))

            decision_list.append([likelihood, probable_class, feature])

    # c. Sort the decision list by likelihood value for features
    decision_list.sort(key=lambda x: x[0], reverse=True)

    return decision_list


# 5. Export the decision list (print/dump to stdout)
def export(decision_list):
    
    print("Likelihood", "|", "Class", "|", "Feature")
    
    # Features are separated by line
    for decision in decision_list:

        # Format: Likelihood | Class | Feature
        print(decision[0], "|", decision[1], "|", decision[2])


# Program's Entry Point
if __name__ == "__main__":

    # 1. Read CmdLine Arguments for training file name and validate it exists
    train_file_name = parse_cmd_line_args()

    # 2. Read Input file and pre-process it
    reviews = pre_process_input(train_file_name)

    # 3. Extract features (unigrams and bigrams) from review text and associate counts
    features = extract_features(reviews)

    # 4. Generate decision list of features
    decision_list = generate_decision_list(features)

    # 5. Export the decision list (print/dump to stdout)
    export(decision_list)

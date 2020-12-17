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
# To facilitate transfer learning
from transformers import BertForSequenceClassification, Trainer, BertTokenizerFast
# To use PyTorch's dataset class for prediction
import torch
# To utilize ArgMax feature
import numpy as np


# Test set Child of PyTorch's dataset class which is required for predictions method
class ReviewsTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, length):
        self.encodings = encodings
        self.length = length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.length


# 1. Read CmdLine Arguments for file names (model and test) and validate they exists
def parse_cmd_line_args():
    # first arg is this file's name, second and third are model dir and test files
    if len(sys.argv) < 3:
        print("Model and test data file names are required")
        exit(0)

    model_dir = sys.argv[1]
    test_file = sys.argv[2]

    # Make sure both files exists

    if not os.path.isdir(model_dir):
        print(f"Directory {model_dir} doesn't exist")
        exit(0)

    if not os.path.isfile(test_file):
        print(f"File {test_file} doesn't exist")
        exit(0)

    return model_dir, test_file


# 2. Read both files and predict sentiments for test file
def read_and_predict(model_dir, test_file):
    reviews = []

    # Extract the model from provided model directory
    model = BertForSequenceClassification.from_pretrained(model_dir)

    # Load the same tokenizer as we used in training
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Initialize HuggingFace's high level trainer interface using pre-trained model
    trainer = Trainer(model=model)

    # Read Test file and Predict class for each review
    with open(test_file, "r", encoding='utf-8') as file:

        # Iterate through each line in file
        for line in file:

            # Split line in following pattern:
            # <Identifier: Numbers optionally with _>,<Review: remaining txt>,X
            line_split = re.match(r'^([0-9_]*),"(.*)",X$', line)

            # Group /1 has filename or review identifier, Group /2 has review text
            review_identifier = line_split.group(1)
            review_text = line_split.group(2).lower()

            # Pass the review text in a list with only element
            encodings = tokenizer([review_text,], truncation=True, padding=True)

            # Build dataset object with length of 1 (our list here has only one element)
            review_text_as_dataset = ReviewsTestDataset(encodings, 1)

            # Predictions come out in the form of one hot vector of classes
            # e.g. [-3.12, 2.14] means likeliness of second class (index 1)
            prediction_one_hot = trainer.predict(review_text_as_dataset).predictions[0]

            # Select the class out of one-hot vector
            # e.g: [-3.12, 2.14] should return 1, [1.23, 0.5] should return 0
            predicted_class = np.argmax(prediction_one_hot)

            # Associate the probable class with review identifier / filename
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

    # 1. Read CmdLine Arguments and validate the existence of paths
    model_dir_name, test_file_name = parse_cmd_line_args()

    # 2. Read both files and predict class for test file
    predictions = read_and_predict(model_dir_name, test_file_name)

    # 3. Export Predictions
    export(predictions)

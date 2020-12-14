"""

This program learns ...

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
# To enable regular expressions
import re
# To enable one of the deep learning algorithms framework
import torch
# To facilitate transfer learning
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments


# Set a global set for training reproducibility
torch.manual_seed(0)


# The following data-set class and required format is taken from
# HuggingFace Transformers documentation (Fine-tuning with custom datasets)
# https://huggingface.co/transformers/custom_datasets.html (Accessed Nov 20, 2020)

# Create a child class of PyTorch Dataset class so we can conveniently use it in our trainer later
class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 1. Read CmdLine Arguments for input file name and validate it exists
def parse_cmd_line_args():

    # first arg is this file's name, remaining are our supplied arguments
    if len(sys.argv) < 4:
        print("Input file name (containing training data) is required")
        exit(0)

    # Take second arg which is training file name
    train_file_name = sys.argv[1]

    # Take third arg which is validation file name
    val_file_name = sys.argv[2]

    # Take fourth arg which is to be a relative directory (e.g. export/) for the refined model export
    model_dir = sys.argv[3]

    # Make sure files exist
    for file in [train_file_name, val_file_name]:
        if not os.path.isfile(file):
            print(f"File {file} doesn't exist")
            exit(0)

    return train_file_name, val_file_name, model_dir


# 2. Load pre-trained model and tokenizer for transfer learning
def load_pretrained_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return model, tokenizer


# 3. Read Input file and pre-process it
def pre_process_input(train_file, val_file, tokenizer):

    dataset = {}
    for dataset_title, dataset_file in {'Train': train_file, 'Validation': val_file}.items():

        # Initialize an empty list of review texts and their labels, to be populated while reading file
        texts, labels = [], []

        # Read File
        with open(dataset_file, "r", encoding='utf-8') as file:

            # Iterate through each line in file
            for line in file:

                line_split = None

                # Split line in following pattern (different for train and val set):
                if dataset_title == 'Train':
                    # TODO: Handle reviews with _ in their identifiers, group them for same id before _
                    # <Review Number possibly with underscore> <,"""> <Review Text> <""",> <Class: 0 or 1>
                    line_split = re.match(r'^([0-9_]*),"""(.*)""",([01])$', line)
                else:
                    # <Review Number possibly with underscore> <,"> <Review Text> <",> <Class: 0 or 1>
                    line_split = re.match(r'^([0-9_]*),"(.*)",([01])$', line)

                if line_split is None:
                    # Skip this line if there is no pattern match
                    continue

                # Group /3 has class, Group /2 has review text
                review_class = int(line_split.group(3))
                review_text = line_split.group(2).lower()

                texts.append(review_text)
                labels.append(review_class)

        # Encode the input (tokenize in the way it is required for the model)
        # Padding is set true to make sure shorter sentences are padded to make it to required length of model
        # Truncation is set true to reduce the length of larger sentences to what model can handle
        # Encoded values are returned as tensors
        encodings = tokenizer(texts, truncation=True, padding=True)
        dataset[dataset_title] = ReviewsDataset(encodings, labels)

    return dataset


# 4. Refine pre-trained model using our own training dataset
def retrain_model(model, dataset):

    # Training Arguments
    # TODO: Grid search on optimizing these arguments
    # TODO: May take few arguments from cmdline
    training_args = TrainingArguments(
        output_dir='generated',  # intermediate outputs (such as training checkpoints) directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=50,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='logs',  # directory for storing logs
        logging_steps=10,
    )

    # TODO: Freeze initial layers before training
    """for param in model.base_model.parameters():
        param.requires_grad = False"""

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=dataset['Train'], eval_dataset=dataset['Validation'])

    # Refine the model on training set
    trainer.train()

    # Optionally, evaluate the refined model on validation set
    trainer.evaluate()

    # TODO: Training curve and evaluation results to be plotted

    # Return updated/refined model
    return trainer.model


# 5. Export the refined model that can be used for testing later
def save_model(model, dir_name):

    # Use the utility provided by transformers to save the model in provided directory
    model.save_pretrained(dir_name)


# Program's Entry Point
if __name__ == "__main__":

    # 1. Read CmdLine Arguments for training file name and validate it exists
    train_file_name, val_file_name, export_model_dir = parse_cmd_line_args()

    # 2. Load pre-trained model and tokenizer for transfer learning
    pretrained_model, pretrained_tokenizer = load_pretrained_model_and_tokenizer()

    # 3. Read data-set files and pre-process them
    datasets = pre_process_input(train_file_name, val_file_name, pretrained_tokenizer)

    # 4. Refine pre-trained model using our own training dataset
    updated_model = retrain_model(pretrained_model, datasets)

    # 5. Export the refined model that can be used for testing later
    save_model(updated_model, export_model_dir)

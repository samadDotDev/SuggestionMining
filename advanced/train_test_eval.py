# TODO: This file will probably be split into train/test/eval to avoid data leakage

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# The following data-set class and train/val/test format is taken from
# HuggingFace Transformers documentation (Fine-tuning with custom datasets)
# https://huggingface.co/transformers/custom_datasets.html (Accessed Nov 20, 2020)

# Create an inherited Pytorch Dataset class so we can conveniently use it in our trainer later
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


# Load the data sets
train_texts, train_labels = ['I love this game', 'I hate it'], [1, 0]
val_texts, val_labels = ['I hate this game', 'I just love it'], [0, 1]
test_texts, test_labels = ['I like how it is played', 'I do not like it at all'], [1, 0]


# Encode the input (tokenize in the way it is required for the model)
# Padding is set true to make sure shorter sentences are padded to make it to required length of model
# Truncation is set true to reduce the length of larger sentences to what model can handle
# Encoded values are returned as tensors
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Combine the encodings with their respective labels in a ReviewsDataset object
train_set = ReviewsDataset(train_encodings, train_labels)
val_set = ReviewsDataset(val_encodings, val_labels)
test_set = ReviewsDataset(test_encodings, test_labels)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=val_set)

trainer.train()

trainer.evaluate()

"""
# Freeze the weights of pre-trained encoder (base_model doesn't include head layers)
for param in model.base_model.parameters():
    param.requires_grad = False

# Train model
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
"""

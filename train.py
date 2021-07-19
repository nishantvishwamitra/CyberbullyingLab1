import torch
#from torchtext import data
from torchtext.legacy import data

SEED = 1234
import pandas as pd
import numpy as np
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext

import nltk
import random
from sklearn.metrics import classification_report
import spacy
from models import LSTM

main_df = pd.read_csv('formspring_dataset.csv', sep = '\t')

# Let's see how many smaples we have
print(main_df.shape)
main_df = main_df.sample(n = main_df.shape[0])
main_df = main_df[['text', 'label']]

# Let's take a look at a few samples from our dataset
print(main_df.head())

# Let's take a look at the cyberbullying labels
print(main_df.label.value_counts())

# let's divide the dataset into non-cyberbullying and cyberbullying samples
o_class = main_df.loc[main_df.label == 0, :]
l_class = main_df.loc[main_df.label == 1, :]

# let's create train, val and test splits
train_val = main_df.iloc[:int(main_df.shape[0] * .80)]
test = main_df.iloc[int(main_df.shape[0] * .80):]
train = train_val.iloc[:int(train_val.shape[0] * .80)]
val = train_val.iloc[int(train_val.shape[0] * .80):]

print(train.shape, val.shape, test.shape)

print(train.label.value_counts())
print(val.label.value_counts())
print(test.label.value_counts())

train.to_csv("dataset/train.csv", index = False)
test.to_csv("dataset/test.csv", index = False)
val.to_csv("dataset/valid.csv", index = False)

# Let's use a tokenizer. This is the first step in NLP
spacy_en = spacy.load('en_core_web_sm')

TEXT = data.Field(sequential = True, tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.long, sequential = False)

# We will use a GPU to train our AI
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading train, test and validation data 
train_data, valid_data, test_data = data.TabularDataset.splits(
  path = "dataset/", train = "train.csv", 
  validation = "valid.csv", test = "test.csv", format = "csv", skip_header = True, 
  fields=[('Text', TEXT), ('Label', LABEL)]
)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of valid examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

# Let's build our cyberbullying vocabulary
TEXT.build_vocab(train_data, vectors = torchtext.vocab.Vectors('glove.840B.300d.txt'), max_size = 20000, min_freq = 10)
LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

BATCH_SIZE = 20

# keep in mind the sort_key option 
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
  (train_data, valid_data, test_data), sort_key=lambda x: len(x.Text),
  batch_size=BATCH_SIZE,
  device=device)

# Lets see the distributtion of samples in our dataset
print(LABEL.vocab.freqs)

# Lets define some hyperparameters
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 374
OUTPUT_DIM = 2
N_EPOCHS = 10

# Load up our AI model
model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# we need to use our pretrained embeddings
pretrained_embeddings = TEXT.vocab.vectors
print(pretrained_embeddings.shape)

model.embedding.weight.data = pretrained_embeddings.cuda()

# Lets defined hte optimizer and loss function
class_weights = torch.tensor([1.0, 14.0]).cuda()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
criterion = nn.CrossEntropyLoss(weight=class_weights)

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
  preds, ind= torch.max(F.softmax(preds, dim=-1), 1)
  correct = (ind == y).float()
  acc = correct.sum()/float(len(correct))
  return acc

# Lets define our training steps
def train(model, iterator, optimizer, criterion):
    
  epoch_loss = 0
  epoch_acc = 0
  
  model.train()
  for batch in iterator:
    
    optimizer.zero_grad()
            
    predictions = model(batch.Text).squeeze(0)
    loss = criterion(predictions, batch.Label)
    acc = binary_accuracy(predictions, batch.Label)
    
    loss.backward()
    
    optimizer.step()
    
    epoch_loss += loss.item()
    epoch_acc += acc.item()
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
  epoch_loss = 0
  epoch_acc = 0
  
  model.eval()
  
  with torch.no_grad():
    for batch in iterator:

      predictions = model(batch.Text).squeeze(0)
      
      loss = criterion(predictions, batch.Label)
      
      acc = binary_accuracy(predictions, batch.Label)

      epoch_loss += loss.item()
      epoch_acc += acc.item()
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

for epoch in range(N_EPOCHS):

  train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
  valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
  
  print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')


test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')

def predict_sentiment(sentence):
  tokenized = [tok for tok in sentence.split()]
  indexed = [TEXT.vocab.stoi[t] for t in tokenized]
  tensor = torch.LongTensor(indexed).to(device)
  
  tensor = tensor.unsqueeze(1)
  prediction = model(tensor)
  preds, ind= torch.max(F.softmax(prediction.squeeze(0), dim=-1), 1)
  return preds, ind

text = 'whats up fag bitch fuck off'
print(predict_sentiment(text)[1].item())

text = 'fuckoff fatass bitch'
print(predict_sentiment(text)[1].item())

text = 'nigger whats your problem'
print(predict_sentiment(text)[1].item())





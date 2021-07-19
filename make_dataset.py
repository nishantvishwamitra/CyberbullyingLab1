from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset, IterableDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import random
import os
import csv

def make_formspring_csv_data():
  with open('XMLMergedFile.xml', 'r') as f:
    data = f.read()

  Bs_data = BeautifulSoup(data, 'xml')

  POST = Bs_data.find_all('POST')

  cb_data = []
  non_cb_data = []

  for post in POST:
    text = post.find('TEXT')
    labeldata = post.find_all('LABELDATA')
    cb_count = 0
    for l in labeldata:
      ans = l.find('ANSWER').getText()
      if 'Yes' in ans:
        cb_count += 1

    text = text.getText()

    if cb_count >= 2:
      cb_data.append((text, 1))
    else:
      non_cb_data.append((text, 0))

  #dataset =  cb_data + non_cb_data[:900]
  dataset =  cb_data + non_cb_data
  random.shuffle(dataset)

  with open('formspring_dataset.csv', mode = 'w') as csv_file:
    fieldnames = ['text', 'label']
    writer = csv.DictWriter(csv_file, fieldnames = fieldnames, delimiter = '\t')

    writer.writeheader()
    for d in dataset:
      writer.writerow({'text': d[0], 'label': d[1]})

make_formspring_csv_data()




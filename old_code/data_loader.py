### Transformers for Humans vs AI Text Identification
### CAPP 30255
### Wesley Janson, Piper Kurtz, and Sam Pavlekovsky

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import numpy as np

# Load in data
data = pd.read_csv('data/data.csv')
clean_data = data[['intro','type']]

def load_data(csv):
  '''
  Reads the raw csv file and split into
  sentences (x) and target (y)
  '''
  df = pd.read_csv(csv)
  
  
  text = df['intro'].values
  labels = df['type'].values
  return labels,text


labels,text = load_data('data/data.csv')


# This function processes training data, establishing number IDs for each vocabulary word,
# converting word sequence into ID sequence (input_as_ids), and providing dict
# to map from word to its ID (word2id), and list to map from ID back to word (id2word)
def process_training_data(corpus_text):
        '''
        Tokenizes a text file.
        '''
        # Create the model's vocabulary and map to unique indices
        word2id = {}
        id2word = []
        list_of_inputs = []
        for entry in corpus_text:
            for word in entry:
                if word not in word2id:
                    id2word.append(word)
                    word2id[word] = len(id2word) - 1

            # Convert string of text into string of IDs in a tensor for input to model
            input_as_ids = []
            for word in entry.split():
                input_as_ids.append(word2id[word])
            list_of_inputs.append(input_as_ids)
            # final_ids = torch.LongTensor(input_as_ids)

        return list_of_inputs,word2id,id2word


tokenizer = get_tokenizer("basic_english")
list_of_tokens = [tokenizer(x) for x in text]
print(list_of_tokens)

list_of_inputs,word2id,id2word = process_training_data(list_of_tokens)




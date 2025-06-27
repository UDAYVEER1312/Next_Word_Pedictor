# Next_Word_Pedictor
This is a simple model that trains on general vocabulary sentences and general information and predicts the next word when a suitable input is passed
# Dataset
For training I used around 1000 sentences fetched from several websites and also used Grok.ai to generaet some of it.
The data is stored in general_speak.txt(There's an extra \n at the end of every sentence which is irrelevant, so during datapreprocessing we need to drop it)
# Model 
For better lerning i used Lstm architecture to predict due to it's Long-short term memory states that better finds the relationships between the words(tokens) in a sequence input.

The `LSTM` architecture :
1. Layer1 : An [embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html#embedding) layer of 150 dimensions (I used a high no. of dimensions for slow convergence but around 60-80 dimensions will offer faster learning)
2. Layer2 : A single Hidden layer of 250 neurons (No stacked layers)
3. Layer3 : The output layer (neurons = Vocab_size) 

# Preprcessing :
Essential libraries
```bash
import nltk
from nltk.tokenize import word_tokenize
import torch
nltk.download('punkt_tab')  # download punkt_tab/punkt for tokenization
```
--- output---
```bash
[nltk_data] Downloading package punkt_tab to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt_tab.zip.
True
```
Tokenizing the whole data : 
```bash
data = open('general_speak.txt','r').read()
data = data.replace('\\n','')  # remove unnecessary \n
tokens = word_tokenize(data.lower())
```

Creating vocabulary
```bash
from collections import Counter
vocab = {'<UNK>':0}   # for unknown word
count = Counter(tokens) # creates a dictionary , removes repeates tokens
for token in count.keys():
    vocab[token] = len(vocab)
len(vocab), len(count)
```
---output---
```bash
(1889, 1888)
```
Fetch each sentence and convert each token to index
```bash
def token_to_index(token:str,vocab):
    return [vocab[t] if t in vocab else vocab['<UNK>'] for t in word_tokenize(token.lower())]
sentences = data.split('\n')  # fetch all sentence
tokenized_sentences = []
for sentence in sentences:
    tokenized_sentences.append(token_to_index(sentence,vocab))
sentences[0],tokenized_sentences[0]
```
---output---
```bash
('The sun sets slowly behind the mountain, casting a warm glow over the valley.',
[1, 2, 3, 4, 5, 1, 6, 7, 8, 9, 10, 11, 12, 1, 13, 14])
```
Initialize the train sequence 
```bash
train_sequence = []
for ts in tokenized_sentences:
    for i in range(len(ts)):
        train_sequence.append(ts[:i+1])
train_sequence[:10] # first 10 train sequence
```
---output---
```bash---
[[1],
 [1, 2],
 [1, 2, 3],
 [1, 2, 3, 4],
 [1, 2, 3, 4, 5],
 [1, 2, 3, 4, 5, 1],
 [1, 2, 3, 4, 5, 1, 6],
 [1, 2, 3, 4, 5, 1, 6, 7],
 [1, 2, 3, 4, 5, 1, 6, 7, 8],
 [1, 2, 3, 4, 5, 1, 6, 7, 8, 9]]
```
Add zero-padding in the beginning of each tokenized_sentence
```bash
max_size = max(len(t) for t in train_sequence)
for i in range(len(train_sequence)):
    train_sequence[i] = [0 for i in range(max_size-len(train_sequence[i]))] + train_sequence[i]
print(train_sequence[10])
```
---output---
```bash
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 1, 6, 7, 8, 9, 10]
```
Set the train_sequence and target
    For each sequence the target is the last token/word
```bash
train_sequence = torch.tensor(train_sequence)
print(train_sequence.shape)
sequences , targets = train_sequence[:,:-1] , train_sequence[:,-1]
sequences[10], targets[10]
```
---output---
```bash
(tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3,
         4, 5, 1, 6, 7, 8, 9]),
 tensor(10))

```
# Training
Loss : `CrosssEntropyLoss`

Optimizer : `Adam`, learining_Rate = 0.001 (keep it small)

Run for 20 epochs and the average loss per batch was 0.66 which is quite good.
```bash
Epoch : 2 | train_loss : 3.0516974925994873
Epoch : 4 | train_loss : 1.9016647338867188
Epoch : 6 | train_loss : 1.2000373601913452
Epoch : 8 | train_loss : 0.871799647808075
Epoch : 10 | train_loss : 0.7572396397590637
Epoch : 12 | train_loss : 0.7110127806663513
Epoch : 14 | train_loss : 0.6853783130645752
Epoch : 16 | train_loss : 0.67520672082901
Epoch : 18 | train_loss : 0.6670853495597839
Epoch : 20 | train_loss : 0.6646623015403748
```

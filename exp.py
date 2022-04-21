from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import time
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import math
# import warnings filter
from warnings import simplefilter
from datasets import load_dataset

from sklearn.manifold import TSNE
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
#from parallel import DataParallelModel, DataParallelCriterion

import torch
import random
import torch
import pdb
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

import torch
import random
import torch

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def sentence_pairs_generation(sentences, labels, pairs):
	# initialize two empty lists to hold the (sentence, sentence) pairs and
	# labels to indicate if a pair is positive or negative

  numClassesList = np.unique(labels)
  idx = [np.where(labels == i)[0] for i in numClassesList]

  for idxA in range(len(sentences)):      
    currentSentence = sentences[idxA]
    label = labels[idxA]
    idxB = np.random.choice(idx[np.where(numClassesList==label)[0][0]])
    posSentence = sentences[idxB]
		  # prepare a positive pair and update the sentences and labels
		  # lists, respectively
    pairs.append(InputExample(texts=[currentSentence, posSentence], label=1.0))

    negIdx = np.where(labels != label)[0]
    negSentence = sentences[np.random.choice(negIdx)]
		  # prepare a negative pair of images and update our lists
    pairs.append(InputExample(texts=[currentSentence, negSentence], label=0.0))
  
	# return a 2-tuple of our image pairs and labels
  return (pairs)

def sst_2():
    train_df = pd.read_csv('data/SST-2/train.tsv', delimiter='\t')
    train_df, val_df = train_test_split(train_df,test_size=0.1,random_state=42)
    # Load the test dataset into a pandas dataframe.
    eval_df = pd.read_csv('data/SST-2/dev.tsv', delimiter='\t')
    label = "label"

    return train_df,eval_df,label



train_df,eval_df,label = sst_2()

text_col=train_df.columns.values[0] 
category_col=train_df.columns.values[1]

x_eval = eval_df[text_col].values.tolist()
y_eval = eval_df[category_col].values.tolist()

#@title SetFit
st_model = 'bert-base-uncased' #@param ['paraphrase-mpnet-base-v2', 'all-mpnet-base-v1', 'all-mpnet-base-v2', 'stsb-mpnet-base-v2', 'all-MiniLM-L12-v2', 'paraphrase-albert-small-v2', 'all-roberta-large-v1','sentence-transformers/bert-base-nli-mean-tokens']
num_training = 312 #@param ["8", "16", "32", "54", "128", "256", "512"] {type:"raw"}
num_itr = 10 #@param ["1", "2", "3", "4", "5", "10"] {type:"raw"}
plot2d_checkbox = False #@param {type: 'boolean'}

set_seed(0)
# Equal samples per class training
train_df_sample = pd.concat([train_df[train_df[label]==0].sample(num_training), train_df[train_df[label]==1].sample(num_training)])
x_train = train_df_sample[text_col].values.tolist()
y_train = train_df_sample[category_col].values.tolist()

train_examples = [] 
for x in range(num_itr):
  train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)


#word_embedding_model = models.Transformer(st_model)
#pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
#model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


orig_model = SentenceTransformer(st_model)
model = SentenceTransformer(st_model)
#orig_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])





# S-BERT adaptation 
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=30, show_progress_bar=True)

# No Fit
X_train_noFT = orig_model.encode(x_train)
X_eval_noFT = orig_model.encode(x_eval)

sgd =  LogisticRegression(max_iter=500)
sgd.fit(X_train_noFT, y_train)
y_pred_eval_sgd = sgd.predict(X_eval_noFT)

print('Acc. No Fit', accuracy_score(y_eval, y_pred_eval_sgd))

# With Fit (SetFit)
X_train = model.encode(x_train)
X_eval = model.encode(x_eval)


sgd =  LogisticRegression(max_iter=300)
sgd.fit(X_train, y_train)
y_pred_eval_sgd = sgd.predict(X_eval)

print('Acc. SetFit', accuracy_score(y_eval, y_pred_eval_sgd))




from __future__ import print_function
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import LSTM, Dropout, TimeDistributed, Dense
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from collections import defaultdict
import numpy as np
import IPython
#from load_data import load_refexps
import pdb
from keras.utils import to_categorical
import pickle

#sentences = load_refexps()
with open("annotations.pkl", "rb") as f:
    sentences = pickle.load(f)

X_indices = []
Y_indices = []
w2i = defaultdict(lambda: len(w2i))
w2i['<eos>'] = 0
w2i['<bos>'] = 1

#words = sorted(list(set(train_corpus)))
#words = '<eos>' + '<bos>' + words
#word_indices = dict((w, i) for i, w in enumerate(words))
#indices_word = dict((i, w) for i, w in enumerate(words))
#IPython.embed()

num_sent = 2000

print("Sentences: " + str(len(sentences)))
sentences = sentences[:num_sent]
print("Using " + str(num_sent) + " sentences")
#sentences = [['Aa', 'Bb', 'Cc'], ['Bb', 'Cc', 'Dd']]

print("Vectorization...")
for sentence in sentences:
    Y_indices.append([w2i[word] for word in sentence])
    sentence = ['<bos>'] + sentence
    X_indices.append([w2i[word] for word in sentence])

print("Padding...")
#maxlen = 10
#vocab_size = 1000
maxlen = max([len(sentence) for sentence in sentences]) + 1
vocab_size = len(w2i)

padded_X = sequence.pad_sequences(X_indices, maxlen=maxlen, padding='post', dtype='int32', value=w2i['<eos>'])
padded_Y = sequence.pad_sequences(Y_indices, maxlen=maxlen, padding='post', dtype='int32', value=w2i['<eos>'])
padded_Y_np = np.zeros((len(padded_Y), maxlen, vocab_size))
for i, y in enumerate(padded_Y):
        padded_Y_np[i] = to_categorical(y, vocab_size)
padded_Y = padded_Y_np

#print('Vectorization...')
#X = np.zeros((len(sentences), maxlen, len(w2i)), dtype=np.bool)
#Y = np.zeros((len(sentences), maxlen, len(w2i)), dtype=np.bool)
#X = np.zeros((len(padded_sentences), maxlen))
#Y = np.zeros((len(padded_sentences), maxlen))
#X = []
#Y = []
#for i, sentence in enumerate(padded_sentences):
#    for t in range(len(sentence)-1):
#        X[i, t, w2i[sentence[t]]] = 1
#        Y[i, t, w2i[sentence[t+1]]] = 1
#    X[i, len(sentence), w2i[sentence[-1]]] = 1
#    Y[i, len(sentence), w2i['<eos>']] = 1

# build the model: a single LSTM
print('Build model...')
#IPython.embed()
#pdb.set_trace()
model = Sequential()
model.add(Embedding(vocab_size, 1024, input_length=maxlen))
model.add(Dropout(0.5))
model.add(LSTM(1024, input_shape=(maxlen, 1024), return_sequences=True, dropout=0.5))
#model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

optimizer = RMSprop(lr=0.01, clipnorm=10.)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# train the model, output generated text after each iteration
for iteration in range(1, 5):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    loss = model.fit(padded_X, padded_Y, batch_size=16, epochs=1)
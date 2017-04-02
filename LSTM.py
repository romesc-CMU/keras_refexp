from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from collections import defaultdict
import numpy as np
#import random
#import sys
#import IPython
from load_data import load_refexps

# train_refexp_list = load_refexps()
train_corpus = []
sentences = load_refexps()
sentence_indices = []
w2i = defaultdict(lambda: len(w2i))
w2i['<eos>'] = 0

for sentence in sentences:
    sentence = ['<bos>'] + sentence
    sentence_indices.append([w2i[word] for word in sentence])
    #train_corpus += train_refexp

#words = sorted(list(set(train_corpus)))
#words = '<eos>' + '<bos>' + words
#word_indices = dict((w, i) for i, w in enumerate(words))
#indices_word = dict((i, w) for i, w in enumerate(words))
#IPython.embed()

print("Sentences" + str(len(sentences)))

maxlen = max([len(sentence) for sentence in sentences])
padded_sentences = sequence.pad_sequence(sentences, maxlen=maxlen, padding='post', dtype='int32', value=w2i['<eos>'])

print('Vectorization...')
#X = np.zeros((len(sentences), maxlen, len(w2i)), dtype=np.bool)
#Y = np.zeros((len(sentences), maxlen, len(w2i)), dtype=np.bool)
X = np.zeros((len(sentences), maxlen))
Y = np.zeros((len(sentences), maxlen))
for i, sentence in enumerate(sentences):
    for t in range(len(sentence)-1):
#        X[i, t, w2i[sentence[t]]] = 1
#        Y[i, t, w2i[sentence[t+1]]] = 1
        X[i, t] = w2i[sentence[t]]
        Y[i, t] = w2i[sentence[t + 1]]
    X[i, len(sentence)] = w2i[sentence[-1]]
    Y[i, len(sentence)] = w2i['<eos>']
#    X[i, len(sentence), w2i[sentence[-1]]] = 1
#    Y[i, len(sentence), w2i['<eos>']] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Embedding(len(w2i), 1024, input_length=maxlen))
model.add(LSTM(16, input_shape=(maxlen, 1024)))
model.add(Dense(len(w2i)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# def sample(preds, temperature=1.0):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    loss = model.fit(X, Y, batch_size=16, nb_epoch=1)
    print(np.exp(2,loss/[len(sent) for sent in sentences]))
#     start_index = random.randint(0, len(text) - maxlen - 1)
#
#     for diversity in [0.2, 0.5, 1.0, 1.2]:
#         print()
#         print('----- diversity:', diversity)
#
#         generated = ''
#         sentence = text[start_index: start_index + maxlen]
#         generated += sentence
#         print('----- Generating with seed: "' + sentence + '"')
#         sys.stdout.write(generated)
#
#         for i in range(400):
#             x = np.zeros((1, maxlen, len(chars)))
#             for t, char in enumerate(sentence):
#                 x[0, t, char_indices[char]] = 1.
#
#             preds = model.predict(x, verbose=0)[0]
#             next_index = sample(preds, diversity)
#             next_char = indices_char[next_index]
#
#             generated += next_char
#             sentence = sentence[1:] + next_char
#             sys.stdout.write(next_char)
#             sys.stdout.flush()
#         print()
#
#

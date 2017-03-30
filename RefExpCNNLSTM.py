import numpy as np
import scipy.io
import sys
import argparse

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.layers.recurrent import LSTM
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, RemoteMonitor

from sklearn.externals import joblib
from sklearn import preprocessing

from spacy.en import English

from utils import grouper, selectFrequentAnswers
from features import get_images_matrix, get_answers_matrix, get_questions_tensor_timeseries
import ipdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_hidden_units_mlp', type=int, default=1024)
    parser.add_argument('-num_hidden_units_lstm', type=int, default=512)
    parser.add_argument('-num_hidden_layers_mlp', type=int, default=3)
    parser.add_argument('-num_hidden_layers_lstm', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-activation_mlp', type=str, default='tanh')
    parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-model_save_interval', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=128)
    #TODO Feature parser.add_argument('-resume_training', type=str)
    #TODO Feature parser.add_argument('-language_only', type=bool, default= False)
    args = parser.parse_args()


    #Retrieve and Format Data
    #TODO
    

    #====================== Define Model 1 ======================
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(ConvolutionalD(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))



    #====================== Define Model 2 ======================
    image_model = Sequential()
    image_model.add(Reshape( (img_dim, ), input_shape = (img_dim,)))


    language_model = Sequential()
    if args.num_hidden_layers_lstm == 1:
        language_model.add(LSTM(output_dim = args.num_hidden_units_lstm, return_sequences=False, input_shape=(None, word_vec_dim)))
        #language_model.add(LSTM(output_dim = args.num_hidden_units_lstm, return_sequences=False, input_shape=(max_len, word_vec_dim)))
    else:
        language_model.add(LSTM(output_dim = args.num_hidden_units_lstm, return_sequences=True, input_shape=(max_len, word_vec_dim)))
        for i in xrange(args.num_hidden_layers_lstm-2):
            language_model.add(LSTM(output_dim = args.num_hidden_units_lstm, return_sequences=True))
        language_model.add(LSTM(output_dim = args.num_hidden_units_lstm, return_sequences=False))

    model = Sequential()
    model.add(Merge([language_model, image_model], mode='concat', concat_axis=1))
    for i in xrange(args.num_hidden_layers_mlp):
        model.add(Dense(args.num_hidden_units_mlp, init='uniform'))
        model.add(Activation(args.activation_mlp))
        model.add(Dropout(args.dropout))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))




    #====================== Saving Model ======================
    json_string = model.to_json()
    model_file_name = '../models/lstm_1_num_hidden_units_lstm_' + str(args.num_hidden_units_lstm) + \
                        '_num_hidden_units_mlp_' + str(args.num_hidden_units_mlp) + '_num_hidden_layers_mlp_' + \
                        str(args.num_hidden_layers_mlp) + '_num_hidden_layers_lstm_' + str(args.num_hidden_layers_lstm)
    open(model_file_name + '.json', 'w').write(json_string)





    #====================== Compile Model 1 ======================

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    
    #====================== Compile Model 2 ======================
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print 'Compilation done'

    features_struct = scipy.io.loadmat(vgg_model_path)
    VGGfeatures = features_struct['feats']
    print 'loaded vgg features'
    image_ids = open('../features/coco_vgg_IDMap.txt').read().splitlines()
    img_map = {}
    for ids in image_ids:
        id_split = ids.split()
        img_map[id_split[0]] = int(id_split[1])

    nlp = English()
    print 'loaded word2vec features...'
    ## training
    print 'Training started...'



    #====================== Train Model ======================
    for k in xrange(args.num_epochs):

        progbar = generic_utils.Progbar(len(questions_train))

        for qu_batch,an_batch,im_batch in zip(grouper(questions_train, args.batch_size, fillvalue=questions_train[-1]), 
                                                grouper(answers_train, args.batch_size, fillvalue=answers_train[-1]), 
                                                grouper(images_train, args.batch_size, fillvalue=images_train[-1])):
            timesteps = len(nlp(qu_batch[-1])) #questions sorted in descending order of length
            #trained with smallest question first (trick for training LSTMs)
            X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, timesteps)
            #ipdb.set_trace()
            X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
            Y_batch = get_answers_matrix(an_batch, labelencoder)
            loss = model.train_on_batch([X_q_batch, X_i_batch], Y_batch)
            progbar.add(args.batch_size, values=[("train loss", loss)])

        
        if k%args.model_save_interval == 0:
            model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k))

    model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k))
    
if __name__ == "__main__":
    main()

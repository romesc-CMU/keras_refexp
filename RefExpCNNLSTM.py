import sys
import os
#sys.path.insert(0, r'/rscalise/storage/refexp/Google_Refexp_toolbox/')
import numpy as np
import cv2
import skimage.io as io
import argparse
import operator

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.layers.recurrent import LSTM
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, RemoteMonitor

# Pre-Trained VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image


from sklearn.externals import joblib
from sklearn import preprocessing

from spacy.en import English
from refexp.Google_Refexp_toolbox.google_refexp_py_lib.refexp import Refexp 

#from Google_Refexp_toolbox.google_refexp_py_lib.refexp import Refexp

# Specify refexp dataset paths
refexp_filename='/home/rscalise/storage/refexp/Google_Refexp_toolbox/google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'
coco_filename='/home/rscalise/storage/refexp/Google_Refexp_toolbox/external/coco/annotations/instances_train2014.json'
imagesDir='/home/rscalise/storage/refexp/Google_Refexp_toolbox/external/coco/images'
imagesType='train2014'

# Create instance of Refexp
refexp = Refexp(refexp_filename, coco_filename)

#from utils import grouper, selectFrequentAnswers
#from features import get_images_matrix, get_answers_matrix, get_questions_tensor_timeseries
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


    # Retrieve and Format Data

    catIds = refexp.getCatIds(catNms=['dog'])
    imgIds = refexp.getImgIds(catIds=catIds)

    random_img_id = imgIds[np.random.randint(0,len(imgIds))]
    img_struct = refexp.loadImgs(random_img_id)[0]
    I = io.imread(os.path.join(imagesDir, imagesType, img_struct['file_name']))

    I_sz = I.shape[0]*I.shape[1] #interpreted 'size' as area since it is indicated as a scalar
    I_asp_ratio = float(I.shape[0])/float(I.shape[1])
    

    #====================== Define Model 1 ======================
    
    # Note: We are using the full model (include_top)  since the paper indicates they use the 'last 1000 dimensional layer of VGGNet'
    image_model = VGG16(weights='imagenet', include_top=True)
     
    # Truncates the model to the 3rd layer
    #model = Model(input=image_model.input, output=image_model.get_layer('block3_conv3').output)

    for R in refexp.getRegionCandidates(img_struct):
        W = R[2]
        H = R[3]
        x_tl = max(R[0], 0)
        y_tl = max(R[1], 0)
        x_br = x_tl+W
        y_br = y_tl+H
        I_R_sz = W*H 
        I_R_asp_ratio = W/H

        # Crop I to candidate region I_R

        I_R = I[y_tl:y_br, x_tl:x_br]

        I_R_locSz_vec = [float(x_tl)/float(W), float(y_tl)/float(H), float(x_br)/float(W), float(y_br)/float(H), float(I_R_sz)/float(I_sz) ]
        


    #==== Preprocess I and I_R

    # compute mean channel values
    I_mean_vals_per_channel = np.floor(np.average(np.average(I, axis=0), axis=0))
    I_R_mean_vals_per_channel = np.floor(np.average(np.average(I_R, axis=0), axis=0))
    

    # squarifying
    I_pad = padImageSquare(I, I_mean_vals_per_channel)
    I_R_pad = padImageSquare(I_R, I_R_mean_vals_per_channel)

    # preprocessing
    ipdb.set_trace()
    I_pp = cv2.resize(I_pad, (224, 224)).astype(np.float32)
    ipdb.set_trace()
    I_pp[:,:,0] -= I_mean_vals_per_channel[0] #subtract mean vals
    I_pp[:,:,1] -= I_mean_vals_per_channel[1]
    I_pp[:,:,2] -= I_mean_vals_per_channel[2]
    #I_pp = I_pp.transpose((2,0,1)) #ordering for theano backend
    I_pp = np.expand_dims(I_pp, axis=0)
    I_R_pp = cv2.resize(I_R_pad, (224, 224)).astype(np.float32)
    I_R_pp[:,:,0] -= I_R_mean_vals_per_channel[0] #subtract mean vals
    I_R_pp[:,:,1] -= I_R_mean_vals_per_channel[1]
    I_R_pp[:,:,2] -= I_R_mean_vals_per_channel[2]
    #I_R_pp = I_R_pp.transpose((2,0,1)) #ordering for theano backend
    I_R_pp = np.expand_dims(I_R_pp, axis=0)




    #==== Generate Representations

    I_feats = image_model.predict(I_pp)
    I_R_feats = image_model.predict(I_R_pp)


    #==== Testing/Visualization

    #import keras.applications.imagenet_utils as ka
    #print ka.decode_predictions(I_feats)
    #plt.imshow(I_pad)
    #plt.show()
    #ipdb.set_trace()

    #print ka.decode_predictions(I_R_feats)
    #plt.imshow(I_R_pad)
    #plt.show()
    #ipdb.set_trace()

    #==== Concatentate to 2005-dimensional vector for input to LSTM
    image_rep_vec = [I_feats, I_R_feats, I_R_locSz_vec]

    ipdb.set_trace()

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

def padImageSquare(img, mean_channel_vals):
    pad_color = mean_channel_vals 

    # Find the smaller dimension index and size
    img_dims = [img.shape[0], img.shape[1]]
    min_idx, min_val = min(enumerate(img_dims), key=operator.itemgetter(1))
    max_idx, max_val = max(enumerate(img_dims), key=operator.itemgetter(1))
    diff = max_val - min_val
    #fix this so that I_pad is always square
    if diff%2 != 0:
        diff += 1

    pad_num = diff/2


    # Pad the smaller dimension
    if min_idx == 0:
        img_padded = cv2.copyMakeBorder(img, pad_num, pad_num, 0, 0, cv2.BORDER_CONSTANT, value=pad_color) 
    else:
        img_padded = cv2.copyMakeBorder(img, 0, 0, pad_num, pad_num, cv2.BORDER_CONSTANT, value=pad_color) 

    return img_padded
    
if __name__ == "__main__":
    main()

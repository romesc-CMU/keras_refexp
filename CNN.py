from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import Model
import numpy as np
import pdb
import os
import cPickle
base_model = vgg16.VGG16(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block3_conv3').output)
files = open('file_name').readlines()
imglist = list()
count = 1
os.mkdir('block3')
cc = 0



for fil in files:
    img_path = '/home/sdalmia/11777/test2014/'  + fil.strip()
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    imglist.append(x)
    if count % 500 == 0:
        Xs = np.array(imglist)
        Xs = vgg16.preprocess_input(Xs)
        features = model.predict(Xs)
        imglist=list()
        features = features.reshape((500,-1))
        # pdb.set_trace( )
        cPickle.dump(features,open('/home/sdalmia/11777/block3/block3_'+str(cc)+'.pkl','w'))
        cc +=1
        print 'Done '  , cc
    count +=1
Xs = np.array(imglist)
Xs = vgg16.preprocess_input(Xs)
features = model.predict(Xs)
features = features.reshape((features.shape[0],-1))
cPickle.dump(features,open('/home/sdalmia/11777/block3/block3_'+str(cc)+'.pkl','w'))
cc +=1
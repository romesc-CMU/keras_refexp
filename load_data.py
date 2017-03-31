import scipy
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
import matplotlib.patches as mpatches
import sys
import os
#import IPython

# Import Refexp python class
# Please MAKE SURE that ./google_refexp_py_lib is in your
# python library search path
path_prefix = "/projects/tir2/users/cmalaviy/11777/Google_Refexp_toolbox/"
sys.path.append(path_prefix + "google_refexp_py_lib")
from refexp import Refexp

def load_refexps():
    # Specify datasets path.
    refexp_filename = path_prefix + 'google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'
    coco_filename = path_prefix + 'external/coco/annotations/instances_train2014.json'
    imagesDir = path_prefix + 'external/coco/images'
    imagesType = path_prefix + 'train2014'

    # Create Refexp instance.
    refexp = Refexp(refexp_filename, coco_filename)
    img_ids = refexp.getImgIds()
    refexp_list = []

    for img_id in img_ids[:3]:
        anns = refexp.getAnnIds(img_id)
        ann = refexp.loadAnns(anns[0])[0]
        #IPython.embed()
        refexp_text = refexp.loadRefexps(ann['refexp_ids'][0])[0]['raw']
        refexp_list.append(refexp_text.encode('utf8')) 

    return refexp_list

#---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    load_refexps()

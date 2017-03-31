import scipy
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
import matplotlib.patches as mpatches
import sys
import os

# Import Refexp python class
# Please MAKE SURE that ./google_refexp_py_lib is in your
# python library search path
sys.path.append("google_refexp_py_lib")
from refexp import Refexp

# Specify datasets path.
refexp_filename='google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'
coco_filename='external/coco/annotations/instances_train2014.json'
imagesDir = 'external/coco/images'
imagesType = 'train2014'

# Create Refexp instance.
refexp = Refexp(refexp_filename, coco_filename)
img_ids = refexp.getImgIds()

for img_id in img_ids[:3]:
    anns = refexp.getAnnIds(img_id)
    ann = refexp.loadAnns(anns[0])[0]
    print ann

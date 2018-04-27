import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import sys
import numpy as np

datadir = sys.argv[1]
model = load_model(datadir+"/CNN/temp_model.h5")

model.layers.pop()
model.layers.pop()

model.build()

model.save(datadir+"/CNN/temp_model-Feature.h5")

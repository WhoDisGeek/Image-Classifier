from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
import numpy as np

model = load_model("./Data/CNN/CNN-Inter.h5")

model.layers.pop()
model.layers.pop()

model.build()

model.save("./Data/CNN/CNN-Inter-Feature.h5")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import numpy as np
import sys

datadir = sys.argv[1]
model = load_model(datadir+"/CNN/temp_model.h5")

rootdir = datadir+'/Test'
total = 0
correct = 0

with open(datadir+'/Classes.txt') as f:
    classes = f.read().splitlines()

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)
        #print(path)
        total = total + 1
        print(total)
        image = cv2.imread(path,0)
        image = cv2.resize(image, (128, 128))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)

        label = np.argmax(predictions[0])
        if classes[label] in path:
            correct = correct + 1

print("Accuracy: "+str(correct)+"/"+str(total)+"[ "+str((correct*100)/total)+" ]")

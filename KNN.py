from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
import numpy as np
import sys
from numpy import dot
from numpy.linalg import norm
import operator

model = load_model("./Data/CNN/CNN-Inter-Feature.h5")

rootdir = './Data/Vectors'

with open('./Data/Classes.txt') as f:
    classes = f.read().splitlines()

image = cv2.imread(sys.argv[1],0)
image = cv2.resize(image, (256, 256))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
prediction = model.predict(image)
prediction = prediction[0]
total = 0
vectors = {}

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)
        #print(path)
        end = file.find('.')
        filename = file[0:end]
        total = total + 1
        print(total)
        
        with open(path) as f:
            lines = f.read().splitlines()

        for x in range(0,len(lines)):
            lines[x] = float(lines[x])
        for x in classes:
            if x in path:
                vectors[x+"/"+file] = dot(prediction, lines)/(norm(lines))

sortvector = sorted(vectors.items(), key=operator.itemgetter(1), reverse=True)
i = 0
finish = "none"

race = {}
for x in classes:
    race[x] = set()
    
while i < len(sortvector) and finish == "none":
    for x in classes:
        if x in sortvector[i][0]:
            end = sortvector[i][0].find('-')
            start = sortvector[i][0].find('/') + 1
            race[x].add(sortvector[i][0][start:end])
            if len(race[x]) == 10:
                finish = x
    i = i+1

filelist = []
imagedir = './Data/Images'
for subdir, dirs, files in os.walk(imagedir):
    for file in files:
        path = os.path.join(subdir, file)
        filelist.append(path)

for x in race[finish]:
    path = finish+"/"+x
    for y in filelist:
        if path in y:
            print(y)

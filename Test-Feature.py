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
from numpy import dot
from numpy.linalg import norm
import operator
datadir = sys.argv[1]

model = load_model(datadir+"/CNN/temp_model-Feature.h5")

vectordir = datadir+'/Vectors'
rootdir = datadir+'/Test/flower'
total = 0
correct = 0

with open(datadir+'/Classes.txt') as f:
	classes = f.read().splitlines()

for subdir, dirs, files in os.walk(rootdir):
	for file in files:
		path = os.path.join(subdir, file)
		total = total + 1
		print(total)
		image = cv2.imread(path,0)
		image = cv2.resize(image, (128, 128))
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		predictions = model.predict(image)
		vectors = {}
	
		for vectorsubdir, vectordirs, vectorfiles in os.walk(vectordir):
			for vectorfile in vectorfiles:
				vectorpath = os.path.join(vectorsubdir, vectorfile)
				end = vectorfile.find('.')
				filename = vectorfile[0:end]
				with open(vectorpath) as f:
					lines = f.read().splitlines()

				for x in range(0,len(lines)):
					lines[x] = float(lines[x])
				for x in classes:
					if x in vectorpath:
						vectors[x+"/"+vectorfile] = dot(predictions, lines)/(norm(lines))

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
	
		if finish in path:
			correct = correct + 1
			print(path+'  =>  '+finish)
			print('correct : '+str(correct)+'/'+str(total)+' : '+str(correct/total)+'%')
		else:
			print(path+'  =>  '+finish)
			print('incorrect : '+str(correct)+'/'+str(total)+' : '+str(correct/total)+'%')
print("Accuracy: "+str(correct)+"/"+str(total)+"[ "+str((correct*100)/total)+" ]")



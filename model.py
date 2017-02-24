# Import necessary library
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Build loadcsv function
def loadcsv (samples, path):
	with open(path) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
	return samples

# Build function to split data
def split (samples, test=0.1, validation=0.2):
	# Split data to training and test set
	train_set, test_set = train_test_split(samples, test_size=test)
	# Split data to training and test set
	train_samples, validation_samples = train_test_split(train_set, test_size=validation)
	return test_set, train_samples, validation_samples

def preprocess(img_path, gray = False):
	if gray == True:
		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	else:
		image = cv2.imread(img_path)
	cropped_img = image[60:140, 0:320] # trim image to only see section with road
	resized_img = cv2.resize(cropped_img, (w, h) , interpolation = cv2.INTER_AREA)
	return resized_img


def generator(samples, batch_size=32, gray = False):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				for i in range(3):
					local_path = batch_sample[i]
					image = preprocess(local_path, gray)
					# Flip images
					flipped_image = cv2.flip(image, 1)
					images.append(image)
					images.append(flipped_image)				
				correction = 0.16
				# Load steering angle
				angle = float(batch_sample[3])
				# Flip steering angle
				flipped_angle = angle * -1.0				
				angles.append(angle)
				angles.append(flipped_angle)
				angles.append(angle+correction)
				angles.append((angle+correction) * -1.0)
				angles.append(angle-correction)
				angles.append((angle-correction) * -1.0)
			X_train = np.asarray(images)
			y_train = np.asarray(angles)
			if gray == True:
			    X_train = X_train.reshape(X_train.shape[0], h, w, 1)
			yield shuffle(X_train, y_train)


samples = []
csvpath1 = './data/driving_log.csv' # my personal driving record
csvpath2 = './data0/driving_log_0.csv' # Udacity sample data filtered 0 steering images
ifgray = False
w, h, ch = 64, 16, 3 # resized image widthxheight
epoch = 5
modelname = "model.h5"


samples = loadcsv(samples, path=csvpath2)
test_set, train_samples, validation_samples = split(samples)
print("Data Ready")
print("Train Set: " + str(len(train_samples)))
print("Validation Set: " + str(len(validation_samples)))
print("Test set:" + str(len(test_set)))

images_test = []
angles_test = []

for line in test_set:
	name = line[0]           
	# Load Grayscaled image
	image = preprocess(name)  
	# Load steering angle
	angle = float(line[3])
	images_test.append(image)
	angles_test.append(angle)
X_test = np.array(images_test)
y_test = np.array(angles_test)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Import Keras
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Build model architecture
def buildmodel():
	model = Sequential()
	model.add(Lambda(lambda x: (x/255.0 - 0.5), input_shape=(h, w, ch)))
	model.add(Convolution2D(32, 5, 5, activation='relu', name='cnn_1'))
	model.add(MaxPooling2D((2,2)))
	model.add(Convolution2D(64, 3, 3, activation = 'relu', name='cnn_2'))
	model.add(MaxPooling2D((2,2)))
	model.add(Convolution2D(128, 1, 5, activation = 'relu', name='cnn_3'))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(50, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer= 'adam')
	print("Finish building")
	return model

model = buildmodel()

history_object = model.fit_generator(train_generator, samples_per_epoch =
	len(train_samples)*6, validation_data = 
	validation_generator,
	nb_val_samples = len(validation_samples)*6, 
	nb_epoch=epoch, verbose=1)

model.save(modelname)
	
print("Testing...")
test_loss = model.evaluate(X_test, y_test, batch_size=1)
print(test_loss)


#Here I used all the training samples, to train a CNN with the following 
#characteristics

#40*98 log mel features

# In[]: 
from os import listdir
from os.path import isdir, join
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# In[]: 
# Create list of all targets (minus background noise)
#dataset_path = 'C:/Users/lenovo/Desktop/speech_project/dataset'
dataset_path = '/nfsd/hda/DATASETS/Project_1'
all_targets = all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]

print(all_targets)
#30 classes after removing background noise 

# In[]: 
# Settings
#feature_sets_path = 'C:/Users/lenovo/Desktop/speech_project'
feature_sets_path = '/nfsd/hda/elkhaloufi'
feature_sets_filename = 'logmel.npz'
#model_filename = 'model1(CNN).h5'
#wake_word = 'stop' #we want to include all words


# In[]: 
# Load feature sets
feature_sets = np.load(join(feature_sets_path, feature_sets_filename))
print(feature_sets.files)

# In[]: 
# Assign feature sets
x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']
x_test = feature_sets['x_test']
y_test = feature_sets['y_test']

# In[]: 
# Look at tensor dimensions
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
# In[]: 
# Peek at labels
print(y_val) #a simple array of target vars 
# In[]: 
# View the dimensions of our input data
print(x_train.shape)
# In[]: 
# CNN for TF expects (batch, height, width, channels) #this is important
# So we reshape the input tensors with a "color" channel of 1
x_train = x_train.reshape(x_train.shape[0], 
                          x_train.shape[1], 
                          x_train.shape[2], 
                          1)
x_val = x_val.reshape(x_val.shape[0], 
                      x_val.shape[1], 
                      x_val.shape[2], 
                      1)
x_test = x_test.reshape(x_test.shape[0], 
                        x_test.shape[1], 
                        x_test.shape[2], 
                        1)
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
#What does channel mean... Can we change it? 

# In[]: 
# Input shape for CNN is size of MFCC of 1 sample
sample_shape = x_test.shape[1:]
print(sample_shape)
# In[]: 
# Build model 
tf.random.set_seed(1234)
model = models.Sequential()
model.add(layers.Conv2D(16, kernel_size=(3, 3), 
                        activation='elu',
                        padding='same',
                        input_shape=sample_shape))
model.add(layers.Conv2D(16, kernel_size=(3, 3), 
                        activation='elu',
                        padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='elu'))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='elu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=2))

#LSTM
model.add(layers.Reshape((-1, 32)))
model.add(layers.LSTM(32,return_sequences=False))

# Classifier
model.add(layers.Dense(64, activation='elu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(36, activation='softmax'))
# In[]: 
# Display model
model.summary()

# In[]: 
# Add training parameters to model
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'])

# In[]:
# Train
history = model.fit(x_train, 
                    y_train, 
                    epochs=30, 
                    batch_size=100, 
                    validation_data=(x_val, y_val))

# In[]:
# Evaluate model with test set
ev=model.evaluate(x=x_test, y=y_test)
print("This is the evaluation:  ",ev)
#saving model
np.save('hist_vgg_lstmelu30.npy',history.history)
#saving to h5
models.save_model(model,'vgg_lstmelu30.h5')

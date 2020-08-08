from os import listdir
from os.path import isdir, join
import librosa
import random
import numpy as np
import matplotlib.pyplot as plt
import python_speech_features



# Dataset path and view possible targets
dataset_path = '/nfsd/hda/DATASETS/Project_1'
for name in listdir(dataset_path):
    if isdir(join(dataset_path, name)):
        print(name)


# In[3]:


# Create an all targets list
all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
print(all_targets)


# In[5]:


# See how many files are in each target folder
num_samples = 0
for target in all_targets:
    print(len(listdir(join(dataset_path, target))))
    num_samples += len(listdir(join(dataset_path, target)))
print('Total samples:', num_samples)


# In[22]:


# Settings
target_list = all_targets
feature_sets_file = 'logmel.npz'
perc_keep_samples = 1.0 # 1.0 is keep all samples #YASMINE CHANGE THIS IF YOU WANT LESS TRAINING TIME WITH LESS SAMPLES
val_ratio = 0.1 #feel free to change
test_ratio = 0.1 #feel free to change
sample_rate = 8000
#num_mfcc = 40 #we only consider 16 features (MFCCs) we can change this number
#we drop the samples that generate more than 16 features 
len_mfcc = 98


# In[7]:


# Create list of filenames along with ground truth vector (y)
filenames = []
y = [] #list of 30 arrays where each array's length is the number of files for that specific target.
for index, target in enumerate(target_list):
    print(join(dataset_path, target))
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)


# In[8]:


# Check ground truth Y vector
print(y)
for item in y:
    print(len(item))


# In[9]:


# Flatten filename and y vectors
filenames = [item for sublist in filenames for item in sublist] #all file names in one list
y = [item for sublist in y for item in sublist]


# In[10]:


print(len(filenames)) #total number of files (samples)
print(len(y)) #corresponding labels for each file 0,0,0,0,0,0,1,1,1,1,1,1,1..... till 29 len( 1s )=1713 which is num of samples
#they have the same length 
# ID  Label


# In[11]:


# Associate filenames with true output and shuffle
filenames_y = list(zip(filenames, y)) #less9i target m3a l sample
random.shuffle(filenames_y) #mixiiiii meziaaan !!
filenames, y = zip(*filenames_y) #3awd fer9ihoum ( ra nefss l index y[5] hiya target dyal filenames[5])


# In[12]:


# Only keep the specified number of samples (shorter extraction/training)
print(len(filenames))
filenames = filenames[:int(len(filenames) * perc_keep_samples)] #YASMINE Change value of perc_keep_samples to select less samples
print(len(filenames))


# In[13]:


# Calculate validation and test set sizes
val_set_size = int(len(filenames) * val_ratio) #ila bghiti tbedli ratios mezian :y
test_set_size = int(len(filenames) * test_ratio)


# In[14]:


print(val_set_size)
print(test_set_size)


# In[15]:


# Break dataset apart into train, validation, and test sets
filenames_val = filenames[:val_set_size]
filenames_test = filenames[val_set_size:(val_set_size + test_set_size)]
filenames_train = filenames[(val_set_size + test_set_size):]


# In[16]:


print(filenames[0]) #one sample ye3ni filenames includes all samples


# In[17]:


# Break y apart into train, validation, and test sets
y_orig_val = y[:val_set_size]
y_orig_test = y[val_set_size:(val_set_size + test_set_size)]
y_orig_train = y[(val_set_size + test_set_size):]


# In[25]:


# Function: Create MFCC from given path
def calc_mfcc(path):
    
    # Load wavefile
    signal, fs = librosa.load(path, sr=sample_rate)
    
    # Create MFCCs from sound clip
   
    logmel= python_speech_features.base.logfbank(signal, 
                                         samplerate=fs, 
                                         winlen=0.030, 
                                         winstep=0.01, 
                                         nfilt=40, 
                                         nfft=2048, 
                                         lowfreq=0, 
                                         highfreq=None, 
                                         preemph=0.97)
    #return logmel.transpose()[:,:32]
    return logmel.transpose()


# Function: Create MFCCs, keeping only ones of desired length
def extract_features(in_files, in_y):
    prob_cnt = 0
    out_x = []
    out_y = []
        
    for index, filename in enumerate(in_files):
    
        # Create path from given filename and target item
        path = join(dataset_path, target_list[int(in_y[index])], 
                    filename)
        
        # Check to make sure we're reading a .wav file
        if not path.endswith('.wav'):
            continue

        # Create MFCCs
        mfccs = calc_mfcc(path)

        # Only keep MFCCs with given length
        if mfccs.shape[1] == len_mfcc:
            out_x.append(mfccs)
            out_y.append(in_y[index])
        else:
            print('Dropped:', index, mfccs.shape)
            prob_cnt += 1
            
    return out_x, out_y, prob_cnt


# In[29]:


# Create train, validation, and test sets
x_train, y_train, prob = extract_features(filenames_train, 
                                          y_orig_train)
print('Removed percentage:', prob / len(y_orig_train)) #removed percentage in train set: 0.08559950556242274
x_val, y_val, prob = extract_features(filenames_val, y_orig_val)
print('Removed percentage:', prob / len(y_orig_val)) #Removed percentage in val set: 0.08559950556242274
x_test, y_test, prob = extract_features(filenames_test, y_orig_test)
print('Removed percentage:', prob / len(y_orig_test))


# In[30]:


# Save features and truth vector (y) sets to disk
np.savez(feature_sets_file, 
         x_train=x_train, 
         y_train=y_train, 
         x_val=x_val, 
         y_val=y_val, 
         x_test=x_test, 
         y_test=y_test)

#so now we have created a file .NPZ in our directory


# In[32]:


# TEST: Load features
feature_sets = np.load(feature_sets_file)
feature_sets.files


feature_sets['x_train'].shape



len(feature_sets['x_train'])

print(len(feature_sets['y_train']))

print(feature_sets['y_val'])

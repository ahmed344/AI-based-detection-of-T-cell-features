# Libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage import io

# Tensorflow
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout

# Utilities
from pattern import Binary_Squares
from binary import Binary_Images




class Binary_Classifier:
    
    def __init__(self, X_train, X_test, y_train, y_test):
        
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
    
    
    def classifier_3_conv2d(self, dim, f = 64, ks = (3, 3), droprate = 0.1):
    
        model = Sequential()
        model.add(Conv2D(f, ks, input_shape = dim, activation = 'relu', padding='same'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(droprate))

        model.add(Conv2D(2*f, ks, activation = 'relu', padding='same'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(droprate))  

        model.add(Conv2D(4*f, ks, activation = 'relu', padding='same'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(droprate))  

        model.add(Flatten())

        #model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    
    def classify(self, no_of_epochs = 300):
    
        # Prepare binary labeled training and test samples
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test    

        # Reshape the data for the classifier
        X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

        # Build the classifier
        classifier = self.classifier_3_conv2d(X_train.shape[1:])
        
        # Fit the classifier variables
        History = classifier.fit(X_train, y_train,
                                 batch_size = int(X_train.shape[0]/4),
                                 epochs = no_of_epochs,
                                 validation_data = (X_test, y_test),
                                 verbose=0) 
        
        return History
    
    
    
class Multible_Binary_Classifier():
    
    def __init__(self, no_of_seeds = 100, no_of_epochs = 1000, norm = None, save = True, augmentation = False):
        
        self.no_of_seeds = no_of_seeds
        self.no_of_epochs = no_of_epochs
        self.norm = norm
        self.save = save
        self.augmentation = augmentation
        
        
    def cell(self, img, mask, pattern, date, cell,
             l = 15, img_scale = 1, l_min = 1.5, l_max = 2.5,
             ratio = 1., remove = None, center_ratio = 4):
        
        # Read the image to be cut and the mask you will cut on
        image = io.imread('{}/{}/{}'.format(date, cell, img))
        Pattern = io.imread('{}/{}/{}'.format(date, cell, pattern))
        mask = io.imread('{}/{}/{}'.format(date, cell, mask))
        
        # Create the two stacks of binary images to check the pattern
        mini_images = Binary_Squares(img = Pattern,
                                     coordinate = '{}/{}/coordinates.csv'.format(date, cell),
                                     l = l,
                                     img_scale = img_scale,
                                     l_min = l_min,
                                     l_max = l_max)
        
        pattern_dots, pattern_shifted_dots = mini_images.masked_squares(mask, ratio = ratio, remove = remove, center_ratio = center_ratio)
        
        
        # Create the two stacks of binary images
        mini_images = Binary_Squares(img = image,
                                     coordinate = '{}/{}/coordinates.csv'.format(date, cell),
                                     l = l,
                                     img_scale = img_scale,
                                     l_min = l_min,
                                     l_max = l_max)
        
        dots, shifted_dots = mini_images.masked_squares(mask, ratio = ratio, remove = remove, center_ratio = center_ratio)
        
        print('The number of mini images = {}'.format(2*dots.shape[0]))
                
        # Split the mini images into a binary labeled training and test samples
        data = Binary_Images(dots, shifted_dots)
    
        # Record the time at the begining of the process
        tic = time.time()
        
        #create a loop to train for several collection training and validation samples
        histories = []
        for Seed in range(self.no_of_seeds):
            #Collect the data for the seed
            X_train, X_test, y_train, y_test = data.train_test_seed(seed = Seed, augmentation = self.augmentation)

            classification = Binary_Classifier(X_train, X_test, y_train, y_test)
            History = classification.classify(no_of_epochs = self.no_of_epochs)

            #Extract the history of the training
            histories.append(History.history)
            
            #Save the history of the training after each seed
            if self.save == True:
                np.save('{}/results/{}/histories.npy'.format(date, cell), histories)
            
            #print the results after each seed
            print("Seed: {}    Loss: {:.2f}    Accuracy: {:.2f} ".format(Seed,
                                                                         np.min(histories[-1]['val_loss']),
                                                                         histories[-1]['val_accuracy'][histories[-1]['val_loss'].index(np.min(histories[-1]['val_loss']))]))

        min_loss, max_acc = [], []
        for history in histories:
            min_loss.append(np.min(history['val_loss']))
            max_acc.append(history['val_accuracy'][history['val_loss'].index(np.min(history['val_loss']))])        
        
        # display the average of the pattern to verify the quality of the localization
        plt.figure(figsize=(12,12))
        
        plt.subplot(221)
        plt.imshow(np.mean(pattern_dots, axis= 0), cmap ='gray')
        plt.axis('off')
        plt.title('on dot')
    
        plt.subplot(222)
        plt.imshow(np.mean(pattern_shifted_dots, axis= 0), cmap ='gray')
        plt.axis('off')
        plt.title('off dot')
        
        plt.subplot(223)
        n, bins, patches = plt.hist(max_acc, bins=20, color='#607c8e')
        plt.grid(axis='y')
        plt.title("Maximal Accuracy ({:.2f})".format(np.mean(max_acc)))
        plt.xlabel('Accuracy')
        plt.ylabel("Frequency")
        maxfreq = n.max()
        plt.xlim(0,1)
        plt.ylim(ymax = maxfreq + maxfreq / 10)
        
        plt.subplot(224)
        n, bins, patches = plt.hist(min_loss, bins=20, color='#607c8e')
        plt.grid(axis='y')
        plt.title("Minimal Loss ({:.2f})".format(np.mean(min_loss)))
        plt.xlabel('Loss')
        plt.ylabel("Frequency")
        maxfreq = n.max()
        plt.xlim(0,1)
        plt.ylim(ymax = maxfreq + maxfreq / 10)
        plt.savefig('{}/results/{}/histograms'.format(date, cell))
        plt.close()
        
        # Record the time at the end of the process
        toc = time.time()
        
        # Print the time taken by the process
        hours, rem = divmod(toc-tic, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Computation time for {} seeds and {} epochs = {:0>2}:{:0>2}:{:05.2f}".format(self.no_of_seeds, self.no_of_epochs, int(hours), int(minutes), seconds))
        
        return histories
    


class Training_History:
    
    def __init__(self, histories):
        
        self.histories = histories
    
    
    def show(self, n):
        
        plt.figure(figsize=(6,10))
        
        # Plot training & validation accuracy values
        plt.subplot(211)
        plt.plot(self.histories[n]['accuracy'])
        plt.plot(self.histories[n]['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        #plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        
        # Plot training & validation loss values
        plt.subplot(212)
        plt.plot(self.histories[n]['loss'])
        plt.plot(self.histories[n]['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.ylim(bottom=0, top = 2)
        plt.show()
            
        print("The minimum validation loss is {:.2f} and the maximum accuracy is {:.2f}%."
              .format(np.min(self.histories[n]['val_loss']), 100*np.max(self.histories[n]['val_accuracy'])))
        
        
        
    def relative_frequency(my_list, NAME, label, title= "Histogaram", show = False):
        n, bins, patches = plt.hist(my_list, bins=20, color='#607c8e')
        plt.grid(axis='y')
        plt.title(title)
        plt.xlabel(label)
        plt.ylabel("Frequency")
        maxfreq = n.max()
        plt.xlim(0,1)
        plt.ylim(ymax = maxfreq + maxfreq / 10)
        plt.savefig('{}/{}.png'.format(NAME, label))
        if show == True:
            plt.show()
        plt.close()
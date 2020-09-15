import numpy as np
import random
from sklearn.model_selection import train_test_split


class Binary_Images():
    
    def __init__(self, stack0, stack1):
        
        self.stack0 = stack0
        self.stack1 = stack1
        
    
    def binary_equate(self, arr1, arr2):
        # Equating the number of mini images between the on dots and off dots
        while arr1.shape[0] > arr2.shape[0]:
            arr1 = np.delete(arr1, np.random.randint(0, arr1.shape[0]), 0)
            
        while arr1.shape[0] < arr2.shape[0]:
            arr2 = np.delete(arr2, np.random.randint(0, arr2.shape[0]), 0)
            
        return arr1, arr2
            
    def binary_labeled(self, equate = True):
        
        if equate == True:
            stack0, stack1 = self.binary_equate(self.stack0, self.stack1)
        
        training_data = []
        for img in stack0:
            training_data.append([img, 0])
            
        for img in stack1:
            training_data.append([img, 1])
            
        #shuffle the data
        random.shuffle(training_data)
        
        #divide the data into set of images and labels 0,1
        X, y = [], []
        for features, label in training_data:
            X.append(features)
            y.append(label)
            
        return np.array(X), np.array(y)
    
    def data_augmentation(self, X_train, y_train):
    
        # extend the data with rotated images and repeated labels
        X_train_ext = np.vstack((X_train, np.rot90(X_train, 1, (1,2)), np.rot90(X_train, 2, (1,2)), np.rot90(X_train, 3, (1,2))))
        y_train_ext = np.hstack((y_train, y_train, y_train, y_train))
    
        training_data = []
        for i in range(X_train_ext.shape[0]):
            training_data.append([X_train_ext[i], y_train_ext[i]])
    
        #shuffle the data
        random.shuffle(training_data)
    
        #divide the data into set of images and labels 0,1
        X, y = [], []
        for features, label in training_data:
            X.append(features)
            y.append(label)
    
        #transform the lists into numpy arrays and print it's shape
        X_train_aug, y_train_aug = np.array(X), np.array(y)
        
        return X_train_aug, y_train_aug
    
    
    def train_test_seed(self, seed = 0, ratio = 0.2, norm = None, augmentation = False):
    
        #Load the mini images of the cell
        X, y = self.binary_labeled(equate = True)
        
        #Subtract the base
        X = X - X.min()
        
        #Normalization
        if norm == "max":
            X = X / np.max(X)
        elif norm == "median":
            X = X / (2 * np.median(X))
        elif norm == "mean":
            X = X / (2 * np.mean(X))
        
        i = 0
        y_test = [1, 1, 0]
        while (np.sum(y_test))*2 != len(y_test):
            np.random.seed(seed*33 + 3333 + i)
            i += 1
            if (int(ratio * len(y) + 1) % 2) == 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = int(ratio * len(y) + 1))
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = int(ratio * len(y) + 1) + 1)
                
        # Data augmentation by rotating the traning image 90 degree 3 times
        if augmentation == True:
            X_train, y_train = self.data_augmentation(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
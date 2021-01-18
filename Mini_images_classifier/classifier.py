#! /home/ahmed/anaconda3/envs/tf-gpu/bin/python

#Libraries
#import sys

# Utilities
from classification import Multible_Binary_Classifier


classification = Multible_Binary_Classifier(no_of_seeds = 10, no_of_epochs = 50, norm = 'mean')
histories = classification.cell(img = "actin1-1.tif",
                                mask = "Mask.tif",
                                pattern = "pattern1_25im-1.tif",
                                date = "data/data_400_20190208",
                                cell = "cell_1_1",
                                img_scale = 10)

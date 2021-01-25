#! /home/ahmed/anaconda3/envs/tf-gpu/bin/python

#Libraries
import sys

# Utilities
from classification import Multible_Binary_Classifier


classification = Multible_Binary_Classifier(no_of_seeds = 100, no_of_epochs = 5000, norm = 'mean')
histories = classification.cell(img = "actin.tif",
                                mask = "Mask.tif",
                                pattern = "pattern.tif",
                                date = "data/data_400_20190208",
                                cell = "cell_" + str(sys.argv[1]),
                                l = 15,
                                l_min = 0.7,
                                l_max = 1.3,
                                img_scale = 10)

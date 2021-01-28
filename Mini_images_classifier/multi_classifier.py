#! /home/ahmed/anaconda3/envs/tf-gpu/bin/python

# Libraries
import os
import fnmatch

# Utilities
from classification import Multible_Binary_Classifier

# Collecting the cell directories in a list
Cells = [cell for cell in [f.name for f in os.scandir("data/data_400_20190208") if f.is_dir()]
         if fnmatch.fnmatch(cell, 'cell_*')]

# Print the cells to be trained on
print(Cells)

# Define the classifer parameters
classification = Multible_Binary_Classifier(no_of_seeds = 100, no_of_epochs = 5000, norm = 'mean')

# Classify multible cells
for Cell in Cells:
    classification.cell(img = "ricm.tif",
                        mask = "Mask.tif",
                        pattern = "pattern.tif",
                        date = "data/data_400_20190208",
                        cell = Cell,
                        l = 15,
                        l_min = 0.7,
                        l_max = 1.3,
                        img_scale = 10)
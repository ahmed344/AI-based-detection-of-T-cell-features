#Libraries
import sys

# Utilities
from classification import Multible_Binary_Classifier


classification = Multible_Binary_Classifier(no_of_seeds = 100, no_of_epochs = 5000, norm = 'mean')
histories = classification.cell(img = "actin1-1.tif",
                                mask = "Mask.tif",
                                pattern = "pattern1_25im-1.tif",
                                date = "data_20190326",
                                cell = "cell_1_1",
                                img_scale = 10)

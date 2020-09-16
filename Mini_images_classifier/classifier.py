#Libraries
import sys

# Utilities
from classification import Multible_Binary_Classifier


classification = Multible_Binary_Classifier(no_of_seeds = 1, no_of_epochs = 5, norm = 'mean')
histories = classification.cell(img = "RICM-1.tif",
                                mask = "Mask.tif",
                                pattern = "pattern1_25im-1.tif",
                                date = "data_20190513",
                                cell = "cell_" + str(sys.argv[1]),
                                img_scale = 10)

import os
import fnmatch

#Collecting the cell folders in a list
Results = [cell for cell in [ f.name for f in os.scandir() if f.is_dir()] if fnmatch.fnmatch(cell, 'cell_*')]

#creat the folders
os.mkdir("results")

#make a numpy array for each set of images and create it's results folder
for result in Results:
    #creat the folders
    os.mkdir("results/{}".format(result))

#! /usr/bin/python

# Libraries
import os
import fnmatch

# Collecting the cell directories in a list
Results = [cell for cell in [ f.name for f in os.scandir() if f.is_dir()] if fnmatch.fnmatch(cell, 'cell_*')]

# Creat the directories
try:
    os.mkdir("results")
except:
    print('The directory results already exists')


# Make a numpy array for each set of images and create it's results directory
for result in Results:
    # Creat the directories
    try:
        os.mkdir("results/{}".format(result))
    except:
        print(result, 'directory already exists')

    # Rename the files if they do exist
    try:
        os.rename("{}/{}".format(result, 'Results.csv'), "{}/{}".format(result, 'coordinates.csv'))
    except:
        print(result, 'Results.csv file does not exist')
    try:
        os.rename("{}/{}".format(result, 'actin1-1.tif'), "{}/{}".format(result, 'actin.tif'))
    except:
        print(result, 'actin1-1.tif file does not exist')
    try:
        os.rename("{}/{}".format(result, 'pattern1_25im-1.tif'), "{}/{}".format(result, 'pattern.tif'))
    except:
        print(result, 'pattern1_25im-1.tif file does not exist')
    try:
        os.rename("{}/{}".format(result, 'RICM-1.tif'), "{}/{}".format(result, 'ricm.tif'))
    except:
        print(result, 'RICM-1.tif file does not exist')
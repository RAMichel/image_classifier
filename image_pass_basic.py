__author__ = 'raymichel'

import os, os.path
from os.path import isfile, join
from PIL import Image
import re
import sys, getopt
import numpy

import sklearn
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search
from sklearn.externals import joblib

# this size chosen as a smaller version to limit pixels (it's still ~400K pixels per image)
STANDARDIZED_SIZE = (426,240)

def get_all_images(foldername):
    """
    Retrieves all images from our destination folder and puts them into an array
    """
    files = []

    print 'Reading files from ' + str(foldername)
    try:
        # all_images = [f for f in os.listdir(foldername) if isfile(join(foldername, f))]
        for (dirpath, dirnames, filenames) in os.walk(foldername):
            files.extend((join(foldername, x), x) for x in filenames)
            break
    except:
        all_images = []
        e = sys.exc_info()[0]
        print 'failure in retrieving images, error:'

    return files

def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARDIZED_SIZE))
    img = img.resize(STANDARDIZED_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = numpy.array(img)
    print str(filename) + " has been matrixized"
    return img

def flatten_matrix(matrix):
    """
    takes in a numpy array and flattens it to a single 1D array of tuples
    """
    s = matrix.shape[0] * matrix.shape[1]
    img_wide = matrix.reshape(1, s)
    print "image from matrix has been flattened"
    return img_wide[0]

def write_all_numpy_arrays(in_imgs, directory_to_write):
    """
    Writes all our .npy files for easy data write and retrieval without maxing out system memory
    """
    for img in in_imgs:
        write_file = directory_to_write+'/'+str(img[1])
        this_img = img_to_matrix(img[0])
        this_img = flatten_matrix(this_img)
        print "Writing to " + write_file
        numpy.save(write_file, this_img)

def build_classifier(train_data_x, train_data_y):
    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],
                  'gamma': [0.01, 0.001, 0.0001]}
    print "Doing Grid Search on SVC fitting of training data."
    classifier = grid_search.GridSearchCV(svm.SVC(), parameters).fit(train_data_x, train_data_y)
    print "Doing classifier estimation."
    classifier = classifier.best_estimator_
    return classifier

def show_usage():
    print 'image_pass_basic.py -i <input_directory> -o <output_directory>'
    print 'Use input_directory and output_directory as the directory CONTAINING the chosen/ and unchosen/ directories'
    sys.exit(2)

def main(argv):
    """
    Does our basic logic progression:
    0) parse relevant arguments
    0b) check for all needed directories
    1) retrieve all images based on input arguments
    2) flatten all images
    2.5) Write flattened image matrix to file to preserve memory
    """

    # (0)
    directory_to_read = ''
    directory_to_write = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:l:",["idir=","odir=","label="])
    except getopt.GetoptError:
        show_usage()

    for opt, arg in opts:
        if opt == '-h':
            show_usage()
        elif opt in ("-i", "--idir"):
            directory_to_read = arg
            if not directory_to_read:
                print 'Please supply a source directory via -i'
                sys.exit(2)
        elif opt in ("-o", "--odir"):
            directory_to_write = arg
            if not directory_to_write:
                print 'Please supply a file output path via -o'
                sys.exit(2)

    #(0b)
    try:
        if not os.path.exists(directory_to_write+'/unchosen/'):
            print "Creating the /unchosen directory."
            os.makedirs(directory_to_write+'/unchosen/')
        if not os.path.exists(directory_to_write+'/chosen/'):
            print "Creating the /chosen directory."
            os.makedirs(directory_to_write+'/chosen/')
    except:
        print "Errors in checking/creating directories for project. Check permissions and restart."
        sys.exit(2)

    # # (1a)
    # all_images = get_all_images(directory_to_read+'/unchosen/')
    # # (2a)
    # write_all_numpy_arrays(all_images, directory_to_write+'/unchosen/')
    # # (1b)
    # all_images = get_all_images(directory_to_read+'/chosen/')
    # # (2b)
    # write_all_numpy_arrays(all_images, directory_to_write+'/chosen/')
    # print "All numpy files written for relevant images."

    # grab our images in numpy array form
    images_unchosen = get_all_images(directory_to_write+'/unchosen/')
    images_chosen = get_all_images(directory_to_write+'/chosen/')
    print "Retrieved all files, about to convert to data."

    # convert files to processable data
    unchosen_data = [numpy.load(img[0]) for img in images_unchosen] # img returns a tuple, we want the relative-path file name
    unchosen_data = numpy.array(unchosen_data)
    chosen_data = [numpy.load(img[0]) for img in images_chosen]
    chosen_data = numpy.array(chosen_data)

    data_x = numpy.concatenate((unchosen_data, chosen_data))
    print "Converted numpy arrays to data, about to build classifier."
    # Pick on random ~20% of the base set to do our classifier build
    test_set = numpy.random.uniform(0,1,len(data_x)) <= 0.8
    data_y = (['unchosen'] * len(images_unchosen)) + (['chosen'] * len(images_chosen))
    data_y = numpy.where(numpy.array(data_y)=="chosen",1,0)

    train_x, train_y = data_x[test_set], data_y[test_set]
    print "Training data assembled."

    classifier = build_classifier(train_x, train_y)
    print "Classifier established and built."

    test_x, test_y = data_x[test_set==False], data_y[test_set==False]

    print "Attempting metrics:"
    print "\n"
    print "Parameters: " + classifier.best_params_
    print "\n"
    print "Best classifier score: "
    print metrics.classification_report(test_y, classifier.predict(test_x))


if __name__ == "__main__":
    main(sys.argv[1:])

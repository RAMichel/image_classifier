__author__ = 'raymichel'

import os, os.path
from os.path import isfile, join
import sys, getopt
import numpy

import sklearn
from sklearn.externals import joblib
import sklearn
from sklearn import svm
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.manifold import SpectralEmbedding

AVAILABLE_CLASSIFIERS = ['svc_basic', 'svc_extensive', 'kneighbors_basic', 'bagging_basic', 'spectral_basic']

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
        print 'failure in retrieving images, error: ' + str(e)

    return files

def show_usage():
    print "classifier_tester.py -c <classifier.pkl> -i <input_numpy_test_directory>"

def main(argv):
    """
    Does our basic logic progression:
    0) parse relevant arguments
    0b) check for all needed directories
    1) retrieve all images based on input arguments
    2) build relevant data (do any specific manipulations for specific classifier types)
    3) attempt comparisons
    """

    # (0)
    directory_to_read = ''
    classifier_file = ''
    classifier_label = ''
    try:
        opts, args = getopt.getopt(argv,"hi:c:l:",["idir=","classifer=","class_label="])
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
        elif opt in ("-c", "--classifier"):
            classifier_file = arg
            if not classifier_file:
                print 'Please supply a path to a classifier file (.pkl)'
        elif opt in ("-l", "--class_label"):
            classifier_label = arg
            if not classifier_label:
                print 'Please supply the type of classifier that you wish to test.'

    #(0b)
    try:
        if (not os.path.exists(directory_to_read+'/unchosen/')) or (not os.path.exists(directory_to_read+'/chosen/')):
            print 'Directories unchosen/ or chosen/ are not valid.'
    except:
        print 'Errors in checking directories for project. Check permissions and restart program.'
        sys.exit(2)
    if not classifier_label or classifier_label not in AVAILABLE_CLASSIFIERS:
        print 'Please select one of the available classifiers:'
        for cls in AVAILABLE_CLASSIFIERS:
            print '\t'+cls
        sys.exit(2)

    # (1) grab our images in numpy array form
    images_unchosen = get_all_images(directory_to_read+'/unchosen/')
    images_chosen = get_all_images(directory_to_read+'/chosen/')
    print "Retrieved all files, about to convert to data."

    # (2) convert files to processable data
    unchosen_data = [numpy.load(img[0]) for img in images_unchosen] # img returns a tuple, we want the relative-path file name
    unchosen_data = numpy.array(unchosen_data)
    chosen_data = [numpy.load(img[0]) for img in images_chosen]
    chosen_data = numpy.array(chosen_data)

    data_x = numpy.concatenate((unchosen_data, chosen_data))
    print "Converted numpy arrays to data, about to build classifier."
    # Pick on random ~20% of the base set to do our classifier build
    data_y = (['unchosen'] * len(images_unchosen)) + (['chosen'] * len(images_chosen))
    data_y = numpy.where(numpy.array(data_y)=="chosen",1,0)

    classifier = joblib.load(classifier_file)
    print "Classifier loaded."

    # This line is to make sure scoping rules work correctly
    test_data_x = data_x

    # Checking to see if we need to do any specific data manipulation for the input test data
    if classifier_label == 'kneighbors_basic':
        print "Classifier is of type KNeighbors, so we need to do a quick PCA transform on our data."
        pca = RandomizedPCA(n_components=20)
        test_data_x = pca.fit_transform(data_x)
    elif classifier_label == 'spectral_basic':
        print "Classifier used Spectral imaging, so we need to do a spectral transformation on our data."
        spc = SpectralEmbedding(n_components=20)
        test_data_x = spc.fit_transform(data_x)

    numpy.set_printoptions(threshold=numpy.nan)
    "Doing prediction on input data set"
    print classifier.predict(test_data_x)


if __name__ == "__main__":
    main(sys.argv[1:])
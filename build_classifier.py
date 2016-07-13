__author__ = 'raymichel'

import os, os.path
from os.path import isfile, join
from PIL import Image
import sys, getopt
import numpy

import sklearn
from sklearn import svm
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.manifold import SpectralEmbedding

# this size chosen as a smaller version to limit pixels (it's still ~400K pixels per image)
STANDARDIZED_SIZE = (426,240)
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

def build_classifier(train_data_x_in, train_data_y, classifier_in='svc_basic'):
    print "Attempting to build classifier."
    train_data_x = train_data_x_in
    transformer = ''
    # classifier = grid_search.GridSearchCV(svm.SVC(), parameters).fit(train_data_x, train_data_y)
    if classifier_in == 'svc_basic':
        classifier = svm.SVC()
        print "Selection was basic svm.SVC."
    elif classifier_in == 'svc_extensive':
        classifier = svm.SVC(kernel="linear", C=0.025, gamma=0.01)
        print "Selection was extensive svm.SVC, with linear kernel, C==0.025 and gamma==0.01."
    elif classifier_in == 'kneighbors_basic':
        transformer = RandomizedPCA(n_components=2000)
        train_data_x = transformer.fit_transform(train_data_x)
        classifier = KNeighborsClassifier()
        print "Selection was KNeighbors basic, using RandomizedPCA to transform data first. n_components==2000."
    elif classifier_in == 'bagging_basic':
        classifier = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
        print "Selection was Bagging basic, with max_samples==0.5 and max_features==0.5."
    elif classifier_in == 'spectral_basic':
        transformer = SpectralEmbedding(n_components=2000)
        train_data_x = transformer.fit_transform(train_data_x)
        classifier = KNeighborsClassifier()
        print "Selection was Spectral basic, using svm.SVC with Spectral data fitting. n_components==2000."
    # default to SVC in case of any sort of parsing error.
    else:
        print "Error in selecting classifier class. Reverting to SVC."
        classifier = svm.SVC()
    classifier.fit(train_data_x, train_data_y)
    print "Doing classifier estimation."
    return classifier, train_data_x, transformer

def show_usage():
    print 'build_classifier.py -i <input_directory>-c <classifier_class>'
    print 'Use input_directory and output_directory as the directory CONTAINING the chosen/ and unchosen/ directories'
    print '-c can be used with:'
    for cls in AVAILABLE_CLASSIFIERS:
        print '\t'+str(cls)
    sys.exit(2)

def main(argv):
    """
    Does our basic logic progression:
    0) parse relevant arguments
    0b) check for all needed directories
    1) retrieve all images based on input arguments
    2) build classifier based on input suggestion and save to file
    3) test classifier against test data
    """

    # (0)
    directory_to_read = ''
    classifier_class = ''
    try:
        opts, args = getopt.getopt(argv,"hi:c:",["idir=","class="])
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
        elif opt in ("-c", "--class"):
            classifier_class = arg
            if classifier_class not in AVAILABLE_CLASSIFIERS:
                print arg + ' is not in the list of available classifiers.'
                print 'Please select a classifier from the following list:'
                for cls in AVAILABLE_CLASSIFIERS:
                    print '\t'+str(cls)
                sys.exit(2)


    #(0b)
    try:
        if not os.path.exists(directory_to_read+'/unchosen/'):
            print "There is no /unchosen directory in the top-level input directory."
            sys.exit(2)
        if not os.path.exists(directory_to_read+'/chosen/'):
            print "There is no /chosen directory in the top-level input directory."
            sys.exit(2)
        if not classifier_class:
            classifer_class = 'svc_basic'
            print "No classifier class specified, defaulting to scikit-SVC."
        if not os.path.exists(directory_to_read+'/'+classifier_class+'/'):
            print "Creating directory to contain the " + classifier_class + " classifier and relevant .npy files."
            os.makedirs(directory_to_read+'/'+classifier_class+'/')
    except:
        print "Errors in checking/creating directories for project. Check permissions and restart."
        sys.exit(2)

    # (1) grab our images in numpy array form
    images_unchosen = get_all_images(directory_to_read+'/unchosen/')
    images_chosen = get_all_images(directory_to_read+'/chosen/')
    print "Retrieved all files, about to convert to data."

    # convert files to processable data
    unchosen_data = [numpy.load(img[0]) for img in images_unchosen] # img returns a tuple, we want the relative-path file name
    unchosen_data = numpy.array(unchosen_data)
    chosen_data = [numpy.load(img[0]) for img in images_chosen]
    chosen_data = numpy.array(chosen_data)

    data_x = numpy.concatenate((unchosen_data, chosen_data))
    print "Converted numpy arrays to data, about to build classifier."
    # Pick on random ~20% of the base set to do our classifier build
    train_set = numpy.random.uniform(0,1,len(data_x)) <= 0.7
    data_y = (['unchosen'] * len(images_unchosen)) + (['chosen'] * len(images_chosen))
    data_y = numpy.where(numpy.array(data_y)=="chosen",1,0)

    # Build our randomized data and labels
    train_x, train_y = data_x[train_set], data_y[train_set]
    print "Training data assembled."

    # (2) Build our classifier and then return any possibly manipulated data that was required for building that classifier
    print "Training classifier with class " + classifier_class
    classifier, train_x, transformer = build_classifier(train_x, train_y, classifier_class)
    print "Classifier established and built."

    # joblib dumps to the defined directory not only the pkl file, but also all associated .npy files for the classifier
    print "Dumping classifier to file."
    joblib.dump(classifier, directory_to_read+'/'+classifier_class+'/classifier_out.pkl')
    if transformer:
        print "Dumping the transformer to file, too."
        joblib.dump(transformer, directory_to_read+'/'+classifier_class+'/transformer.pkl')

    # (3)
    numpy.set_printoptions(threshold=numpy.nan)
    print "This is the y data for the training set. The following prediction should show this:"
    print train_y
    print "\n"
    print "Doing re-prediction on training class for testing"
    print classifier.predict(train_x)


if __name__ == "__main__":
    main(sys.argv[1:])

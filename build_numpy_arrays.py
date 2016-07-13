__author__ = 'raymichel'

import os, os.path
from os.path import isfile, join
from PIL import Image
import sys, getopt
import numpy

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
        print 'failure in retrieving images, error: ' + str(e)

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

def show_usage():
    print "python build_numpy_arrays.py -i <top_input_dir_with_chosen_and_unchosen> -o <top_output_directory>"

def main(argv):
    """
    Does a very simple parsing of arguments, and then building the relevant pixel numpy arrays.
    """
    directory_to_read = ''
    directory_to_write = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:l:",["idir=","odir="])
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

    # (1a)
    all_images = get_all_images(directory_to_read+'/unchosen/')
    # (2a)
    write_all_numpy_arrays(all_images, directory_to_write+'/unchosen/')
    # (1b)
    all_images = get_all_images(directory_to_read+'/chosen/')
    # (2b)
    write_all_numpy_arrays(all_images, directory_to_write+'/chosen/')
    print "All numpy files written for relevant images."

if __name__ == "__main__":
    main(sys.argv[1:])
##This system is used to do three objectives in three different executable files:

1. Compile images into single-dimension numpy arrays for pixel features.
2. Build a classifier (of a selectable list) and train it on an input of data.
3. Use a specified Python-pickled classifier on an input data to do predictions.

Once step (3.) is completed, it will output a single array of boolean values, where:
    0 = unchosen/unmatched
    1 = chosen/matched

###The directory structure required for the image files:
```
/
--/<image_directory>
----/chosen     # of the "selected" label for images
----/unchosen   # of the "unselected" label for images
--/<output_directory>   # This directory needs to be created for the output files
--*
```
###The output directories after using build_numpy_arrays.py:
```
/
--/<image_directory>
----/chosen     # of the "selected" label for images
----/unchosen   # of the "unselected" label for images
--/<output_directory>   # This directory needs to be created for the output files
----/chosen     # of the "selected" .npy arrays
----/unchosen   # of the "unselected" .npy arrays
--*
```
###The output directory after any selected use of build_classifier.py:
```
/
--/<image_directory>
----/chosen     # of the "selected" label for images
----/unchosen   # of the "unselected" label for images
--/<output_directory>   # This directory needs to be created for the output files
----/chosen     # of the "selected" .npy arrays
----/unchosen   # of the "unselected" .npy arrays
----/<classifier_type>/classifier_out.pkl       # The pickled Python classifier class.
----/<classifier_type>/*.npy                    # The .npy arrays related to the classififer that it uses for predictions
--*
```
##Use:

###1. Compile images into single-dimension numpy arrays for pixel features.

   ``` python build_numpy_arrays.py -i <image_directory> -o <output_directory>```

###2. Build a classifier of a selectable list and train it on an input of random data.

    python build_classifier.py -i <output_directory> -c <classifier_class>

    <classifier_class> can relate to one of the following:
        svc_basic
        svc_extensive
        kneighbors_basic
        bagging_basic
        spectral_basic
        
###3. Use a specified Python-pickled classifier on an input data to do predictions.

    python classifier_tester.py -i <any_output_directory> -d <directory_containing_classifier> -c <path_to_classifier_out.pkl> -l <classifier_class>
    
    <classifier_class> is the same as used in (2.), and helps in accurate data transformations to do predictions.
    <any_output_directory> is any *_out directory after running build_numpy_arrays.py.
    <directory_containing_classifier> is the direct directory above the classifier_out.pkl file, and is used for extra required transformations with specific classifiers.



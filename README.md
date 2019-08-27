Project Description :

create_training_data.py will create the training data in 'finaldata.csv' file using images from passed input directory.

location of generated 'finaldata.csv' file will be in passed input directory.

CSV file will include entries in "text | label" format.

create_model.py reads'finaldata.csv' file as input and create model accordingly. It will output the accuracy of various models for the given training data set.

extract_utility.py contains functions to extract the data from images and clean those data.

classifier_utility contains functions to train the model.

Input directory contains four folder. Don't change name of those folders. You can put your documents images in those folders according to document type.

How to run the code :

First : Run create_training_data.py to create the training data using one of the following ways.

 1. Pass input directory from command line :

     create_training_data.py -i <input_dir> 

     Note : You have to uncomment the taking command line argument functionality in given               python file itself. By default it is accepting hard coded input directory value.

 2. Set'input_dir' variable value to your input directory value in python file itself.

Second : Run  create_model.py to train the model using one of the following ways.

 1. Pass input file from command line :

     create_model.py -i <input_file> 

     Note : You have to uncomment the taking command line argument functionality in given               python file itself. By default it is accepting hard coded input file value.

 2. Set'input_file' variable value to your input file value in python file itself.

"# image_classification_ocr" 

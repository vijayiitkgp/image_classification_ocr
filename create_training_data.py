# -*- coding: utf-8 -*-
"""
Created on Sat Mar  10 22:02:54 2019

@author: VIJAY
"""
import numpy as np
#import os, string, cv2, glob, pytesseract, re, argparse, shutil
import os, string, cv2, glob, re, argparse, shutil

from PIL import Image
#from pytesseract import image_to_string
import csv
#from extract_utility import clean_data, get_string

#Following is the argument line code
""""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This program create training data from a set of documents.")
    parser.add_argument("-i", "--input_dir", help="Input directory for the files to be extracted")
    args = parser.parse_args()

    input_dir = args.input_dir

"""
input_dir='/training data'

list = ['AADHAR', 'PAN', 'VOTER', 'DL']

for item in list :
    image_dir=input_dir+'/'+item
    input_file=input_dir+'/'+item+'.csv'
    print(input_dir, input_file)
    
    for filename in glob.glob(os.path.join(image_dir, '*.jpeg')) :
       print(filename)


    for infile in glob.glob( os.path.join(image_dir, '*.png') ):
     print (infile)    
    im_names = glob.glob(os.path.join(image_dir, '*.png')) + \
                   glob.glob(os.path.join(image_dir, '*.jpg')) + \
                   glob.glob(os.path.join(image_dir, '*.jpeg'))
    print(im_names)               

    for im_name in im_names:
        print(im_name)
        """
        arr = clean_data(image_to_string(Image.open(im_name), lang='eng'))
        i = 1
        while i < 8:
                        result = clean_data(get_string(im_name, i))
                        arr = arr + ' , ' + result                        
                        i += 1

        with open(input_file, 'a') as csvFile:
                         writer = csv.writer(csvFile)
                         arr = arr + ' | ' +item                        
                         csv_data = np.array([arr])                         
                         writer.writerow(csv_data)
                         csvFile.close()

output_file=input_dir+'/finaldata.csv'
output_writer = csv.writer(open(output_file, 'w'))
for item in list :
  input_file=input_dir+'/'+item+'.csv'
  input_reader = csv.reader(open(input_file, 'r'))
  for row in input_reader:
     temp = ''.join(row)
     if temp.strip() == '':
        continue
     output_writer.writerow(row)
"""
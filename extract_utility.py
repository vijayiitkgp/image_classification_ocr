# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 00:27:05 2019

@author: VIJAY
"""
import numpy as np
import os, string, cv2, pytesseract, re 
from PIL import Image
from pytesseract import image_to_string

def clean_data(text) :
                    data = text.translate ({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"}).replace('\n\n\n\n\n', ' ').replace('\n\n', ' ').replace('\n\n\n\n', ' ').replace('\n\n\n', ' ').replace('\n', ' ')                   
                    print(data)
                    data = re.split(r'\W+', data)
                    print(data)
                    table = str.maketrans('', '', string.punctuation)
                    data = [w.translate(table) for w in data]
                    print(data)
                    #data = list(filter(None, data))
                    data=' '.join(data)
                    print(data)
                    return data

# =============================================================================== #
#    Threshold Methods                                                            #
# =============================================================================== #
# 1. Binary-Otsu w/ Gaussian Blur (kernel size = 9)                               #
# 2. Binary-Otsu w/ Gaussian Blur (kernel size = 7)                               #
# 3. Binary-Otsu w/ Gaussian Blur (kernel size = 5)                               #
# 4. Binary-Otsu w/ Median Blur (kernel size = 5)                                 #
# 5. Binary-Otsu w/ Median Blur (kernel size = 3)                                 #
# 6. Adaptive Gaussian Threshold (31,2) w/ Gaussian Blur (kernel size = 5)        #
# 7. Adaptive Gaussian Threshold (31,2) w/ Median Blur (kernel size = 5)          #
# =============================================================================== #

def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.threshold(cv2.medianBlur(img, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        6: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        7: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
    }
    return switcher.get(argument, "Invalid method")


def get_string(img_path, method):
    # Read image using opencv
    img = cv2.imread(img_path)
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]
        
    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)    

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    #  Apply threshold to get image with only black and white
    img = apply_threshold(img, method)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, lang="eng")

    return result                

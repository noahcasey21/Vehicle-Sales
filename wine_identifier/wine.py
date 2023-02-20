""" Steps: 
    0 - create a dataset of wine bottles for recognition
    1 - take in images
    2 - find individual wine bottles (if any): lots of recognition refining
    3 - determine type, brand, size, any other possible features
    4 - find that wine on the web
    4a - build a database of wines for faster retrieval?
    5 - compare
"""

import numpy as np
import csv
from sklearn.model_selection import train_test_split

class WineImageModel:
    """class to manage the code training and testing the model"""
    def __init__(self, data):
        self.data = data


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

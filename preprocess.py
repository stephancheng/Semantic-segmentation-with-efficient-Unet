# -*- coding: utf-8 -*-
"""
Created on Mon May 31 17:09:27 2021

@author: Stephan
"""
import os
import splitfolders
import cv2

# img = cv2.imread('semantic_annotations/semantic_annotations/gtLabels/00000_FV.png')
# img = cv2.imread('semantic_annotations/semantic_annotations/gtLabels/00000_FV.png', 0)
# print(img)

# split the rgb_image folder
splitfolders.ratio("rgb_images", output="images", seed=0, ratio=(.8, .2), group_prefix=None)

# split the semantic_annotations folder
os.chdir("semantic_annotations/")

splitfolders.ratio("semantic_annotations", output="Labels", seed=0, ratio=(.8, .2), group_prefix=None)
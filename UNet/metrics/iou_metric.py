# -*- coding: utf-8 -*-
"""
Created on Sun May  1 16:37:38 2022
"""

from sklearn.metrics import jaccard_score
from keras import backend as K
import cv2
import numpy as np

target = cv2.imread("./metrics/target.png")
target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

prediction = cv2.imread("./metrics/output.jpg")
prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
prediction = prediction[43:480,:]

# for each label
labels = [0,72,147,220]
jaccards = []
for label in labels:
    jac = jaccard_score(prediction.flatten(), target.flatten(),average = None, labels = [label]) 
    print(f"Label: {label} iou: {jac}")
    jaccards.append(jac)
    
iou = np.mean(jaccards)
print(iou)

# all data 
jac = jaccard_score(prediction.flatten(), target.flatten(), average = "macro", labels=[0,72,147])
# etiket dengesizliğini almaz direkt hesaplar, yukarıdaki 
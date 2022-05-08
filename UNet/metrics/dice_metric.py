# -*- coding: utf-8 -*-
"""
Created on Sun May  1 16:23:35 2022
"""

import cv2
import numpy as np


def dice(ground_truth, prediction):
    labels = [0,72,147,220]
    dices = []
    
    for label in labels:
        dice = np.sum(prediction[ground_truth==label] == label) *(2.0) / (np.sum(prediction[prediction==label] == label) + np.sum(ground_truth(ground_truth==label)==label))
        print(f"Label: {label} iou: {dice}")
        dices.append(dice)
        
    dice_metric = np.mean(dices)
    return dice_metric

target = cv2.imread("./metric/target.png")
target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)


prediction = cv2.imread("./metric/otuput.jpg")
prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
prediction = prediction[43:480,:]

dice_metric = dice(target,prediction)
print("Dice Similarity: {}".format(dice_metric))

cv2.imshow("Prediction ", prediction)
cv2.imshow("Target ", target)
cv2.waitkey(0)
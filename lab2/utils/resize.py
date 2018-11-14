import cv2
import os

path = "./test/"
dirs = os.listdir(path)

for item in dirs:
    src = cv2.imread(path + item)
    resized = cv2.resize(src, (128, 128))
    cv2.imwrite(path + "resized_" + item, resized)

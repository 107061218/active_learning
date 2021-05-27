import os
import cv2
import numpy as np


def create_folder(imgpath, vispath):
    desired_folder = [os.path.join(imgpath, 'FP'), os.path.join(imgpath, 'FN'), os.path.join(vispath, 'FP'), os.path.join(vispath, 'FN')]

    for folder in desired_folder:
        if os.path.isdir(folder):
            print(folder + ' exist')
        else:
            os.makedirs(folder)


def privacy(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    output = np.zeros(image.shape)

    dilated = cv2.dilate(image[:, :, 2], kernel)

    sobelx = cv2.Sobel(image[:, :, 1], cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(image[:, :, 1], cv2.CV_64F, 0, 1)
    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))
    edges = cv2.bitwise_or(sobelx, sobely)

    erode = cv2.erode(image[:, :, 0], kernel)

    output[:, :, 0] = erode
    output[:, :, 1] = edges
    output[:, :, 2] = dilated

    output = np.clip(output, 0, 255)

    # convert to gray scale
    output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    output = 255 - output
    output = output[..., np.newaxis]
    output = np.concatenate((output, output, output), axis=-1)

    return output

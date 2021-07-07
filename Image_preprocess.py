import numpy as np
from PIL import Image
from matplotlib.pyplot import imread
import cv2
from configs import *

MEANS = np.zeros(3)

def reshape_image(image):
    # Reshape image to match expected input of VGG16
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    return image

def fill_MEANS(image):
    MEANS = np.mean(image, axis=(0, 1))


def normalize_image(image):
    # Substract the mean to match the expected input of VGG16
    image = image - MEANS
    return image

def unnormalize_image(image):
    # Un-normalize the image so that it looks good
    image = image + MEANS
    return image

def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """
    fill_MEANS(image)
    image = reshape_image(image)
    image = normalize_image(image)

    return image

image = imread("train-pic.jpg")
img = Image.fromarray(image, 'RGB')
# img.show() #inja aks ro kamel baz kard
reshapeImage = reshape_image(image)
re_img = Image.fromarray(reshapeImage, 'RGB')
re_img.show()
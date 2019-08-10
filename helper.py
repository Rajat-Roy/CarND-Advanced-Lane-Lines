import numpy as np

def clip_image(image,  maximum):
    select = image > maximum
    image[select] = maximum
    return image

def normalize_image(image):
    if np.max(image) > 1:
        return image.astype(np.float32)/255
    else:
        return image
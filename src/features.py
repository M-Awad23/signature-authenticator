from skimage.feature import hog
import numpy as np

def extract_hog_features(images):
    features = []
    for img in images:
        img_uint8 = (img * 255).astype('uint8')
        fd = hog(
            img_uint8,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features.append(fd)
    
    return np.array(features)
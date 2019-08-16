import numpy as np
import cv2

class Filter:    
    def __init__(self):
        self.ksize=3
        self.sobel_thresh=(20, 100)
        self.mag_thresh=(30, 100)
        self.dir_thresh=(0.7, 1.3)
        self.abs_sobelx = None
        self.abs_sobely = None
        self.gradx = None
        self.grady = None
        self.mag_binary = None
        self.dir_binary = None
        self.combined = None
        self.color_transformed = None
        
    # Define a function that extracts threshold,
    def binary_thresh(self, binary_img, thresh=(0, 255)):
        binary = np.zeros_like(binary_img)
        binary[(binary_img >= thresh[0]) & (binary_img <= thresh[1])] = 1
        return binary

    # Define a function to color transform
    def color_transform(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hls = cv2.normalize(hls, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        L = np.zeros_like(hls[:,:,1])
        S = np.zeros_like(hls[:,:,2])

        Blend = np.zeros_like(hls)
        Target = np.zeros_like(hls)

        saturation = np.mean(hls[:,:,2])

        if saturation > 0.25:
            L[hls[:,:,1] > 0.93] = 1
            S[hls[:,:,2] > 0.7] = 1
        elif (saturation < 0.25) & (saturation > 0.15):
            L[hls[:,:,1] > 0.7] = 1
            S[hls[:,:,2] > 0.4] = 1
        else:
            L[hls[:,:,1] > 0.7] = 1
            S[hls[:,:,2] > 0.4] = 1

        Target = cv2.cvtColor(S,cv2.COLOR_GRAY2RGB)
        Blend = cv2.cvtColor(L,cv2.COLOR_GRAY2RGB)

        #Screen Blending: http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html
        Composite = 1 - (1-Target) * (1-Blend)
        Composite = cv2.normalize(Composite, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        self.color_transformed = cv2.cvtColor(Composite, cv2.COLOR_RGB2GRAY)

    # Define a function that applies Sobel x or y,
    def abs_sobelxy(self, img):
        # 1) Take the gradient in x and y separately
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, self.ksize)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, self.ksize)
        # 2) Take the absolute value of the x and y gradients
        self.abs_sobelx = np.absolute(sobelx)
        self.abs_sobely = np.absolute(sobely)

    # then takes an absolute value and applies a threshold.
    def abs_sobel_thresh(self):
        
        # Apply the following steps to img
        # 1) Take the derivative in x and y

        scaled_sobel = cv2.normalize(self.abs_sobelx, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        self.gradx = self.binary_thresh(scaled_sobel, self.sobel_thresh)

        scaled_sobel = cv2.normalize(self.abs_sobely, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        self.grady = self.binary_thresh(scaled_sobel, self.sobel_thresh)

    # Define a function that applies Sobel x and y, 
    # then computes the magnitude of the gradient
    # and applies a threshold
    def mag_threshold(self): 
        # 1) Calculate the magnitude 
        gradmag = np.sqrt(self.abs_sobelx**2 + self.abs_sobely**2)
        # 2) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        gradmag = cv2.normalize(gradmag, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        # 3) Create a binary mask where mag thresholds are met
        self.mag_binary = self.binary_thresh(gradmag, self.mag_thresh)

    # Define a function that applies Sobel x and y, 
    # then computes the direction of the gradient
    # and applies a threshold.
    def dir_threshold(self):
        # 1) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        absgraddir = np.arctan2(self.abs_sobely, self.abs_sobelx)
        # 2) Create a binary mask where direction thresholds are met
        self.dir_binary = self.binary_thresh(absgraddir, self.dir_thresh)

    # Define a function to apply all gradient transforms
    def all_grad_filter(self, img):
        self.abs_sobelxy(img)
        # Apply each of the thresholding functions
        self.abs_sobel_thresh()
        self.abs_sobel_thresh()
        self.mag_threshold()
        self.dir_threshold()

        self.combined = np.zeros_like(self.dir_binary)
        self.combined[((self.gradx == 1) & (self.grady == 1)) | ((self.mag_binary == 1) & (self.dir_binary == 1))] = 1

    # Use color transforms, gradients, etc., to create a thresholded binary image
    def combined(self, img):  

        # Apply color transform
        self.color_transform(img)
        self.all_grad_filter(self.color_transformed)   
        
        return self.combined
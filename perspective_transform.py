import numpy as np
import cv2

# Define a function that takes an image, number of x and y points, 
# camera matrix and distortion coefficients
def unwarp(img, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    # Grab the image shape
    # For source points I'm grabbing the outer four detected corners
    src = np.float32([[433, 563], [866, 563], [1041, 675], [280, 675]])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = np.float32([[280, 565], [1042, 563], [1041, 675], [280, 675]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    warped = cv2.warpPerspective(undist, M, (img.shape[1], img.shape[0]))
    return warped, M, Minv
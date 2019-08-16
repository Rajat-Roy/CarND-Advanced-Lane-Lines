import numpy as np
import cv2
import glob

# Define a class to implement camera calibration, camera transforms and remove distortion
class Camera:    
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.src = None
        self.dst = None
        self.M = None
        self.Minv = None
        
    # sets camera perspective matrices 
    def set_matrix(self, _src, _dst):
        self.src = _src
        self.dst = _dst
        # Given src and dst points, calculate the perspective transform matrix
        self.M = cv2.getPerspectiveTransform(_src, _dst)
        self.Minv = cv2.getPerspectiveTransform(_dst, _src)
        
    # performs the camera calibration    
    def calibrate(self, images_path, nx, ny): 
        # Make a list of calibration images (chessboard images)
        images = glob.glob(images_path)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

    # returns the undistorted image (remove lens distortion)
    def undistort(self, img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist
    
    # returns the warped image (bird eye view) from camera perspective
    def warp(self, img):
        warped = cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]))
        return warped
    
    # returns the camera perspective from warped image (bird eye view)
    def unwarp(self, img):
        unwarped = cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0]))
        return unwarped
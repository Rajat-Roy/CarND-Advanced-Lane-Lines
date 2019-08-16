import numpy as np
import cv2

import lane
import transformer
import threshold

from moviepy.editor import VideoFileClip

def process_image(image, laneProcessor, camera, thres_filter):
     
    # Apply a distortion correction to raw image.
    undistorted = camera.undistort(image)
    
    # Use color transforms, gradients, etc., to create a thresholded binary image.    
    thres_filter.color_transform(undistorted)
    thres_filter.all_grad_filter(thres_filter.color_transformed)
    filtered_img = thres_filter.combined
    
    # Apply a perspective transform to rectify binary image ("birds-eye view").
    warped = camera.warp(filtered_img)
    
    # Detect lane pixels and fit to find the lane boundary.
    laneProcessor.detect_lane(undistorted, warped)
    
    # Draw lane area using best fit
    laneProcessor.draw_vid(camera.Minv)
    result = laneProcessor.out_frame    
    laneProcessor.next_frame()    
    
    return result
        
def execute():
    
    # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
    camera = transformer.Camera()
    camera.calibrate('camera_cal/calibration*.jpg', 9, 6)
    
    # Calculate Camera matrices
    src = np.float32([[433, 563], [866, 563], [1041, 675], [280, 675]])
    dst = np.float32([[280, 565], [1042, 563], [1041, 675], [280, 675]])
    camera.set_matrix(src, dst)

    thres_filter = threshold.Filter()
    laneProcessor = lane.LaneProcessor()
    
    output = 'output_videos/project_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    
    out_clip = clip1.fl_image(lambda image: process_image(image, laneProcessor,
                                                          camera, 
                                                          thres_filter)).subclip(40,45)#NOTE: this function expects color images!!
    out_clip.write_videofile(output, audio=False)
    return output
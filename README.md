**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.jpg "undistort_output"
[image2]: ./test_images/signs_vehicles_xygrad.png "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"

[image7]: ./examples/undistorted.jpg "undistorted"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Note: Most of the code has adapted from course lessons and quizzes

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrate()` method of the `Camera` class located in [./transformer.py](https://github.com/Rajat-Roy/CarND-Advanced-Lane-Lines/blob/master/transformer.py)

```python
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
```

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to one of the calibration image using the `cv2.undistort()` function which is used in the `undistort()` method of the `Camera` class:

```python
# returns the undistorted image (remove lens distortion)
def undistort(self, img):
    undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    return undist
```
and obtained this result: 

![alt text][image1]



### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to this image.

(note: this is not actually from one of the test images):

![alt text][image2]

Code cell 3 of the ipynb notebook contains the code:
```python
test_img = mpimg.imread('test_images/signs_vehicles_xygrad.png')
undistorted = camera.undistort(test_img)

# Plotting thresholded images
f, axes = plt.subplots(1, 2, figsize=(20,10))
axes[0].set_title('Original')
axes[0].imshow(test_img)

axes[1].set_title('Corrected')
axes[1].imshow(undistorted)
```
and the result is

![alt text][image7]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.
Thresholding methods are defined in the `Filter` class of the [./threshold.py](https://github.com/Rajat-Roy/CarND-Advanced-Lane-Lines/blob/master/threshold.py) file.
It is applied in the code cell 4 of the ipynb notebook.

```python
import threshold
thresh_filter = threshold.Filter()

test_img = mpimg.imread('test_images/test6.jpg')
undistorted = camera.undistort(test_img)
thresh_filter.color_transform(undistorted)

# Plotting thresholded images
f, axes = plt.subplots(1, 2, figsize=(20,10))
axes[0].set_title('Original')
axes[0].imshow(test_img)

axes[1].set_title('Color Transformed Binary')
axes[1].imshow(thresh_filter.color_transformed, cmap='gray')
```

Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is defined in the `Camera` class.  I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32([[433, 563], [866, 563], [1041, 675], [280, 675]])
dst = np.float32([[280, 565], [1042, 563], [1041, 675], [280, 675]])
```
I used the `set_matrix()` and `warp()` method in the following manner in code cell 2 and 6 of the ipynb notebook:

```python
import numpy as np

src = np.float32([[433, 563], [866, 563], [1041, 675], [280, 675]])
dst = np.float32([[280, 565], [1042, 563], [1041, 675], [280, 675]])
camera.set_matrix(src, dst)

print("\nCamera Warp Matrix M: \n{0}".format(camera.M))

print("\nCamera Inverse Warp Matrix Minv: \n{0}".format(camera.Minv))
```
```python
warped = camera.warp(thresh_filter.combined)
warped[warped > 0] = 255

# Plotting thresholded images
f, axes = plt.subplots(1, 2, figsize=(20,10))
axes[0].set_title('Gradient Filtered')
axes[0].imshow(thresh_filter.combined, cmap='gray')

axes[1].set_title('Perspective transformed ("birds-eye view")')
axes[1].imshow(warped, cmap='gray')
```
This resulted in the following `M` and `Minv` matrices:

```
Camera Warp Matrix M: 
[[-6.19878012e-01 -1.53260191e+00  1.03016610e+03]
 [ 1.00508686e-02 -1.99415747e+00  9.14361435e+02]
 [ 1.48901756e-05 -2.42895990e-03  1.00000000e+00]]

Camera Inverse Warp Matrix Minv: 
[[ 1.81208826e-01 -7.74750723e-01  5.21726994e+02]
 [ 2.84780232e-03 -5.07549263e-01  4.61149763e+02]
 [ 4.21896640e-06 -1.22128064e-03  1.00000000e+00]]
```


I verified that my perspective transform was working as expected by applying it on a thresholded test image to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane-pixel methods are defined in the `LaneProcessor` class of the  [./lane.py](https://github.com/Rajat-Roy/CarND-Advanced-Lane-Lines/blob/master/lane.py) file.
I used the `find_lane_pixels()` method to find the fist set of lane pixels, which uses the window technique to find pixels from thresholded binary image and fits a second order polynomial.

Then I used the `search_around_poly()` method to find the lane pixels of the subsequent frames using previous fits and get new fits.

To verify that it is working as expected, I applied it on a test image in the 7th code cell of the ipynb notebook as follows:

```python
import lane
import cv2
laneProcessor = lane.LaneProcessor()
laneProcessor.detect_lane(undistorted, warped)
lane_pixels = laneProcessor.out_img

# Plotting thresholded images
f, axes = plt.subplots(1, 2, figsize=(20,10))
axes[0].set_title('Warped')
axes[0].imshow(warped, cmap='gray')

axes[1].set_title('Lane Pixels')
axes[1].imshow(lane_pixels)
```

and got the following result:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I defined the `measure_curvature_real()` and `curvature_from_fit()` methods for radius of curvature calculation in the `LaneProcessor` class.

Code cell 8 of the ipynb notebook demonstrates an example for a test image:

```python
left_curverad, right_curverad = laneProcessor.measure_curvature_real(laneProcessor.leftx, laneProcessor.lefty, 
                                                                 laneProcessor.rightx, laneProcessor.righty)
print("Radius of curvature: {0:.2f} (m)".format(np.minimum(left_curverad, right_curverad)))
```

The output is:

```
Radius of curvature: 572.65 (m)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I defined a `draw()` method in the `LaneProcessor` class to draw the lane area onto the undistorted image of the input image.
Here is a demonstration of the visualization implemented in code cell 9 of the ipynb notebook.

```python
laneProcessor.draw_img(camera.Minv)
result =laneProcessor.out_img


# Plotting thresholded images
f, axes = plt.subplots(1, 2, figsize=(20,10))
axes[0].set_title('Original')
axes[0].imshow(undistorted)

axes[1].set_title('Detected Lane Area')
axes[1].imshow(result)
```

The result is :

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

Although most of the functionalities were taken from chapter quizzes, some of the techniques had to be re-adjusted for better results.

#### Color thresholding

For some reason I was not satisfied with the thresholding of only S channel of the image. I opened up photoshop and played around and found that except for extreme cases, S and L channel together produces a better thresholding when used with some specific image blending.

I found the `Screen Blending` to be most useful. 
This [page](http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html) provided me with the formula:

```1 - (1-Target) * (1-Blend)```

The code is in the `Filter` class.

```python
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
```

#### Averaging data of previous frames

When using prior frames for extrapolation of n fits and n last lane positions, I found that for the first n frames I had to fill up rest of the un-processed n values upto the data available.

For example:
* if data upto frame 1 is available, fill 1st and rest (n-1)th data with frame 1 data
* if data upto frame 2 is available, fill frame 1 and frame 2 data in 1st and 2nd field respectively and then copy frame 2 data to the rest of (n-2) fields
* and so on ...

#### Limitations of the pipeline
This pipeline tends to fail in the following situations:
* Low lights like at night, under bridges, inside tunnels or heavy cloud
* Poorly  paited lanes
* Zig-zag curvy roads (like snakes)

#### Further improvements
This pipeline can definitely be improved by following considerations:
* Use of better thresholding
* Pluging in deep learning
* Use of stronger polynomial fits

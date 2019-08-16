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

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the calibrate method of the Camera class located in [./transformer.py](https://github.com/Rajat-Roy/CarND-Advanced-Lane-Lines/edit/master/transformer.py)

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

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to one of the calibration image using the `cv2.undistort()` function which is used in the undistort method of the Camera class:

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

To demonstrate this step, I will describe how I apply the distortion correction to this image 

(note: this is not actually from one of the test images):

![alt text][image2]

Cell 3 of the .ipynb contains the code:
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
Thresholding methods are defined in the Filter class of the [./threshold.py](https://github.com/Rajat-Roy/CarND-Advanced-Lane-Lines/edit/master/threshold.py) file.
It is applied in the cell 4 of the ipynb notebook.

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

The code for my perspective transform is defined in the Camera class.  I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32([[433, 563], [866, 563], [1041, 675], [280, 675]])
dst = np.float32([[280, 565], [1042, 563], [1041, 675], [280, 675]])
```
I used the set_matrix and warp method in the following manner in code cell 6 and 7 of the ipynb notebook:

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
This resulted in the following M and Minv matrices:

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

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

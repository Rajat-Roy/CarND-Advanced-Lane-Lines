import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.samples = 10
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = np.array(np.zeros(720*self.samples), np.float32).reshape(self.samples, 720)
        #average x values of the fitted line over the last n iterations
        self.bestx = None        
        #polynomial coefficients of the last n iterations
        self.recent_fits = np.array(np.zeros(3*self.samples), np.float32).reshape(self.samples, 3)
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = []  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None         
        self.recent_rcs = np.zeros(self.samples)
        self.best_rc = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        # iterartor
        self.count = 0
        
        self.frame_num = 0

class LaneProcessor:
    def __init__(self):
        
        self.image = None
        self.binary_warped = None
        self.out_img = None
        self.out_frame = None
        
        # Create line objects for tracking
        self.left_line = Line()
        self.right_line = Line()
        
        self.left_fit = None
        self.right_fit = None
        self.left_fitx = None
        self.right_fitx = None
        
        self.leftx = None
        self.lefty =  None
        self.rightx = None
        self.righty = None
        
    # Calculate the radius of curvature in meters for both lane lines
    # left_curverad, right_curverad = measure_curvature_real()
    def measure_curvature_real(self, leftx, lefty, rightx, righty):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # lefty, righty, leftx, rightx
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Start by generating our fake example data
        # Make sure to feed in your real data instead in your project!
        # ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)

        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        yleft_eval = np.max(lefty)
        yright_eval = np.max(righty)

        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*left_fit_cr[0]*yleft_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*yright_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return left_curverad, right_curverad

    # Calculate the radius of curvature in meters
    def curvature_fit(self, fit, y_vals, shape):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/shape[0] # meters per pixel in y dimension
        xm_per_pix = 3.7/shape[1] # meters per pixel in x dimension

        # x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+c from lesson
        if len(fit)>0:
            A = xm_per_pix/(ym_per_pix**2)*fit[0]
            B = (xm_per_pix/ym_per_pix)*fit[1]       
            f1= 2*A*np.max(y_vals)+B
            f2 = 2*A   
            curverad = ((1 + f1**2)**1.5) // np.absolute(f2)
        else:  
            curverad = 0
        return curverad
    
    def detect_lane(self, image, warped):
        self.image = image
        self.binary_warped = warped
        self.left_fit = self.left_line.current_fit
        self.right_fit = self.right_line.current_fit    
        self.leftx = self.left_line.allx
        self.lefty = self.left_line.ally 
        self.rightx = self.right_line.allx
        self.righty = self.right_line.ally

        if self.left_line.frame_num>3:
            if ((np.mean(self.right_line.bestx) - np.mean(self.left_line.bestx))> 500):
                self.search_around_poly()
            else:
                self.find_lane_pixels()
                self.fit_poly()
        else:
            self.find_lane_pixels()
            self.fit_poly()

        # Store values into Line Class
        self.left_line.detected = True
        self.right_line.detected = True

        #radius of curvature of the line in meters
        self.left_line.radius_of_curvature = self.curvature_fit(self.left_line.best_fit, self.lefty, self.binary_warped.shape)
        self.right_line.radius_of_curvature = self.curvature_fit(self.right_line.best_fit, self.righty, self.binary_warped.shape)

        # Initialize past value arrays with initial value 
        self.init_line()

        # Average last n xfits
        self.left_line.bestx = np.mean(self.left_line.recent_xfitted, axis=0).astype(np.float64)
        self.right_line.bestx = np.mean(self.right_line.recent_xfitted, axis=0).astype(np.float64)

        #polynomial coefficients averaged over the last n iterations
        self.left_line.best_fit = np.mean(self.left_line.recent_fits, axis=0).astype(np.float64)
        self.right_line.best_fit = np.mean(self.right_line.recent_fits, axis=0).astype(np.float64) 

        #distance in meters of vehicle center from the line
        self.left_line.line_base_pos = (self.binary_warped.shape[1]/2 - np.mean(self.left_line.bestx))*(3.7/self.binary_warped.shape[1])
        self.right_line.line_base_pos = (np.mean(self.right_line.bestx) - self.binary_warped.shape[1]/2)*(3.7/self.binary_warped.shape[1])

        #polynomial coefficients for the most recent fit
        self.left_line.current_fit = self.left_fit
        self.right_line.current_fit = self.right_fit   


        # Average of last n Radius of curvature
        self.left_line.best_rc = np.mean(self.left_line.recent_rcs)
        self.right_line.best_rc = np.mean(self.right_line.recent_rcs)


        #difference in fit coefficients between last and new fits
        self.left_line.diffs = self.left_line.recent_fits[self.left_line.count-1] - self.left_line.recent_fits[self.left_line.count-2]
        self.right_line.diffs = self.right_line.recent_fits[self.right_line.count-1] - self.right_line.recent_fits[self.right_line.count-2] 

        #x values for detected line pixels
        self.left_line.allx = self.leftx
        self.right_line.allx = self.rightx

        #y values for detected line pixels
        self.left_line.ally = self.lefty
        self.right_line.ally = self.righty    
        

    def find_lane_pixels(self):  
        binary_warped = cv2.normalize(self.binary_warped, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds] 
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]
        # save output
        self.out_img = cv2.normalize(out_img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    
    def init_line(self):
        if self.left_line.frame_num<self.left_line.samples:
            for i in range(self.left_line.samples - self.left_line.frame_num):

                index = self.left_line.frame_num + i           

                self.left_line.recent_xfitted[index] = np.array(self.left_fitx)
                self.left_line.recent_fits[index] = np.array(self.left_fit)
                self.left_line.recent_rcs[index] = self.left_line.radius_of_curvature
                
        if self.right_line.frame_num<self.right_line.samples:
            for i in range(self.right_line.samples - self.right_line.frame_num):

                index = self.right_line.frame_num + i           

                self.right_line.recent_xfitted[index] = np.array(self.right_fitx)
                self.right_line.recent_fits[index] = np.array(self.right_fit)
                self.right_line.recent_rcs[index] = self.right_line.radius_of_curvature

        # store last n xfitts and
        # polynomial coefficients of the last n iterations        
        if self.left_line.count < self.left_line.samples:        
            self.left_line.recent_xfitted[self.left_line.count] = np.array(self.left_fitx)
            self.left_line.recent_fits[self.left_line.count] = np.array(self.left_fit)
            self.left_line.recent_rcs[self.left_line.count] = self.left_line.radius_of_curvature
            self.left_line.count += 1
        else:
            self.left_line.count = 0
            
        if self.right_line.count < self.right_line.samples:        
            self.right_line.recent_xfitted[self.right_line.count] = np.array(self.right_fitx)
            self.right_line.recent_fits[self.right_line.count] = np.array(self.right_fit)
            self.right_line.recent_rcs[self.right_line.count] = self.right_line.radius_of_curvature
            self.right_line.count += 1
        else:
            self.right_line.count = 0

    def fit_poly(self):
         ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
        try:
            self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
            self.right_fit = np.polyfit(self.righty, self.rightx, 2)
        except TypeError:
            self.left_fit = np.array([1,1,1])
            self.right_fit = np.array([1,1,1])

        # Generate x and y values for plotting
        ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0])
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        self.left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

    def curvature_from_fit(fit, y_vals, shape):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/shape[0] # meters per pixel in y dimension
        xm_per_pix = 3.7/shape[1] # meters per pixel in x dimension

        # x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+c from lesson
        if len(fit)>0:
            A = xm_per_pix/(ym_per_pix**2)*fit[0]
            B = (xm_per_pix/ym_per_pix)*fit[1]       
            f1= 2*A*np.max(y_vals)+B
            f2 = 2*A   
            curverad = ((1 + f1**2)**1.5) // np.absolute(f2)
        else:  
            curverad = 0
        return curverad

    # Polynomial fit values from the previous frame
    # Make sure to grab the actual values from the previous step in your project!
    def search_around_poly(self):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 100

        # Grab activated pixels
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                        self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
                        self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                        self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
                        self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds] 
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        self.fit_poly()

    def draw(self, Minv):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Generate x and y values for plotting
        ploty = np.linspace(0, self.binary_warped.shape[0]-1,  self.binary_warped.shape[0])
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_line.bestx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.bestx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (self.image.shape[1], self.image.shape[0]))
        newwarp = cv2.normalize(newwarp, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        self.image = cv2.normalize(self.image, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

        # Combine the result with the original image
        result = cv2.addWeighted(self.image, 1, newwarp, 0.4, 0)
        result[result>1] = 1
        result = cv2.normalize(result, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)



        # Print Radius of Curvature
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = ((self.image.shape[1]//2),50)
        fontScale              = 0.8
        fontColor              = (255,255,255)
        lineType               = 2

        cv2.putText(result, 'Radius of Curvature: '+ str(np.minimum(self.left_line.best_rc, self.right_line.best_rc)) + ' (m)', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        # Print Position
        car_position = np.absolute(self.right_line.line_base_pos - self.left_line.line_base_pos)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = ((self.image.shape[1]//5),50)
        fontScale              = 0.8
        fontColor              = (255,255,255)
        lineType               = 2


        cv2.putText(result, 'Vehicle position: {0:.2f} (m)'.format(car_position), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        self.out_frame = result

    def draw_img(self, Minv):
        self.draw(Minv)
        self.out_img = cv2.normalize(self.out_frame, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        
    def draw_vid(self, Minv):
        self.draw(Minv)
    
    def next_frame(self):
        self.left_line.frame_num +=1
        self.right_line.frame_num +=1
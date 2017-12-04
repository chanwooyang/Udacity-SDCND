# Define a class to receive the characteristics of each line detection
import numpy as np

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = None 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # Lane index
        self.index = None

    def ewma_filter(self,filtered_valued,new_value,beta):

        if filtered_valued == None: # Initialize
            filtered_valued = new_value
        else:
            filtered_valued = beta*filtered_valued+(1-beta)*new_value

        return filtered_valued

    def reset(self):
        # Reset to initial states except best_fit
        self.detected = False  
        self.recent_xfitted = None 
        self.bestx = None     
        # self.best_fit = None  
        self.current_fit = [np.array([False])]  
        # self.radius_of_curvature = None 
        # self.line_base_pos = None 
        self.diffs = np.array([0,0,0], dtype='float') 
        # self.index = None
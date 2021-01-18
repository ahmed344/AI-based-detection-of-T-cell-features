# Libraries
import numpy as np
import pandas as pd
from skimage.transform import resize
from scipy import ndimage
from scipy.optimize import minimize

# Utilities
from fitting import Fit_Gaussian



class Lattice():
    
    def __init__(self, coordinates, l_min = 0.5, l_max = 1.3, showing = False):
        
        # Coordinates of the measured lattice.
        coord = pd.read_csv(coordinates)
        self.coordinates = np.dstack((coord["XM"], coord["YM"]))[0]       
        
        # Maximum and minimum number of pixels between first nhds.
        self.l_max = l_max
        self.l_min = l_min
        
        # Showing the histograms of pitch and angle.
        self.showing = showing                            

    def pitch_angle(self):

        # Make a list of all the pitches and angles
        pitches, angles = [], []
        for p in self.coordinates:
            for q in self.coordinates:
                if p[0] != q[0]:
                    d = np.linalg.norm(p-q)
                    if d < self.l_max and d > self.l_min:
                        pitches.append(d)
                        if (p[1]-q[1]) > 0:
                            angles.append(np.arccos((p[0]-q[0])/d)*(180/np.pi))
                        #else:
                            #angles.append(-np.arccos((p[0]-q[0])/d)*(180/np.pi))

        # Transform the lists of all the pitches and angles int numpy array
        pitches = np.array(pitches)
        angles = np.array(angles)

        # Find the mean of the gaussian fitted on the pitches histogram
        try:
            gauss = Fit_Gaussian(pitches)
            pitch, _, _, _ = gauss.hist_fitting(show = self.showing)
        except:
            print('Warning: failed to fit a general Gaussian on the pitch histogram')
            gauss = Fit_Gaussian(pitches, normalized = True)
            pitch, _ = gauss.hist_fitting(show = self.showing)
        
        # Chose a peak to fit a gaussian on
        n, bins = np.histogram([angle for angle in angles if angle > 30 and angle < 130], bins = 100)
        th = n.argmax() + 30

        # Find the mean of the gaussian fitted on the angles chosen peak
        try:
            gauss = Fit_Gaussian([angle for angle in angles if angle > th - 30 and angle < th + 30])
            angle, _, _, _ = gauss.hist_fitting(show = self.showing)
        except:
            print('Warning: failed to fit a general Gaussian on the angle histogram')
            gauss = Fit_Gaussian([angle for angle in angles if angle > th - 30 and angle < th + 30], normalized = True)
            angle, _= gauss.hist_fitting(show = self.showing)

        return pitch, angle

    
    def corners(self):
        
        # The X and Y component of the coordinates measured by analyze particles
        x_max, y_max = self.coordinates.max(axis = 0)
        x_min, y_min = self.coordinates.min(axis = 0)
        
        return x_max, y_max, x_min, y_min
    

    def basis(self):
        
        # Compute the pitch and the angle
        a, angle = self.pitch_angle()
        
        # Compute the two angles for the two basis vectors
        angle1 = angle * (np.pi/180)
        angle2 = (angle + 60) * (np.pi/180)

        # Find the two basis vectors
        a1 = np.array([a * np.cos(angle1), a * np.sin(angle1)])
        a2 = np.array([a * np.cos(angle2), a * np.sin(angle2)])

        return a1, a2
    
    
    def simulation_error(self, parameters):
        
        # The parameters
        X0, Y0, a, angle = parameters
        
        # Compute the origin
        origin = np.array([X0, Y0])
        
        # Compute the two angles for the two basis vectors
        angle1 = angle * (np.pi/180)
        angle2 = (angle + 60) * (np.pi/180)

        # Find the two basis vectors
        a1 = np.array([a * np.cos(angle1), a * np.sin(angle1)])
        a2 = np.array([a * np.cos(angle2), a * np.sin(angle2)])

        # Find the corners
        x_max, y_max, x_min, y_min = self.corners()
        
        lx = int((x_max - x_min)/a + self.l_max)
        ly = int((y_max - y_min)/a + self.l_max)
        
        simulated_dots, point = [], []
        for n in range(-lx, lx):
            for m in range(-ly, ly):
                point = origin + n * a1 + m * a2
                if x_min < point[0] < x_max and y_min < point[1] < y_max:
                    simulated_dots.append(point)
                    
        simulated_dots = np.array(simulated_dots)
                    
        return np.sum([np.linalg.norm(dot - self.coordinates, axis=1).min() for dot in simulated_dots])
    
    
    def simulation(self, parameters):
        
        # The parameters
        X0, Y0, a, angle = parameters
        
        # Compute the origin
        origin = np.array([X0, Y0])
        
        # Compute the two angles for the two basis vectors
        angle1 = angle * (np.pi/180)
        angle2 = (angle + 60) * (np.pi/180)

        # Find the two basis vectors
        a1 = np.array([a * np.cos(angle1), a * np.sin(angle1)])
        a2 = np.array([a * np.cos(angle2), a * np.sin(angle2)])

        # Find the corners
        x_max, y_max, x_min, y_min = self.corners()
        
        lx = int((x_max - x_min)/a + self.l_max)
        ly = int((y_max - y_min)/a + self.l_max)
        
        simulated_dots, point = [], []
        for n in range(-lx, lx):
            for m in range(-ly, ly):
                point = origin + n * a1 + m * a2
                if x_min < point[0] < x_max and y_min < point[1] < y_max:
                    simulated_dots.append(point)
                    
        return np.array(simulated_dots)

    
    def simulate(self, dots = 'on'):
        
        # Estimate the pitch and angle
        a, angle = self.pitch_angle()

        # Find the corners
        x_max, y_max, x_min, y_min = self.corners()
        
        # Minimize the lattice simulation error
        res = minimize(self.simulation_error,
                       [x_max/2, y_max/2, a, angle],
                       bounds=[(x_max/2 - self.l_max/2, x_max/2 + self.l_max/2),
                               (y_max/2 - self.l_max/2, y_max/2 + self.l_max/2),
                               (a-0.01, a+0.01), (angle-1, angle+1)])
        
        if dots == 'on':
            return self.simulation(res.x)
        
        elif dots == 'off' or dots == 'both':
            # The optimized parameters
            X0, Y0, a, angle = res.x
            
            # Compute the optimized angles and basis vector
            angle1 = angle * (np.pi/180)
            a1 = np.array([a * np.cos(angle1), a * np.sin(angle1)])
            
            # Comput the shift vector
            v = 0.5*a1 + (np.sqrt(3)/6) * np.array([-a * np.sin(angle1), a * np.cos(angle1)])
            shift = [v[0], v[1], 0, 0]
            
            if dots == 'off':
                return self.simulation(res.x + shift)
            
            else:
                return self.simulation(res.x), self.simulation(res.x + shift)
        
        else:
            print('Error: (dots = {}) is not defined'.format(dots))



class Squares():
    
    def __init__(self, img, coordinates, l = 15, img_scale = 1):
        
        # The image to be cut into mini images
        self.img = img
        
        # Coordinates of the mini images scaled with the image tob cut.
        self.coordinates = coordinates * img_scale
        
        # Dimension of the mini images
        self.l = l
        
    
    def squares(self):
        
        # Compute the maximum coordinates
        MAX = self.coordinates.max(axis = 0)

        # Make a stack of mini images excluding the edges
        dots = []
        for p in self.coordinates:
            if int(self.l/2+1) < p[0] < MAX[0] - int(self.l/2+1) and int(self.l/2+1) < p[1] < MAX[1] - int(self.l/2+1):  
                dots.append(self.img[int(p[1])-int(self.l/2):int(p[1])+int(self.l/2+1),
                                     int(p[0])-int(self.l/2):int(p[0])+int(self.l/2+1)])
                
        return np.array(dots)
    
    @classmethod
    def mask_reduction(self, mask, remove = 'center', center_ratio = 4):
        # Normalization of the mask
        mask = mask/mask.max()
    
        # Resize the mask.
        mask_resized = resize(mask, (int(mask.shape[0]/center_ratio), int(mask.shape[1]/center_ratio)), order=0)
    
        # Measure the center of mass for the mask and for the resized mask.
        center_of_mask = ndimage.measurements.center_of_mass(mask)
        center_of_mask_resized = ndimage.measurements.center_of_mass(mask_resized)
    
        # Adjust the center of mass of the resized mask to be over the center of mass of the mask
        y = int(center_of_mask[0] - center_of_mask_resized[0])
        x = int(center_of_mask[1] - center_of_mask_resized[1])
    
        # Remove either the center or the edge
        if remove == 'center':
            mask[y:y+mask_resized.shape[0], x:x+mask_resized.shape[1]] -= mask_resized
        elif remove == 'edge':
            mask -= mask 
            mask[y:y+mask_resized.shape[0], x:x+mask_resized.shape[1]] += mask_resized
            
        return mask

    
    def masked_squares(self, mask, ratio = 1., remove = None, center_ratio = 4):
        
        # Normalization of the mask
        mask = mask/mask.max()
        
        # Get only the center or the edge of the mask
        if remove == 'center' or remove == 'edge':
            mask = self.mask_reduction(mask, remove = remove, center_ratio = center_ratio)
        
        # Compute the maximum coordinates
        MAX = self.coordinates.max(axis = 0)

        # Make a stack of mini images excluding the edges
        dots = []
        for p in self.coordinates:
            if int(self.l/2+1) < p[0] < MAX[0] - int(self.l/2+1) and int(self.l/2+1) < p[1] < MAX[1] - int(self.l/2+1):
                if mask[int(p[1])-int(self.l/2):int(p[1])+int(self.l/2+1), int(p[0])-int(self.l/2):int(p[0])+int(self.l/2+1)].mean() >= ratio:
                    dots.append(self.img[int(p[1])-int(self.l/2):int(p[1])+int(self.l/2+1),
                                         int(p[0])-int(self.l/2):int(p[0])+int(self.l/2+1)])
                
        return np.array(dots)
    
    

class Binary_Squares():
    
    def __init__(self, img, coordinate, l = 15, img_scale = 1, l_min = 0, l_max = 2.5):
        
        # The image to be cut into mini images
        self.img = img
        
        # Coordinates of the mini images scaled with the image tob cut.
        self.coordinate = coordinate
        
        # the scale of the image to the coordinate
        self.img_scale = img_scale
        
        # Maximum and minimum number of pixels between first nhds.
        self.l_max = l_max
        self.l_min = l_min
        
        # Dimension of the mini images
        self.l = l
        
    def squares(self):
        
        # Localize each dot on the lattice and make shifted set of dots
        lattice = Lattice(coordinates = self.coordinate, l_min = self.l_min, l_max = self.l_max)
        simulated_dots, shifted_simulated_dots = lattice.simulate(dots = 'both')
        
        # Make the mini images on dots
        mini_images = Squares(img = self.img, coordinates = simulated_dots, l = self.l, img_scale = self.img_scale)
        dots = mini_images.squares()
        
        # Make the mini images off dots
        mini_images = Squares(img = self.img, coordinates = shifted_simulated_dots, img_scale = self.img_scale)
        shifted_dots = mini_images.squares()
        
        return dots, shifted_dots
    
    
    def masked_squares(self, mask, ratio = 1., remove = None, center_ratio = 4):
        
        # Localize each dot on the lattice and make shifted set of dots
        lattice = Lattice(coordinates = self.coordinate, l_min = self.l_min, l_max = self.l_max)
        simulated_dots, shifted_simulated_dots = lattice.simulate(dots = 'both')
        
        # Make the mini images on dots
        mini_images = Squares(img = self.img, coordinates = simulated_dots, l = self.l, img_scale = self.img_scale)
        dots = mini_images.masked_squares(mask, ratio = ratio, remove = remove, center_ratio = center_ratio)
        
        # Make the mini images off dots
        mini_images = Squares(img = self.img, coordinates = shifted_simulated_dots, img_scale = self.img_scale)
        shifted_dots = mini_images.masked_squares(mask, ratio = ratio, remove = remove, center_ratio = center_ratio)
        
        return dots, shifted_dots
        

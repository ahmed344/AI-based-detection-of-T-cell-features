import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Fit_Gaussian():
    
    def __init__(self, data, normalized = False):
        
        self.data = data                     # The image to the Gaussian on.
        self.normalized = normalized       # Using normalized Gaussian. 

    # Define a gaussian
    def Gauss(x, x0, sigma, y0, A):
        return y0 + A * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    # Define a normalized gaussian
    def Gauss_normalized(x, x0, sigma):
        return (1/np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x - x0)**2 / (2 * sigma**2))

    # Fit a gaussian on a histogram
    def hist_fitting(self, bins = 200, show = False):
        """Get a gaussian fitting of a histogram.
        Parameter:
            data - as numpy array
        Returns: 
            X0, sigma, Y0, A - of the histogram gaussian
            or
            X0, sigma - of the normalized histogram gaussian
        """
        # Make a histogram
        if self.normalized == True:
            n, bins_ = np.histogram(self.data, bins=bins, density = True)
        else:
            n, bins_ = np.histogram(self.data, bins=bins)


        # Data
        x = np.linspace(bins_.min(),bins_.max(),bins_.shape[0]-1)
        y = n

        # Apply the fitting 
        if self.normalized == True:
            popt,pcov = curve_fit(Fit_Gaussian.Gauss_normalized, x, y,
                                  p0 = (x.max()/2, x.max()/3),
                                  bounds = (0, [x.max(), x.max()/2]))
        else:
            popt,pcov = curve_fit(Fit_Gaussian.Gauss, x, y,
                                  p0 = (x.max()/2, x.max()/3, 0, 1/np.sqrt(2 * np.pi * (x.max()/3)**2)),
                                  bounds = (0, [x.max(), x.max()/2, np.inf, np.inf]))            

        # Display the results
        if show == True:
            plt.figure(figsize=(8,5))
            
            if self.normalized == True:
                plt.plot(x, Fit_Gaussian.Gauss_normalized(x, *popt), 'r-',
                         label='Gauss: $x_0$ = {:.4f}, $\sigma$ = {:.4f}'.format(*popt))
            else:
                plt.plot(x, Fit_Gaussian.Gauss(x, *popt), 'r-',
                         label='Gauss: $x_0$ = {:.4f}, $\sigma$ = {:.4f}, $Y_0$ = {:.4f}, A = {:.4f}'.format(*popt))
            
            plt.plot(x, y, 'b+:', label='data')            
            plt.legend()
            plt.title('Histogram Gaussian')
            plt.xlabel('value')
            plt.ylabel('frequency')
            plt.grid()

        return popt  #(X0, sigma, Y0, A) or just (X0, sigma) for normalized Gaussian


class Fit_2D_Gaussian():
    
    def __init__(self, img, normalized = False, symmetric = False):
        
        self.img = img                     # The image to the Gaussian on.
        self.normalized = normalized       # Using normalized Gaussian. 
        self.symmetric = symmetric         # Using symmetric Gaussian or not.
    
    
    def gaussian_2D(mesh, x0, y0, sigma_x, sigma_y, A, z0):

        # get x and y from the mesh
        x, y = mesh

        # The 2 dimensional Gaussian
        gaussian = z0 + A * np.exp(-((x-x0)**2)/(2*sigma_x**2) - ((y-y0)**2)/(2*sigma_y**2))

        # Return 2D Gaussian function as 1D array
        return gaussian.ravel()
    
    
    def gaussian_2D_symmetric(mesh, x0, y0, sigma, A, z0):

        # get x and y from the mesh
        x, y = mesh

        # The 2 dimensional Gaussian
        gaussian = z0 + A * np.exp(-((x-x0)**2)/(2*sigma**2) - ((y-y0)**2)/(2*sigma**2))

        # Return 2D Gaussian function as 1D array
        return gaussian.ravel()


    def gaussian_2D_normalized(mesh, x0, y0, sigma_x, sigma_y):

        # get x and y from the mesh
        x, y = mesh

        # The 2 dimensional Normalized Gaussian
        gaussian = (1/(2*np.pi*sigma_x*sigma_y)) * np.exp(-((x-x0)**2)/(2*sigma_x**2) - ((y-y0)**2)/(2*sigma_y**2))

        # Return 2D Gaussian function as 1D array
        return gaussian.ravel()

    
    def gaussian_2D_normalized_symmetric(mesh, x0, y0, sigma):

        # get x and y from the mesh
        x, y = mesh

        # The 2 dimensional Normalized Gaussian
        gaussian = (1/(2*np.pi*sigma*sigma)) * np.exp(-((x-x0)**2)/(2*sigma**2) - ((y-y0)**2)/(2*sigma**2))

        # Return 2D Gaussian function as 1D array
        return gaussian.ravel()

    
    def fitting(self):
        """Get a 2D gaussian fitting of an image.
        Parameter:
            img - image as numpy array
        Returns: 
            img_gaussian, parameters
        """

        # Create the coordinates of the image
        img_mesh = np.meshgrid(np.linspace(0, self.img.shape[1], self.img.shape[1]),
                               np.linspace(0, self.img.shape[0], self.img.shape[0]))
        
        # Creat 2d indices for the gaussian image
        x, y = np.meshgrid(np.linspace(0, self.img.shape[1], self.img.shape[1]),
                           np.linspace(0, self.img.shape[0], self.img.shape[0]))

        if self.normalized == True:

            if self.symmetric == True:
                # Fit the Gaussian
                popt, pcov = curve_fit(Fit_2D_Gaussian.gaussian_2D_normalized_symmetric, img_mesh, self.img.ravel(),
                                       p0=(self.img.shape[1]/2, self.img.shape[0]/2, self.img.shape[1]/2),
                                       bounds=([0,0,0],
                                               [self.img.shape[1], self.img.shape[0], self.img.shape[1]/2]))

                # Determine the parameters of the Gaussian
                x0, y0, sigma = popt

                # Now compute the full width at half maximum for x and y
                FWHM = np.abs(4*sigma * np.sqrt(-0.5*np.log(0.5)))

                # Return the parameters of a 2D Gaussian
                parameters = x0, y0, sigma, FWHM
                img_gaussian = (1/2*np.pi*sigma*sigma) * np.exp(-((x-x0)**2)/(2*sigma**2) - ((y-y0)**2)/(2*sigma**2))


            else:
                # Fit the Gaussian
                popt, pcov = curve_fit(Fit_2D_Gaussian.gaussian_2D_normalized, img_mesh, self.img.ravel(),
                                       p0=(self.img.shape[1]/2, self.img.shape[0]/2, 
                                           self.img.shape[1]/2, self.img.shape[0]/2),
                                       bounds=([0,0,0,0],
                                               [self.img.shape[1], self.img.shape[0], 
                                                self.img.shape[1]/2, self.img.shape[0]/2]))

                # Determine the parameters of the Gaussian
                x0, y0, sigma_x, sigma_y = popt

                # Now compute the full width at half maximum for x and y
                FWHM_x = np.abs(4*sigma_x * np.sqrt(-0.5*np.log(0.5)))
                FWHM_y = np.abs(4*sigma_y * np.sqrt(-0.5*np.log(0.5)))

                # Return the parameters of a 2D Gaussian
                parameters = x0, y0, sigma_x, sigma_y, FWHM_x, FWHM_y
                img_gaussian = (1/2*np.pi*sigma_x*sigma_y) * np.exp(-((x-x0)**2)/(2*sigma_x**2) - ((y-y0)**2)/(2*sigma_y**2))


        else:

            if self.symmetric == True:
                # Fit the Gaussian
                popt, pcov = curve_fit(Fit_2D_Gaussian.gaussian_2D_symmetric, img_mesh, self.img.ravel(),
                                       p0=(self.img.shape[1]/2, self.img.shape[0]/2, self.img.shape[1]/2,1,0),
                                       bounds=([0,0,0, -np.inf, -np.inf],
                                               [self.img.shape[1], self.img.shape[0], self.img.shape[1]/2,
                                                np.inf, np.inf]))

                # Determine the parameters of the Gaussian
                x0, y0, sigma, A, z0 = popt

                # Now compute the full width at half maximum for x and y
                FWHM = np.abs(4*sigma * np.sqrt(-0.5*np.log(0.5)))

                # Return the parameters of a 2D Gaussian
                parameters = x0, y0, sigma, A, z0, FWHM
                img_gaussian = z0 + A * np.exp(-((x-x0)**2)/(2*sigma**2) - ((y-y0)**2)/(2*sigma**2))

            else:
                # Fit the Gaussian
                popt, pcov = curve_fit(Fit_2D_Gaussian.gaussian_2D, img_mesh, self.img.ravel(),
                                       p0=( self.img.shape[1]/2, self.img.shape[0]/2, 
                                           self.img.shape[1]/2, self.img.shape[0]/2, 1, 0),
                                       bounds=([0,0,0,0, -np.inf, -np.inf],
                                               [self.img.shape[1], self.img.shape[0], 
                                                self.img.shape[1]/2, self.img.shape[0]/2, np.inf, np.inf]))

                # Determine the parameters of the Gaussian
                x0, y0, sigma_x, sigma_y, A, z0 = popt

                # Now compute the full width at half maximum for x and y
                FWHM_x = np.abs(4*sigma_x * np.sqrt(-0.5*np.log(0.5)))
                FWHM_y = np.abs(4*sigma_y * np.sqrt(-0.5*np.log(0.5)))

                # Return the parameters of a 2D Gaussian
                parameters = x0, y0, sigma_x, sigma_y, A, z0, FWHM_x, FWHM_y
                img_gaussian = z0 + A * np.exp(-((x-x0)**2)/(2*sigma_x**2) - ((y-y0)**2)/(2*sigma_y**2))

        return img_gaussian, parameters

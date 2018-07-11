import os # Access Directories
import numpy as np # Process Data
import imageio # Read/Write Images
import skimage.transform as skitrans
import skimage.io as skio #Features extraction
import skimage.color as skicol #Features extraction
import cv2
from scipy.stats.mstats import gmean
#import models.CAE as cae
from skimage.feature import greycomatrix, greycoprops

class ImagesProcessor():

    # Initialize the class variables
    def __init__(self):
        self.input_imagesDir = 'input/'
        self.output_imagesDir = 'output/'

    ##########################################################################################
    ################################## I/O operations ########################################
    ##########################################################################################

    # Read the filename image and return it as a numpy array
    def readImage(self, filename):
        return np.uint8(imageio.imread(filename))

    # Write the dir image from a numpy array
    def writeImage(self, img, dir):
        imageio.imwrite(dir, img[:, :, :])

    ##########################################################################################
    ################################## Images Manipulations ##################################
    ##########################################################################################

    # Resize Image to the correspondant shape
    def resizeImage(self, img, shape):
        #return skitrans.resize(img, shape)
        return cv2.resize(img , (shape[0], shape[1]))

    # Transform a 4 dimensional array into one single vector to feed the models
    def flattenImages(self, imgs):
        n = imgs.shape[0]
        dim = np.prod(imgs.shape[1:])
        flatten = imgs.reshape((n, dim))
        return flatten

    # Normalize the values of an array between min and max 
    def normalizeArray(self, array, min, max):
        minValue = np.min(array)
        maxValue = np.max(array)
        normalize = lambda t: (t - minValue)/(maxValue - minValue) * max - min
        return np.vectorize(normalize)(array)

    # Split the img into x new images with the specified shape and overlaping_gap in pixels
    # If the images dimensions doesn't fit perfectly with the shape and overlap desired
    # a bigger overlap is produce for the last slices.
    def sliceImage(self, img, newShapes, overlaping):
        result = []

        for i in range(0,img.shape[0],newShapes[0] - overlaping):
            for j in range(0,img.shape[1],newShapes[1] - overlaping):

                if(i + newShapes[0] >= len(img)):
                    x = len(img) - newShapes[0]
                else:
                    x = i

                if (j + newShapes[1] >= len(img[i])):
                    y = len(img[i]) - newShapes[1]
                else:
                    y = j

                tile = img[x:x+newShapes[0], y:y+newShapes[1],:]
                result.append(tile)

        result = np.array(result)
        return result

    ##########################################################################################
    ################################## Features Extraction ###################################
    ##########################################################################################

    def extractGrayHistogram(self,image):
        grayImage = skicol.rgb2grey(image)
        histogram, bins = np.histogram(grayImage, range=(0, 1))
        return np.array(histogram).astype('float')

    def extractRGBHistogram(self,image):
        colors = []
        for color in range(image.shape[2]):
            band = image[:, :, color].reshape(-1)
            values, bins = np.histogram(band, range=(0, 255))
            colors += list(values)
        histogram = np.array(colors).astype('float')
        return histogram

    def extractHUEHistogram(self,image):
        hue = skicol.rgb2hsv(image)[:, :, 0].reshape(-1)
        histogram, bins = np.histogram(hue, range=(0, 1))
        histogram = np.array(histogram).astype('float')
        return histogram

    # From https://stackoverflow.com/questions/47745541/shadow-removal-in-python-opencv/48875676
    def extractChromaticity(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        img = cv2.GaussianBlur(img, (5,5), 0)

        # Separate Channels
        r, g, b = cv2.split(img) 

        im_sum = np.sum(img, axis=2)
        im_mean = gmean(img, axis=2)

        # Create "normalized", mean, and rg chromaticity vectors
        #  We use mean (works better than norm). rg Chromaticity is
        #  for visualization
        n_r = np.ma.divide( 1.*r, g )
        n_b = np.ma.divide( 1.*b, g )

        mean_r = np.ma.divide(1.*r, im_mean)
        mean_g = np.ma.divide(1.*g, im_mean)
        mean_b = np.ma.divide(1.*b, im_mean)

        rg_chrom_r = np.ma.divide(1.*r, im_sum)
        rg_chrom_g = np.ma.divide(1.*g, im_sum)
        rg_chrom_b = np.ma.divide(1.*b, im_sum)

        # Visualize rg Chromaticity --> DEBUGGING
        rg_chrom = np.zeros_like(img)

        rg_chrom[:,:,0] = np.clip(np.uint8(rg_chrom_r*255), 0, 255)
        rg_chrom[:,:,1] = np.clip(np.uint8(rg_chrom_g*255), 0, 255)
        rg_chrom[:,:,2] = np.clip(np.uint8(rg_chrom_b*255), 0, 255)

        return rg_chrom

   # def extractCAEfeatures(self, image):
   #     autoencoder = cae.CAE(image.shape[1:],nbNeuronsLayers=[16, 8, 8], nbConvFilters=(3,3), poolScale=(2, 2))
   #     autoencoder.createModel()
   #     autoencoder.train(image, image, epochs=50)
   #     return autoencoder.extractFeatures(image)

    def extractTexturefeatures(self, image):
        grayImage = skicol.rgb2grey(image).astype('uint8')
        glcm = greycomatrix(grayImage, [5], [0], 256, symmetric=True, normed=True)
        return np.array([greycoprops(glcm, 'dissimilarity')[0, 0], 
                        greycoprops(glcm, 'homogeneity')[0, 0], 
                        greycoprops(glcm, 'ASM')[0, 0] , 
                        greycoprops(glcm, 'contrast')[0, 0] ,
                        greycoprops(glcm, 'energy')[0, 0] ,
                        greycoprops(glcm, 'correlation')[0, 0] ]).astype('float')

#if __name__ == "__main__":
    #IP = ImagesProcessor()
    #img = IP.readImage(IP.input_imagesDir + "test.jpg")
    #imgs = IP.splitImage(img, (512, 512, 3), 100)
    #for x in range(0,len(imgs)):
    #    IP.writeImage(imgs[x], IP.output_imagesDir + 'slice_'+str(x)+'.jpg')
    #print(imgs.shape)
    #test = [0,9,8,7,6,5,4,3,2,1, -3]
    #print(IP.normalizeArray(test, 0, 10))

    #img = IP.readImage(IP.input_imagesDir + "test.jpg")
    #print(IP.extractHueHistograms([img]))



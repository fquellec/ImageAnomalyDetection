import os # Access Directories
import numpy as np # Process Data
import imageio # Read/Write Images
import skimage.transform as skitrans
import skimage.io as skio #Features extraction
import skimage.color as skicol #Features extraction

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
        return imageio.imread(filename)

    # Write the dir image from a numpy array
    def writeImage(self, img, dir):
        imageio.imwrite(dir, img[:, :, :])

    ##########################################################################################
    ################################## Images Manipulations ##################################
    # Resize Image to the correspondant shape, Not Tested yet
    def resizeImage(self, img, shape):
        return skitrans.resize(img, shape)

    # Transform a 4 dimensional array into one single vector to feed the models
    def flattenImages(self, imgs):
        n = imgs.shape[0]
        dim = np.prod(imgs.shape[1:])
        flatten = imgs.reshape((n, dim))
        return flatten

    # Normalize the values of an array between min and max with 
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

    def extractGrayHistograms(self,images):
        assert len(images) > 0, 'No images to compute'

        histograms = []
        for image in images:
            grayImage = skicol.rgb2grey(image)
            histogram, bins = np.histogram(grayImage, range=(0, 1))
            histograms.append(histogram)
        histograms = np.array(histograms)
        histograms = histograms.astype('float')
        return histograms

    def extractRGBHistograms(self,images):
        assert len(images) > 0, 'No images to compute'

        histograms = []
        for image in images:
            colors = []
            for color in range(image.shape[2]):
                band = image[:, :, color].reshape(-1)
                values, bins = np.histogram(band, range=(0, 255))
                colors += list(values)
            histograms.append(colors)
        histograms = np.array(histograms)
        histograms = histograms.astype('float')
        return histograms

    def extractHueHistograms(self,images):
        assert len(images) > 0, 'No images to compute'

        histograms = []
        for image in images:
            hue = skicol.rgb2hsv(image)[:, :, 0].reshape(-1)
            histogram, bins = np.histogram(hue, range=(0, 1))
            histograms.append(histogram)
        histograms = np.array(histograms)
        histograms = histograms.astype('float')
        return histograms

    

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



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import os
import utils.ImagesProcessor as ip


TRAINING_RAW_PATH = "training/raw"
TRAINING_PATH = "training/processed"
INPUT_RAW_PATH = "input/raw"
INPUT_PATH = "input/processed"
OUTPUT_PATH = "output"
EXT_IMAGES = ".jpg"

IP = ip.ImagesProcessor()
InputShape = np.array([300, 300, 3])

directory = os.fsencode(TRAINING_RAW_PATH)
X_train = []
for file in os.listdir(directory):
	filename = os.fsdecode(file)
	if filename.lower().endswith(EXT_IMAGES): 
		img = IP.readImage(TRAINING_RAW_PATH + "/" + filename)
		img = IP.resizeImage(img, InputShape[:-1])	
		#colors_hist = IP.extractRGBHistogram(img)
		#texture = IP.
		chromaticity = IP.extractChromaticity(img)
		X_train.append(np.reshape(img, -1))

X_train = np.array(X_train)

# fit the model
clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.3)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
print(y_pred_train)

# Get and process the inputs
directory = os.fsencode(INPUT_RAW_PATH)
X_test = []
for file in os.listdir(directory):
	filename = os.fsdecode(file)
	if filename.lower().endswith(EXT_IMAGES): 
		img = IP.readImage(INPUT_RAW_PATH + "/" + filename)
		img = IP.resizeImage(img, InputShape[:-1])	
		colors_hist = IP.extractRGBHistogram(img)
		chromaticity = np.reshape(IP.extractChromaticity(img), -1)

		print(filename + " - " + str(clf.predict(np.reshape(img, (1, -1)))))
		#texture = IP.
		
		X_test.append(colors_hist)

#X_test = np.array(X_test)
#y_pred_test = clf.predict(X_test)
#print(y_pred_test)

'''
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                 edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                edgecolors='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers))
plt.show()
'''
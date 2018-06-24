import numpy as np

class KNN:
	def __init__(self):


	def euclidienne_distance(data1, data2):    
    return np.sum(np.power(np.array(instance1) - np.array(instance2), 2))
	

    def get_neighbors(training_set, 
                  test_instance, 
                  k, 
                  distance=euclidienne_distance):

    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 08:04:53 2023

@author: charles-arthurpacton
"""
import matplotlib.pyplot as plt
import copy
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans

k = 4
n_of_points = 60
n_of_iterations = 200




def split_points(points: np.array, n_of_point_groups: int) -> np.array:
    changed_points = copy.copy(points)
    index = np.arange(len(points))
    groups_index = np.split(index, n_of_point_groups)
    
    for id_group,group_index in enumerate(groups_index):
        changed_points[group_index] = points[group_index] + 5*id_group
    
    return changed_points

points = np.random.rand(n_of_points,2) * 5 
points = split_points(points, 3)


def initialize_clusters(points: np.array, k_clusters: int) -> np.array:
    vector_with_all_indexes = np.arange(points.shape[0])
    vector_with_all_indexes = np.random.permutation(vector_with_all_indexes)    
    required_indexes = vector_with_all_indexes[:k_clusters] 
    
    return points[required_indexes]

def calculate_metric(points: np.array, centroid: np.array) -> np.array:
    return np.square(norm(points-centroid, axis=1))

def compute_distances(points: np.array, centroids_points: np.array) -> np.array:
    return np.asarray([calculate_metric(points, centroid) for centroid in centroids_points])

def assign_centroids(distances: np.array):
    return np.argmin(distances, axis = 1)


def calculate_objective(cluster_belongs: np.array, distances: np.array) -> np.array:
    distances = distances.T
    selected_min =  distances[np.arange(len(distances)), cluster_belongs]
    return np.sum(selected_min)
    

def calculate_new_centroids(points: np.array, clusters_belongs: np.array, n_of_clusters: int) -> np.array:
    new_clusters = []
    for cluster_id in range(n_of_clusters):
        j = np.where(clusters_belongs == cluster_id)
        points_sel = points[j]
        new_clusters.append(np.mean(points_sel, axis=0))

    return np.array(new_clusters)

def fit(points: np.array, n_of_centroids: int, n_of_oterations: int, error: float = 0.001) -> tuple:
    centroid_points = initialize_clusters(points, n_of_centroids)
    last_objective = 10000


    for n in range(n_of_oterations):
        distances = compute_distances(points, centroid_points)
        cluster_belongs = np.argmin(distances, axis=0)
        
        objective = calculate_objective(cluster_belongs, distances)
        
        if abs(last_objective - objective) < error:
            break
      
        last_objective = objective

        centroid_points = calculate_new_centroids(points, cluster_belongs, n_of_centroids)

    return centroid_points, last_objective



centroids, _ = fit(points, 3, n_of_iterations)


plt.figure()
plt.scatter(points[:,0],points[:,1])
plt.scatter(centroids[:,0].T,centroids[:,1].T)

################@@

class KmeansAlgo:
    def __init__(self, n_of_centroids: int):
        self.n_of_centroids = n_of_centroids
    
    def initialize_clusters(self, points: np.array) -> np.array:
        indexes = np.arange(points.shape[0])
        np.random.shuffle(indexes)                    
        required_indexes = indexes[:self.n_of_centroids]                         
        return points[required_indexes]
    
    def calculate_metric(self, points: np.array, centroid: np.array) -> np.array:
        return np.square(norm(points-centroid, axis=1))
    
    def compute_distances(self, points: np.array, centroids_points: np.array) -> np.array:
        return np.asarray([calculate_metric(points, centroid) for centroid in centroids_points])
    
    def calculate_objective(self, cluster_belongs: np.array, distances: np.array) -> np.array:
        distances = distances.T
        selected_min =  distances[np.arange(len(distances)), cluster_belongs]
        return np.sum(selected_min)
    
    def assign_clusters(self, distances: np.array) -> np.array:
        return np.argmin(distances, axis = 1)
    
    def calculate_new_centroids(self, points: np.array, clusters_belongs: np.array) -> np.array:
        new_clusters = []
        for cluster_id in range(self.n_of_centroids):
            j = np.where(clusters_belongs == cluster_id)
            points_sel = points[j]
            new_clusters.append(np.mean(points_sel, axis=0))

        return np.array(new_clusters)
    
    
    def fit(self, points: np.array, n_of_oterations: int, error: float = 0.001) -> tuple:
        centroid_points = initialize_clusters(points, self.n_of_centroids)
        last_objective = 10000


        for n in range(n_of_oterations):
            distances = self.compute_distances(points, centroid_points)
            cluster_belongs = np.argmin(distances, axis=0)

            objective = self.calculate_objective(cluster_belongs, distances)

            if abs(last_objective - objective) < error:
                break

            last_objective = objective

            centroid_points = self.calculate_new_centroids(points, cluster_belongs)

        return centroid_points, last_objective        

k = 3
n_of_points = 60
n_of_iterations = 200

#generate points
points = np.random.rand(n_of_points,2) * 5 
points = split_points(points, 3)
# In [214]:
kmeans = KmeansAlgo(k)
centroids, _ = kmeans.fit(points, n_of_iterations)

plt.figure()
plt.scatter(points[:,0], points[:,1])
plt.scatter(centroids[:,0], centroids[:,1])


###################################

k_all = range(2, 10)
all_objective = []

for n_of_cluster in k_all:
    _, objective = fit(points, n_of_cluster, n_of_iterations)
    all_objective.append(objective)
    
    

#########################################################################################################


#/Users/charles-arthurpacton/Desktop/Brno/Machine learning/Exercises/LAB02
from matplotlib.image import imread
from sklearn.cluster import KMeans
import numpy as np

loaded_image = imread('/Users/charles-arthurpacton/Desktop/Brno/Machine_learning/Exercises/LAB02/fish.jpg')

plt.imshow(loaded_image)
plt.show()


def compress_image(image: np.array, number_of_colours: int) -> np.array:
    original_shape = image.shape
    
    image_reshaped = np.reshape(image, (image.shape[0]*image.shape[1],image.shape[2]))
    kmeans = KMeans(n_clusters=number_of_colours, random_state=0).fit(image_reshaped)
    image_new = kmeans.predict(image_reshaped)
    

    cluster_labels = kmeans.labels_
    colors = kmeans.cluster_centers_.astype('uint8')
    
    new_image = colors[cluster_labels].reshape(original_shape)
    
    
    return new_image

img = compress_image(loaded_image, 30)

plt.figure()
plt.imshow(img)
plt.show()

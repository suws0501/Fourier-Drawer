import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
dic = defaultdict(set)

def pre_processing(img, thres1, thres2):
    imgPre = cv2.GaussianBlur(img,(3,3),1)
    imgPre = cv2.Canny(imgPre, thres1, thres2)
    kernel = np.ones((3, 3), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=3)
    imgPre = cv2.erode(imgPre,kernel,iterations=3)

    return imgPre

def create_point_list(edge):
    x = []
    y = []
    pts = []
    for i in range(len(edge)):
        for j in range(len(edge[0])):
            if edge[i][j] > 0:
                x.append(j)
                y.append(i)
                pts.append((j, i))

    return x, y, pts

def dist(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def create_adj_matrix(pts, threshold1, threshold2):
    adj_matrix = defaultdict(set)
    for pt1 in pts:
        for pt2 in pts:
            if pt1 != pt2 and threshold1 <= dist(pt1, pt2) <= threshold2:
                adj_matrix[pt1].add(pt2)
                adj_matrix[pt2].add(pt1)

    return adj_matrix


def dfs(graph, start):
    visited = set()
    stack = [start]
    result = []

    while stack:
        current_node = stack.pop()
        if current_node not in visited:
            visited.add(current_node)

            # Add unvisited neighbors to the stack
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    stack.append(neighbor)
                    result.append(neighbor)

    return (result)


def reorder_fft_array(X):
    N = len(X)
    reordered_X = np.zeros(N, dtype=complex)
    index = np.zeros(N, dtype=int)
    # Iterate through frequencies in the desired order
    for k in range(N):
        # Calculate the index for the current frequency
        if k == 0:
            index[k] = 0
        elif k % 2 == 1:
            index[k] = (k + 1) // 2  # Positive frequency
        else:
            index[k] = -(k // 2)  # Negative frequency

        # Access the element at the calculated index
        reordered_X[k] = X[index[k]]

    return reordered_X, index


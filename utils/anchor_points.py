# -*- coding: utf-8 -*-
import numpy as np
import xml.etree.ElementTree as ET
import os
import csv
import glob

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


# This method return the area by multiplying width*height of all the boxes
def area(x):
    if len(x.shape) == 1:
        return x[0] * x[1]
    else:
        return x[:, 0] * x[:, 1]


def kmeans_iou(k, centroids, points, iter_count=0, iteration_cutoff=30, feature_size=13):

    best_clusters = []
    best_avg_iou = 0
    best_avg_iou_iteration = 0

    npoi = points.shape[0]

    # store the area of the boxes
    area_p = area(points)  # (npoi, 2) -> (npoi,)

    while True:
        cen2 = centroids.repeat(npoi, axis=0).reshape(k, npoi, 2)

        # computing diff between all the points and the centroids
        cdiff = points - cen2
        cidx = np.where(cdiff < 0)
        cen2[cidx] = points[cidx[1], cidx[2]]

        # computing w * h
        wh = cen2.prod(axis=2).T  # (k, npoi, 2) -> (npoi, k)
        # distance metric is iou
        dist = 1. - (wh / (area_p[:, np.newaxis] + area(centroids) - wh))  # -> (npoi, k)
        # to assign the datapoints to the clusters
        belongs_to_cluster = np.argmin(dist, axis=1)  # (npoi, k) -> (npoi,)
        clusters_niou = np.min(dist, axis=1)  # (npoi, k) -> (npoi,)
        # store all the points in respective clusters
        clusters = [points[belongs_to_cluster == i] for i in range(k)]
        avg_iou = np.mean(1. - clusters_niou)
        # storing the best cluster based on the best avg iou obtained
        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_clusters = clusters
            best_avg_iou_iteration = iter_count

        # print("\nIteration {}".format(iter_count))
        # print("Average iou to closest centroid = {}".format(avg_iou))
        # print("Sum of all distances (cost) = {}".format(np.sum(clusters_niou)))

        # calculating the new centroid, which is the mean of the clusters
        new_centroids = np.array([np.mean(c, axis=0) for c in clusters])
        isect = np.prod(np.min(np.asarray([centroids, new_centroids]), axis=0), axis=1)
        aa1 = np.prod(centroids, axis=1)
        aa2 = np.prod(new_centroids, axis=1)
        shifts = 1 - isect / (aa1 + aa2 - isect)

        # break condition when the number of shifts are zero or the iteration_count is exceeded
        if sum(shifts) == 0 or iter_count >= best_avg_iou_iteration + iteration_cutoff:
            break

        centroids = new_centroids
        iter_count += 1

    # Get anchor boxes from best clusters
    anchors = np.asarray([np.mean(cluster, axis=0) for cluster in best_clusters])
    anchors = anchors[anchors[:, 0].argsort()]
    print("k-means clustering anchor points (original coordinates) \
    \nFound at iteration {} with best average IoU: {} \
    \n{}".format(best_avg_iou_iteration, best_avg_iou, anchors*feature_size))

    return anchors

def load_udacity_dataset(csv_fname):
    data = []
    with open(csv_fname, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar ='|',)
        for row in spamreader:
            img_name = row[0]
            w = 1920
            h = 1200

            labels = row[1:]
            all = list()
            for i in range(0, len(labels),5):
                xmin = int(labels[i])
                ymin = int(labels[i+1])
                xmax = int(labels[i+2])
                ymax = int(labels[i+3])
                b = (xmin,xmax,ymin,ymax)
                bb = convert_bbox((w,h),b)
                data.append(bb[2:])
    return np.array(data)

# This method return the bbox data: (width and heights)
def pascal_voc_dataset(path):
    data = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert_bbox((w, h), b)
            data.append(bb[2:])
    return np.array(data)


if __name__ == "__main__":

    # k-means picking the first k points as centroids
    img_size = 416
    k = 5

    # Pascal Voc dataset
    anno_filename = '/home/rohit/personal/civilmaps/darkflow_uda/darkflow/utils/VOCdevkit/VOC2007/Annotations'
    pascal_data = pascal_voc_dataset(anno_filename)
    centroids = pascal_data[np.random.choice(np.arange(len(pascal_data)), k, replace=False)]
    print (" PASCAL VOC: ")
    pascal_anchors = kmeans_iou(k, centroids, pascal_data, feature_size=img_size / 32)

    # Udacity dataset
    csv_fname = os.path.join('/home/rohit/personal/civilmaps/udacity_dataset/udacity.csv')
    udacity_data = load_udacity_dataset(csv_fname)
    centroids = udacity_data[np.random.choice(np.arange(len(udacity_data)),k,replace=False)]
    print (" Udacity Dataset: ")
    udacity_anchors = kmeans_iou(k, centroids, udacity_data, feature_size = img_size/32)

print ('done')

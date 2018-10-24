import os
import pickle
import ntpath
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import imp


def load_fruit(pickle_path):
    fruits = pickle.load(open(pickle_path, 'rb'))
    category = ntpath.basename(pickle_path).split('.')[0]
    return category, fruits


def analyze_fruit(pickle_path):
    name, samples = load_fruit(pickle_path)
    mean = get_mean(samples)
    mOK = 0
    fOK = 0
    size = len(samples) // 4
    samples = samples[:size]
    test = 0
    print("Analyzing class {}. {} samples".format(name, size))
    for key, fruit1 in enumerate(samples):
        for fruit2 in samples[key + 1:]:
            dist = np.linalg.norm(fruit1 - fruit2)
            if dist < 10:
                fOK += 1
            test += 1

    for fruit in samples:
        dist = np.linalg.norm(mean - fruit)
        if dist < 25:
            mOK += 1

    result1 = (fOK * 100) / test
    result2 = (mOK * 100) / size
    print("Done. Similar are {}%, {}% close to mean".format(result1, result2))
    return name, result1, result2


def get_mean(fruits):
    cnt = len(fruits)
    avg = 0
    for im in fruits:
        avg += im
    return avg / cnt


def get_results(set):
    ffStat = {}
    mfStat = {}

    for dir in os.listdir("data/{}".format(set)):
        if os.path.isdir("data/{}/{}".format(set, dir)):
            continue
        name, result1, result2 = analyze_fruit("data/{}/{}".format(set, dir))
        ffStat[name] = result1
        mfStat[name] = result2
    plt.bar(range(len(ffStat)), list(ffStat.values()))
    plt.xticks(range(len(ffStat)), list(ffStat.keys()), rotation='vertical')
    plt.show()

    plt.bar(range(len(mfStat)), list(mfStat.values()))
    plt.xticks(range(len(mfStat)), list(mfStat.keys()), rotation='vertical')
    plt.show()


def find_nearest(cluster_center, data):
    diff = 100
    index = 0
    for key, fruit in enumerate(data):
        dist = np.linalg.norm(cluster_center - fruit)
        if dist < diff:
            diff = dist
            index = key
    return index


def k_means():
    all_data = pickle.load(open('data/all_data.pickle', 'rb'))
    test_data = all_data['test_data']
    leght = len(test_data) // 8
    test_data = test_data[:leght]
    test_labels = all_data['test_labels']
    test_labels = test_labels[:leght]
    size = len(test_data[0])
    in_data = np.reshape(test_data, (len(test_data), size * size * 3))
    in_data = np.float32(in_data)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(in_data)
    for center in kmeans.cluster_centers_:
        r_center = np.reshape(center, (100, 100, 3))
        index = find_nearest(r_center, test_data)
        images = []
        images.append(r_center)
        images.append(test_data[index])
        stack = np.hstack(images)
        cv2.imshow(test_labels[index], stack)
        cv2.waitKey()

def db_scan():
    all_data = pickle.load(open('data/all_data.pickle', 'rb'))
    test_data = all_data['test_data']
    leght = len(test_data) // 8
    test_data = test_data[:leght]
    test_labels = all_data['test_labels']
    test_labels = test_labels[:leght]
    size = len(test_data[0])
    in_data = np.reshape(test_data, (len(test_data), size * size * 3))
    in_data = np.float32(in_data)

    db = DBSCAN(eps=25, min_samples=2).fit(in_data)

    n_cluster = len(np.unique(db.labels_)) - 1
    for i in range(0, n_cluster):
        clust_i = np.where(i == db.labels_)[0]
        cluster = np.take(test_data, clust_i, axis=0)
        avg = get_mean(cluster)
        index = find_nearest(avg, test_data)
        images = []
        images.append(avg)
        images.append(test_data[index])
        stack = np.hstack(images)
        cv2.imshow(test_labels[index], stack)
        cv2.waitKey()







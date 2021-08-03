import os

import cv2
import numpy as np
import pandas as pd
import tqdm

IMG_TYPE = ['.jpg', '.jpeg', '.png', '.tiff']


def stats_connected_elements(mask):
    connectivity = 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity,
                                                                            cv2.CV_32S)
    mean_area = np.mean(stats[1:, cv2.CC_STAT_AREA])
    return {'Number': num_labels - 1, 'Average area': mean_area}


def get_stats_from_folder_masks(root_folder, n_classes, labels=None):
    files = os.listdir(root_folder)
    files = [f for f in files if any([ext in f for ext in IMG_TYPE])]
    if labels is None:
        labels = np.arange(n_classes)
    labels_stats = ['Number', 'Average area']
    col = pd.MultiIndex.from_product([labels, labels_stats])
    df = pd.DataFrame(columns=col, index=range(len(files)))
    df['Name'] = files
    cols = df.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    i = 0
    for f in tqdm.tqdm(files):
        file = os.path.join(root_folder, f)
        mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        for c, l in zip(range(n_classes), labels):
            stats = stats_connected_elements(mask == c)
            for k in stats:
                df[(l, k)][i] = stats[k]
        i += 1
    return df

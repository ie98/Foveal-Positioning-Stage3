
import cv2
import numpy as np

import matplotlib.pyplot as plt

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"



def select_max_region2(mask):

    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    maxcount = np.max(labels)
    count = plt.hist(labels.ravel(), maxcount+1)[0]
    maxAreaNum = np.max(count[1:])
    indexNum = np.where(count == maxAreaNum)
    newMask = np.zeros(shape=[mask.shape[0],mask.shape[1]])
    pointIndex = np.where(labels==indexNum)
    newMask[pointIndex] = 1


    return newMask



def select_max_region(mask):
    new_mask = np.zeros(shape=[mask.shape[0]+4,mask.shape[1]+4],dtype=np.uint8)
    new_mask[2:-2,2:-2] = mask

    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(new_mask, connectivity=8)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels==max_idx+1, 1, 0)

    return max_region[2:-2,2:-2]



def FillHole(mask):
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.uint8)
    _,contours,hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out


def thresholdSegment(patch):



    R = 109
    max = np.max(patch)
    min = np.min(patch)
    patch_256 = -1*((((patch-min)/(max - min))*256).astype(np.int)-256)

    count = plt.hist(patch_256.ravel(),256)[0]

    threshold_count = int(((R+1)**2)*0.1)
    curr_count = 0
    threshold_pix = None
    for c in range(0,len(count)):
        if curr_count >= threshold_count:
            threshold_pix = c
            break
        curr_count = curr_count + count[c]


    index = np.where(patch_256<=threshold_pix)

    # 计算最大联通图
    mask = np.zeros((patch.shape[0],patch.shape[1]))
    mask[index[0],index[1]] = 1

    max_mask = select_max_region(mask.astype(np.uint8))
    max_mask_index = np.where(max_mask == 1)




    # 空隙填补
    max_mask = FillHole(max_mask)
    max_mask[max_mask>0] = 1
    mask_index = np.where(max_mask == 1)
    pre_x_avg = np.mean(mask_index[1])
    pre_y_avg = np.mean(mask_index[0])


    return pre_x_avg,pre_y_avg


















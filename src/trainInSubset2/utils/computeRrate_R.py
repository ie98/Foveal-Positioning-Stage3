import math
import os
import numpy as np
import pandas as pd

class computerRrate_test:
    def __init__(self):
        self.subSet_Details = np.array(pd.read_csv('./subset1_labels/subSet_1_Detail.csv'))

    def computer(self,preRes,saveFilePath,epoch,valImgNames):

        # preRes = np.array(pd.read_csv('./val_detail_248.csv'))
        resDict = {}
        R_rate_list = []
        up_x_list = []
        up_y_list = []
        real_dist_list = []
        count_R8 = 0
        count_R4 = 0
        count_R2 = 0
        count_R = 0
        count_RR = 0
        for i in range(len(preRes)):
            if valImgNames[i] != self.subSet_Details[i, 1]:
                print('image name error !!!')
            label_x = self.subSet_Details[i, 5]
            label_y = self.subSet_Details[i, 6]
            img_x = self.subSet_Details[i, 2]
            img_y = self.subSet_Details[i, 3]
            R = self.subSet_Details[i, 4]
            pre_x = preRes[i, 0]
            pre_y = preRes[i, 1]
            up_x = pre_x * (img_x / 2304)
            up_y = pre_y * (img_y / 1536)
            up_x_list.append(up_x)
            up_y_list.append(up_y)

            real_dist = math.sqrt((up_x - label_x) ** 2 + (up_y - label_y) ** 2)

            real_dist_list.append(real_dist)

            r_rate = real_dist / R
            if r_rate <= 0.125:
                count_R8 = count_R8 + 1
            if r_rate <= 0.25:
                count_R4 = count_R4 + 1
            if r_rate <= 0.5:
                count_R2 = count_R2 + 1
            if r_rate <= 1:
                count_R = count_R + 1
            if r_rate <= 2:
                count_RR = count_RR + 1
            R_rate_list.append(r_rate)

        allDetails = {
            'name': self.subSet_Details[:, 1],
            'W': self.subSet_Details[:, 2],
            'H': self.subSet_Details[:, 3],
            'label_x':self.subSet_Details[:, 5],
            'label_y':self.subSet_Details[:, 6],
            'pre_x': up_x_list,
            'pre_y': up_y_list,
            'dist': real_dist_list,
            'R': self.subSet_Details[:, 4],
            'R_rate': R_rate_list
        }
        if os.path.exists(saveFilePath) is False:
            os.makedirs(saveFilePath)
        dataframe = pd.DataFrame(allDetails)

        dataframe.to_csv('{}/{}_R_rate.csv'.format(saveFilePath,epoch), sep=',')

        return count_R8,count_R4,count_R2,count_R,count_RR,np.mean(np.array(real_dist_list))

import numpy as np
import torch
import dataProcess.testDataProcess as DL
# from network.mainNet.mainNet import mainNet
from Unet.UNetVersion.models.UNet_3Plus import UNet_3Plus
import torch.nn as nn
import torch.nn.functional as F
import time
import cv2
from torchvision import transforms
import pandas as pd
import os
import math
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
from getDetail.computeRrate import computerRrate_test
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

saveFileName = 'test in subset2 Train in Unet3+ & crossVal TVT & epochs 500 & lr 0.001 & lrf 0.001 & segment & patchSize R & maskSize circleR 25 & randomDeviation 0.25R & batchsize 16'
class trainer():

    def __init__(self,arg_file=None):
        pass


    def eucliDist(self,pre,label):
        temp = np.square( pre - label )
        return np.sqrt( temp[:,0] + temp[:,1] )


    # def getTrainData(self):
    #     can_use_imgs_dict, label , image_names = DL.loadLabel()
    #     imgs, new_labels = DL.loadImgs(can_use_imgs_dict, label)
    #     train_loader , val_loader , val_image_names = DL.dataload(imgs, new_labels,image_names)
    #     return train_loader, val_loader , val_image_names


    def getTemplate(self,templatePath):
        input_transform = transforms.Compose([
            # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
            transforms.ToPILImage(),
            transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
        ])
        template = cv2.imread(templatePath)  # 查询图像
        template = template.astype(np.uint8)
        template = input_transform(template).unsqueeze(0)
        return template

    def select_max_region(self,mask):

        nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        background = 0
        for row in range(stats.shape[0]):
            if stats[row, :][0] == 0 and stats[row, :][1] == 0:
                background = row
        stats_no_bg = np.delete(stats, background, axis=0)
        max_idx = stats_no_bg[:, 4].argmax()
        max_region = np.where(labels == max_idx + 1, 1, 0)

        return max_region

    def FillHole(self,mask):
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)
        _, contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        len_contour = len(contours)
        contour_list = []
        for i in range(len_contour):
            drawing = np.zeros_like(mask, np.uint8)  # create a black image
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)

        out = sum(contour_list)
        return out


    def train(self,device):
        # 使用tensorBoard
        print('Strat Tensorboard with "tensorboard -- logdir=runs",view at http://locahost:6006/')
        tb_writer = SummaryWriter()
        device = device
        model = UNet_3Plus(in_channels=3,n_classes=2).cuda(device)
        Dict = torch.load('/home/stu013/mapping/step_3/src/modelParam/MESSIDOR/Train in Unet3+ & crossVal TVT & epochs 500 & lr 0.001 & lrf 0.001 & segment & patchSize R & maskSize circleR 25 & randomDeviation 0.25R & batchsize 16/modelParam_578_6.107135389473199.pth',map_location={'cuda:1':'cuda:0'})
        paramDict = Dict['modelParam']
        model.load_state_dict(paramDict)
        # model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1],output_device=0)
        batchSize = 1


        testDataLoader , testImgName = DL.getDataLoader(batchSize)


        # 保存每个iteration的loss和accuracy，以便后续画图
        plt_train_loss = []
        plt_val_loss = []
        # plt_train_acc = []
        # plt_val_acc = []

        # 用测试集训练模型model(),用验证集作为测试集来验证

        best_dist = 99999999
        best_loss = 999999
        best_model = {}

        R = 109
        best_r8 = 0
        best_valLoss = 99999
        R_8 = 0
        R_4 = 0
        R_2 = 0
        R_1 = 0
        dist = 0.
        label_site = None
        pred_site = None
        curr_dist = None
        computerRrate = computerRrate_test()
        real_r8_count = 0


        # 验证集val
        model.eval()
        with torch.no_grad():

            for i, data in tqdm(enumerate(testDataLoader)):
                val_pred = model(data[0].cuda(device))


                val_pred = F.softmax(val_pred,dim=1)
                pred = val_pred.detach().cpu().numpy()


                # pred[pred>=0.5] = 1
                # pred[pred<0.5] = 0
                pred_point = np.ndarray(shape=[pred.shape[0],2])
                for p in range(pred.shape[0]):
                    cur_pred = pred[p,:]
                    bg_pre = cur_pred[0, :]
                    obj_pre = cur_pred[1, :]
                    finall_pre = np.zeros(shape=[pred.shape[2], pred.shape[3]])
                    index_obj = np.where(obj_pre > bg_pre)
                    finall_pre[index_obj] = 1

                    #finall_pre = self.select_max_region(finall_pre.astype(np.uint8))
                    # 空隙填补
                    #if len(index_obj[0]) is not 0:
                    #    finall_pre = self.FillHole(finall_pre)
                    #    finall_pre[finall_pre != 0] = 1
                    index = np.where(finall_pre == 1)
                    if len(index[0]) == 0 :
                        pre_x = data[2][p,0,0]
                        pre_y = data[2][p,0,1]
                    else:
                        pre_x = np.mean(index[1])+data[2][p,0,0]
                        pre_y = np.mean(index[0])+data[2][p,0,1]
                    pred_point[p,0] = pre_x
                    pred_point[p,1] = pre_y

                # 模型成绩测试
                dist_list = self.eucliDist(pred_point,data[1].detach().cpu().numpy()[:,0,:])
                R_rate = dist_list/R
                R_8_list = R_rate[R_rate<=0.125]
                R_8 = R_8 + len(R_8_list)
                R_4_list = R_rate[R_rate <= 0.25]
                R_4 = R_4 + len(R_4_list)
                R_2_list = R_rate[R_rate <= 0.5]
                R_2 = R_2 + len(R_2_list)
                R_1_list = R_rate[R_rate <= 1]
                R_1 = R_1 + len(R_1_list)
                if curr_dist is None:
                    curr_dist = dist_list
                else:
                    curr_dist = np.concatenate((curr_dist,dist_list),axis=0)

                if label_site is None:
                    label_site = data[2].detach().cpu().numpy()[:,0,:]
                else:
                    label_site = np.concatenate((label_site,data[2].detach().cpu().numpy()[:,0,:]),axis=0)

                if pred_site is None:
                    pred_site = pred_point
                else:
                    pred_site = np.concatenate((pred_site,pred_point),axis=0)



                dist = dist + np.sum(dist_list)

                # s1, s2 = self.estimate(val_pred.cpu().clone().detach().numpy(), data[2].cpu().detach())
                # score1 = score1 + s1
                # # s2 = self.estimate(val_pred, data[2].cpu().detach())
                # score2 = score2 + s2

            # val_detail = np.concatenate(val_image_names, label_site, pred_site, curr_dist)

            real_r8_count = computerRrate.computer(pred_site,'./outputFile/MESSIDOR_test_subset2/{}'.format(saveFileName),293,testImgName.reshape(-1))

            dist = dist / 568 # 验证图片有 568 张
            R_8_per = R_8 / 568
            R_4_per = R_4 / 568
            R_2_per = R_2 / 568
            R_1_per = R_1 / 568

            print('avg_dist = {}'.format(dist))
            logging.info('avg_dist = {}'.format(dist))

            print('R_8_per = {},R_4_per = {},R_2_per = {},R_1_per = {}'.format(R_8_per,R_4_per,R_2_per,R_1_per))
            logging.info('R_8_per = {},R_4_per = {},R_2_per = {},R_1_per = {}'.format(R_8_per,R_4_per,R_2_per,R_1_per))
            print('real_r8_per = {}'.format(real_r8_count/568))
            # 保存用于画图
            # plt_train_acc.append(train_acc / 1723)




            dataframe = pd.DataFrame(
                {'image_name': testImgName.reshape(-1), 'label_site_x': label_site[:, 0].reshape(-1),
                 'label_site_y': label_site[:, 1].reshape(-1),
                 'pred_site_x': pred_site[:, 0].reshape(-1),
                 'pred_site_y': pred_site[:, 1].reshape(-1),
                 'curr_dist': curr_dist.reshape(-1)})
            if os.path.exists('./outputFile/MESSIDOR_test_subset2/{}'.format(saveFileName)) is False:
                os.makedirs('./outputFile/MESSIDOR_test_subset2/{}'.format(saveFileName))
            dataframe.to_csv('./outputFile/MESSIDOR_test_subset2/{}/test_detail.csv'.format(saveFileName), sep=',')






import pynvml


# 字节数转GB
def bytes_to_gb(sizes):
    sizes = round(sizes / (1024 ** 3), 2)
    return f"{sizes} GB"


def gpu(num):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(num)  # 显卡句柄（只有一张显卡）
    gpu_name = pynvml.nvmlDeviceGetName(handle)  # 显卡名称
    gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 显卡内存信息
    data = dict(
        gpu_name=gpu_name.decode("utf-8"),
        gpu_memory_total=bytes_to_gb(gpu_memory.total),
        gpu_memory_used=bytes_to_gb(gpu_memory.used),
        gpu_memory_free=bytes_to_gb(gpu_memory.free),
    )
    return data


if __name__ == '__main__':

    device = 0
    var = 1
    i = 0
    # while var == 1:
    #     time.sleep(1)
    #     gpu_0 = gpu(0)
    #     gpu_1 = gpu(1)
    #
    #     free = []
    #     free.append(float(gpu_0['gpu_memory_free'].split(' ')[0]))
    #     free.append(float(gpu_1['gpu_memory_free'].split(' ')[0]))
    #     if free[0] > 6:
    #         device = 0
    #         break
    #     if free[1] > 6:
    #         device = 1
    #         break
    #
    #     if i%60==0:
    #         print(i)
    #     i = i+1

    trianer = trainer()
    trianer.train(device)

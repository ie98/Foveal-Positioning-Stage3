import numpy as np
import torch
import dataProcess.crossVal_circleMask as DL
# from network.mainNet.mainNet import mainNet
from unet_fix.unet_model import UNet
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
from vat import VATLoss
from getDetail.computeRrate import computerRrate_val
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

saveFileName = 'train in unetfix pretrained & no freezeWeight & subset1 & crossVal TVT & epochs 200 & lr 0.00001 & lrf 0.001 & segment & patchSize R & maskSize circleR 21 & randomDeviation 0.35R & batchsize 16'
class trainer():

    def __init__(self,arg_file=None):
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
        if os.path.exists('./outputFile/MESSIDOR_seg/{}'.format(saveFileName)) is False:
            os.makedirs('./outputFile/MESSIDOR_seg/{}'.format(saveFileName))
        if os.path.exists('./outputFile/MESSIDOR_seg/{}/my.log'.format(saveFileName)) is False:
            file = open('./outputFile/MESSIDOR_seg/{}/my.log'.format(saveFileName), 'w')
            file.close()
        logging.basicConfig(filename='./outputFile/MESSIDOR_seg/{}/my.log'.format(saveFileName), level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)



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


    def loss_2(self,pred,labelPoint):
        distMap = np.zeros(shape=(pred.shape))
        pred = F.softmax(pred,dim=1)
        print('1231321')

        pass




    def train(self,device):
        # 使用tensorBoard
        print('Strat Tensorboard with "tensorboard -- logdir=runs",view at http://locahost:6006/')
        tb_writer = SummaryWriter()
        device = device
        model = UNet(n_channels=3,n_classes=2)
        Dict = torch.load(
            '/home/stu013/mapping/step_3/src/modelParam/MESSIDOR/train in unetfix & subset1 & crossVal TVT & epochs 200 & lr 0.00001 & lrf 0.001 & segment & patchSize R & maskSize circleR 21 & randomDeviation 0.25R & batchsize 16/modelParam_48_6.46_7.0.pth')
        paramDict = Dict['modelParam']
        # model_static_dict = model.state_dict()
        # model_pretrained_dict = {k:v for k,v in paramDict.items() if k in model_static_dict}
        # model_static_dict.update(model_pretrained_dict)
        # model.load_state_dict(model_static_dict)
        model.load_state_dict(paramDict)
        # for name, para in model.named_parameters():
        #     # 除最后的全连接层外，其他权重全部冻结
        #     if "fc" not in name and 'down_add' not in name:
        #         para.requires_grad_(False)
        model = model.cuda(device)
        # model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1],output_device=0)
        pg = [p for p in model.parameters() if p.requires_grad]
        # model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1],output_device=0)
        batchSize = 16


        train_dataLoader, val_dateLoader,_ , val_image_names,_ = DL.getDataLoader(batchSize)
        # np.savetxt('vn.txt',val_image_names)
        num_epoch = 500
        start_epoch = 0
        # addEpochs = 0
        loss = nn.CrossEntropyLoss()
        loss_2 = nn.MSELoss()
        learning_rate = 0.00001
        # learning_rate = 0.001
        lrf = 0.0001
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=0.005)
        optimizer = torch.optim.Adam(pg, lr=learning_rate, weight_decay=0.005)
        # Scheduler 学习率下降曲线
        lf = lambda x: ((1 + math.cos(x * math.pi / 50)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        logging.info('batchsize is {}'.format(batchSize))
        logging.info('lr is {}'.format(learning_rate))
        logging.info('epochs is {}'.format(num_epoch))

        # 保存每个iteration的loss和accuracy，以便后续画图
        plt_train_loss = []
        plt_val_loss = []
        # plt_train_acc = []
        # plt_val_acc = []

        # 用测试集训练模型model(),用验证集作为测试集来验证
        curr_lr = learning_rate
        best_dist = 99999999
        best_loss = 999999
        best_model = {}

        R = 109
        best_r8 = 0
        best_valLoss = 99999

        for epoch in range(start_epoch, num_epoch):
            # for epoch in range(num_epoch,num_epoch+addEpochs):
            epoch_start_time = time.time()
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0
            dist_1 = 0.
            label_site = None
            pred_site_1 = None
            curr_dist_1 = None
            val_detail = None
            R_8_1 = 0
            dist_2 = 0.

            pred_site_2 = None
            curr_dist_2 = None

            R_8_2 = 0
            model.train()  # 确保 model 是在 训练 model (开启 Dropout 等...)
            # if epoch % 2 == 0:
            #   train_LD = train_dataLoader
            #    val_LD = val_dateLoader
            # else:
            #    train_LD = val_dateLoader
            #    val_LD = train_dataLoader
            train_LD = train_dataLoader
            val_LD = val_dateLoader
            for i, data in tqdm(enumerate(train_LD)):
                # print(data[0].shape)
                # x.append(data[0].cuda())
                # x.append(data[1].cuda())
                optimizer.zero_grad()  # 用 optimizer 将模型参数的梯度 gradient 归零

                main_Pred, point = model(data[0].cuda(device))  # 利用 model 得到预测的概率分布，这边实际上是调用模型的 forward 函数
                batch_loss = 0.2*loss(main_Pred.cuda(device), data[1].cuda(device)) + 0.8*loss_2(point.cuda(device),
                                                                                         data[4].cuda(device))
                # add_loss = self.loss_2(main_Pred.cuda(device),data[4])

                batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
                optimizer.step()  # 以 optimizer 用 gradient 更新参数

                # train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                train_loss += batch_loss.item()

            scheduler.step()

            # 验证集val
            model.eval()
            with torch.no_grad():

                for i, data in tqdm(enumerate(val_LD)):
                    val_pred, point = model(data[0].cuda(device))

                    # batch_loss = sum([(x - y) ** 2 for x, y in zip(data[1].cuda(), val_pred)]) / len(train_pred)
                    batch_loss = 0.2*loss(val_pred.cuda(device), data[1].cuda(device)) + 0.8*loss_2(point.cuda(device),
                                                                                            data[4].cuda(device))
                    # val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())

                    val_loss += batch_loss.item()
                    val_pred = F.softmax(val_pred, dim=1)
                    pred = val_pred.detach().cpu().numpy()
                    pred_point = point.detach().cpu().numpy()

                    # pred[pred>=0.5] = 1
                    # pred[pred<0.5] = 0
                    pred_point_1 = np.ndarray(shape=[pred.shape[0], 2])

                    pred_point_2 = np.ndarray(shape=[pred.shape[0], 2])
                    for p in range(pred.shape[0]):
                        cur_pred = pred[p, :]
                        bg_pre = cur_pred[0, :]
                        obj_pre = cur_pred[1, :]
                        finall_pre = np.zeros(shape=[pred.shape[2], pred.shape[3]])
                        index_obj = np.where(obj_pre > bg_pre)
                        finall_pre[index_obj] = 1
                        index = np.where(finall_pre == 1)
                        if len(index[0]) == 0:
                            pre_x = data[3][p, 0, 0]
                            pre_y = data[3][p, 0, 1]
                        else:
                            pre_x = np.mean(index[1]) + data[3][p, 0, 0]
                            pre_y = np.mean(index[0]) + data[3][p, 0, 1]
                        pred_point_1[p, 0] = pre_x
                        pred_point_1[p, 1] = pre_y
                        pred_point_2[p, 0] = pred_point[p, 0] + data[3][p, 0, 0]
                        pred_point_2[p, 1] = pred_point[p, 1] + data[3][p, 0, 1]

                    # 模型成绩测试
                    dist_list_1 = self.eucliDist(pred_point_1, data[2].detach().cpu().numpy()[:, 0, :])
                    dist_list_2 = self.eucliDist(pred_point_2, data[2].detach().cpu().numpy()[:, 0, :])

                    # 对于 分割 的定位结果
                    R8_dist_1 = dist_list_1 / R
                    R_8_list_1 = R8_dist_1[R8_dist_1 <= 0.125]
                    R_8_1 = R_8_1 + len(R_8_list_1)
                    if curr_dist_1 is None:
                        curr_dist_1 = dist_list_1
                    else:
                        curr_dist_1 = np.concatenate((curr_dist_1, dist_list_1), axis=0)

                    # 对于 模型直接定位结果
                    R8_dist_2 = dist_list_2 / R
                    R_8_list_2 = R8_dist_2[R8_dist_2 <= 0.125]
                    R_8_2 = R_8_2 + len(R_8_list_2)
                    if curr_dist_2 is None:
                        curr_dist_2 = dist_list_2
                    else:
                        curr_dist_2 = np.concatenate((curr_dist_2, dist_list_2), axis=0)

                    dist_1 = dist_1 + np.sum(dist_list_1)
                    dist_2 = dist_2 + np.sum(dist_list_2)

                    # s1, s2 = self.estimate(val_pred.cpu().clone().detach().numpy(), data[2].cpu().detach())
                    # score1 = score1 + s1
                    # # s2 = self.estimate(val_pred, data[2].cpu().detach())
                    # score2 = score2 + s2

                # val_detail = np.concatenate(val_image_names, label_site, pred_site, curr_dist)

                dist_1 = dist_1 / 114  # 验证图片有 568 张
                R_8_per_1 = R_8_1 / 114
                print('avg_dist_1 = {}'.format(dist_1))
                logging.info('avg_dist_2 = {}'.format(dist_1))

                print('R_8_per_1 = {}'.format(R_8_per_1))
                logging.info('R_8_per_1 = {}'.format(R_8_per_1))

                dist_2 = dist_2 / 114  # 验证图片有 568 张
                R_8_per_2 = R_8_2 / 114
                print('avg_dist_2 = {}'.format(dist_2))
                logging.info('avg_dist_2 = {}'.format(dist_2))

                print('R_8_per_2 = {}'.format(R_8_per_2))
                logging.info('R_8_per_2 = {}'.format(R_8_per_2))
                # 保存用于画图
                # plt_train_acc.append(train_acc / 1723)

                plt_train_loss.append(train_loss / train_LD.__len__())
                # plt_val_acc.append(val_acc / 431)
                plt_val_loss.append(val_loss / val_LD.__len__())



                print('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f | Val  loss: %3.6f' % \
                      (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                       plt_train_loss[-1], plt_val_loss[-1]))

                logging.info('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f | Val  loss: %3.6f' % \
                             (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                              plt_train_loss[-1], plt_val_loss[-1]))

                # tags = ["train_loss","val_loss", "val_dist", "learning_rate"]
                # tb_writer.add_scalar(tags[0], plt_train_loss[-1], epoch)
                # tb_writer.add_scalar(tags[1],plt_val_loss[-1],epoch)
                # tb_writer.add_scalar(tags[2], dist, epoch)
                # tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

            if epoch >= 0:

                if R_8_per_2 > best_r8 or val_loss < best_valLoss or dist_2 < best_dist or num_epoch - epoch == 50 or num_epoch - epoch == 40 or num_epoch - epoch == 30 or num_epoch - epoch == 20 or num_epoch - epoch == 10 or num_epoch - epoch == 0:
                    if os.path.exists('./modelParam/MESSIDOR/{}'.format(saveFileName)) is False:
                        os.makedirs('./modelParam/MESSIDOR/{}'.format(saveFileName))

                    best_r8 = max(best_r8, R_8_per_2)
                    best_valLoss = min(best_valLoss, val_loss)
                    best_dist = min(best_dist, dist_2)
                    best_model['dist'] = dist_2
                    best_model['r8_per'] = R_8_per_2
                    best_model['modelParam'] = model.state_dict()
                    torch.save(best_model,
                               './modelParam/MESSIDOR/{}/modelParam_{}_{}.pth'.format(saveFileName, epoch, dist_2))


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

    device = 1
    var = 1
    i = 0
    while var == 1:
        time.sleep(1)
        gpu_0 = gpu(0)
        gpu_1 = gpu(1)

        free = []
        free.append(float(gpu_0['gpu_memory_free'].split(' ')[0]))
        free.append(float(gpu_1['gpu_memory_free'].split(' ')[0]))
        if free[0] > 3:
            device = 0
            break
        if free[1] > 3:
            device = 1
            break

        if i%60==0:
            print(i)
        i = i+1

    trianer = trainer()
    trianer.train(device)


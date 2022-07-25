
import os
import sys
sys.path.append('..')
import numpy as np
import torch
import dataProcess.crossVal_maskSize27 as DL
# from network.mainNet.mainNet import mainNet
from net.unet_model import UNet
import torch.nn as nn
import torch.nn.functional as F
import time
import cv2
from torchvision import transforms
import pandas as pd

import math
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
from vat import VATLoss
from getDetail.computeRrate import computerRrate_val
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

saveFileName = 'maskSize circleR 27  & randomDeviation 0.25R & batchsize 16'
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


    def train(self,device):
        # 使用tensorBoard
        print('Strat Tensorboard with "tensorboard -- logdir=runs",view at http://locahost:6006/')
        tb_writer = SummaryWriter()
        device = device
        model = UNet(n_channels=3,n_classes=2).cuda(device)
        # Dict = torch.load('/home/stu013/mapping/step_3/src/modelParam/MESSIDOR/crossVal TVT & epochs 200 & lr 0.0001 & lrf 0.01 & segment & patchSize R & maskSize circleR 25  & randomDeviation 0.25R & batchsize 16/modelParam_144_6.078578207226439.pth')
        # paramDict = Dict['modelParam']
        # model.load_state_dict(paramDict)
        # model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1],output_device=0)
        batchSize = 16

        train_dataLoader, val_dateLoader,_ , val_image_names,_ = DL.getDataLoader(batchSize)
        # np.savetxt('vn.txt',val_image_names)
        num_epoch = 300
        start_epoch = 0
        # addEpochs = 0
        loss = nn.CrossEntropyLoss()
        learning_rate = 0.0001
        # learning_rate = 0.001
        lrf = 0.0001
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)
        # Scheduler 学习率下降曲线
        lf = lambda x: ((1 + math.cos(x * math.pi / num_epoch)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        logging.info('batchsize is {}'.format(batchSize))
        logging.info('lr is {}'.format(learning_rate))
        logging.info('epochs is {}'.format(num_epoch))

        vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)




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
        avg_dist_list = []
        valLoss_list = []

        for epoch in range(start_epoch,num_epoch):
        # for epoch in range(num_epoch,num_epoch+addEpochs):
        #     if epoch % 100 == 0 and epoch != 0:
        #         maskSize = maskSize - 2
        #         train_dataLoader, val_dateLoader, _, val_image_names, _ = DL.getDataLoader(batchSize,maskSize)
        #

            epoch_start_time = time.time()
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0
            dist = 0.
            label_site = None
            pred_site = None
            curr_dist = None
            val_detail = None
            R_8 = 0
            model.train()  # 确保 model 是在 训练 model (开启 Dropout 等...)
            #if epoch % 2 == 0:
             #   train_LD = train_dataLoader
            #    val_LD = val_dateLoader
            #else:
            #    train_LD = val_dateLoader
            #    val_LD = train_dataLoader
            train_LD = train_dataLoader
            val_LD = val_dateLoader
            for i, data in tqdm(enumerate(train_LD)):

                # print(data[0].shape)
                # x.append(data[0].cuda())
                # x.append(data[1].cuda())
                optimizer.zero_grad()  # 用 optimizer 将模型参数的梯度 gradient 归零


                main_Pred = model(data[0].cuda(device))  # 利用 model 得到预测的概率分布，这边实际上是调用模型的 forward 函数
                batch_loss = loss(main_Pred.cuda(device), data[1].cuda(device))
                # if epoch > 30:
                #     lds = vat_loss(model, data[0].detach().clone().cuda(device))
                #     batch_loss = main_loss + 0.5 * lds # 计算 loss （注意 prediction 跟 label 必须同时在 CPU 或是 GPU 上）
                # else:
                #     batch_loss = main_loss

                batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
                optimizer.step()  # 以 optimizer 用 gradient 更新参数

                # train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                train_loss += batch_loss.item()

            scheduler.step()

            # 验证集val
            model.eval()
            with torch.no_grad():

                for i, data in tqdm(enumerate(val_LD)):
                    val_pred = model(data[0].cuda(device))

                    # batch_loss = sum([(x - y) ** 2 for x, y in zip(data[1].cuda(), val_pred)]) / len(train_pred)
                    batch_loss = loss(val_pred.cuda(device), data[1].cuda(device))
                    # val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())


                    val_loss += batch_loss.item()
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
                        index = np.where(finall_pre == 1)
                        if len(index[0]) == 0:
                            pre_x = data[3][p,0,0]
                            pre_y = data[3][p,0,1]
                        else:
                            pre_x = np.mean(index[1])+data[3][p,0,0]
                            pre_y = np.mean(index[0])+data[3][p,0,1]
                        pred_point[p,0] = pre_x
                        pred_point[p,1] = pre_y

                    # 模型成绩测试
                    dist_list = self.eucliDist(pred_point,data[2].detach().cpu().numpy()[:,0,:])
                    R8_dist = dist_list/R
                    R_8_list = R8_dist[R8_dist<=0.125]
                    R_8 = R_8 + len(R_8_list)
                    if curr_dist is None:
                        curr_dist = dist_list
                    else:
                        curr_dist = np.concatenate((curr_dist,dist_list),axis=0)

                    if label_site is None:
                        label_site = data[3].detach().cpu().numpy()[:,0,:]
                    else:
                        label_site = np.concatenate((label_site,data[3].detach().cpu().numpy()[:,0,:]),axis=0)

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

                dist = dist / 114 # 验证图片有 568 张
                avg_dist_list.append(dist)
                R_8_per = R_8 / 114
                print('avg_dist = {}'.format(dist))
                logging.info('avg_dist = {}'.format(dist))

                print('R_8_per = {}'.format(R_8_per))
                logging.info('R_8_per = {}'.format(R_8_per))
                # 保存用于画图
                # plt_train_acc.append(train_acc / 1723)



                plt_train_loss.append(train_loss / train_LD.__len__())
                # plt_val_acc.append(val_acc / 431)
                plt_val_loss.append(val_loss / val_LD.__len__())

                dataframe = pd.DataFrame(
                    {'image_name': val_image_names.reshape(-1), 'label_site_x': label_site[:, 0].reshape(-1),
                     'label_site_y': label_site[:, 1].reshape(-1),
                     'pred_site_x': pred_site[:, 0].reshape(-1),
                     'pred_site_y': pred_site[:, 1].reshape(-1),
                     'curr_dist': curr_dist.reshape(-1)})
                if os.path.exists('./outputFile/MESSIDOR_seg/{}'.format(saveFileName)) is False:
                    os.makedirs('./outputFile/MESSIDOR_seg/{}'.format(saveFileName))
                dataframe.to_csv('./outputFile/MESSIDOR_seg/{}/val_detail_{}.csv'.format(saveFileName,epoch), sep=',')
                if epoch == 0:
                    f = open('./outputFile/MESSIDOR_seg/{}/detail.txt'.format(saveFileName),'a+')
                    f.write('batchSize is {}\n\r'.format(batchSize))
                    f.write('lr is {}\n\r'.format(learning_rate))
                    f.write('epochs is {}\n\r'.format(num_epoch))
                    f.close()
                # 将结果 print 出來

                print('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f | Val  loss: %3.6f' % \
                      (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                       plt_train_loss[-1], plt_val_loss[-1]))

                logging.info('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f | Val  loss: %3.6f' % \
                      (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                       plt_train_loss[-1], plt_val_loss[-1]))

                tags = ["train_loss","val_loss", "val_dist", "learning_rate"]
                tb_writer.add_scalar(tags[0], plt_train_loss[-1], epoch)
                tb_writer.add_scalar(tags[1],plt_val_loss[-1],epoch)
                tb_writer.add_scalar(tags[2], dist, epoch)
                tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

            if epoch >= 50:


                if R_8_per > best_r8 or val_loss < best_valLoss or dist < best_dist or num_epoch - epoch == 50 or num_epoch - epoch == 40 or num_epoch - epoch == 30 or num_epoch - epoch == 20 or num_epoch - epoch == 10 or num_epoch - epoch == 0:
                    if os.path.exists('./modelParam/MESSIDOR/{}'.format(saveFileName)) is False:
                        os.makedirs('./modelParam/MESSIDOR/{}'.format(saveFileName))

                    best_r8 = max(best_r8,R_8_per)
                    best_valLoss = min(best_valLoss,val_loss)
                    best_dist = min(best_dist,dist)
                    best_model['dist'] = dist
                    best_model['r8_per'] = R_8_per
                    best_model['modelParam'] = model.state_dict()
                    torch.save(best_model,
                               './modelParam/MESSIDOR/{}/modelParam_{}_{}_{}.pth'.format(saveFileName, epoch, dist,R_8_per))

        dataframe = pd.DataFrame(
            {'avg_dist': avg_dist_list,
             'val_loss':plt_val_loss
             })
        if os.path.exists('./outputFile/MESSIDOR_seg/{}'.format(saveFileName)) is False:
            os.makedirs('./outputFile/MESSIDOR_seg/{}'.format(saveFileName))
        dataframe.to_csv('./outputFile/MESSIDOR_seg/{}/distAndValLoss.csv'.format(saveFileName), sep=',')


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
        if free[0] > 4:
            device = 0
            break
        if free[1] > 4:
            device = 1
            break

        if i%60==0:
            print(i)
        i = i+1

    trianer = trainer()
    trianer.train(device)


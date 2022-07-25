import cv2
import os
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
basePath = '../../dataset/MESSIDOR'

imgName = '20051019_38557_0100_PP.tif'

img = cv2.imread(os.path.join(basePath,'images',imgName),-1)

print(img)

def getImg(imgName):
    img = cv2.imread(os.path.join(basePath,'images', imgName), -1)
    return img


labelFileName = 'data.xls'



def getLabels(labelFilesName):
    labels = pd.read_excel(os.path.join(basePath, 'label', labelFileName))
    return labels




def loadImgs(img_name,label,step2Pre):
    R = 109
    patch_size = R+1
    # mask_size = 51
    y = 1536
    x = 2304
    new_labels = np.ndarray(shape=[1, 2])
    new_s2 = np.ndarray(shape=[1, 2])
    img = cv2.imread(os.path.join(basePath,'images', img_name))

    # 计算图像下采样后，标签的位置
    rate_x = x / img.shape[1]
    rate_y = y / img.shape[0]
    new_label_x =  int(label[0,0] * rate_x)
    new_label_y =  int(label[0,1] * rate_y)
    new_labels[0,0] = new_label_x
    new_labels[0,1] = new_label_y

    new_s2_x = int(step2Pre[0, 0] * rate_x)
    new_s2_y = int(step2Pre[0, 1] * rate_y)

    new_s2[0, 0] = new_s2_x
    new_s2[0, 1] = new_s2_y

    # 图像下采样
    img.astype(np.uint8)
    img = cv2.resize(img, dsize=(x, y), interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite('./{}.jpg'.format(img_name),img)

    # 测试时使用
    # img[new_s2_y,new_s2_x,0] = 255

    label_mask = np.zeros(shape=[patch_size,patch_size]).astype(np.int)

    # 随机生成 -0.4R 到 0.4R 之间的随机整数
    # random_x = random.randint(int(-0.25*R),int(0.25*R))
    # random_y = random.randint(int(-0.25*R),int(0.25*R))

    patch_center_x = new_s2_x
    patch_center_y = new_s2_y

    left_x = patch_center_x - int(R/2)
    left_y = patch_center_y - int(R/2)
    leftUp_point = np.ndarray(shape=[1, 2])
    leftUp_point[0,0] = left_x
    leftUp_point[0,1] = left_y

    # 截取输入样本图像
    img_patch = img[patch_center_y-int(R/2):patch_center_y+R-int(R/2)+1,patch_center_x-int(R/2):patch_center_x+R-int(R/2)+1,:]

    # img[[left_y,left_y+1,left_y+R+1,left_y+R+1+1],left_x:left_x+R+1,:] = (0,0,0)
    # img[left_y:left_y+R+1,[left_x,left_x+1,left_x+R+1,left_x+R+1+1],:] = (0,0,0)

    # 构建分割标注

    # label_mask[int(R/2)-int(mask_size/2):int(R/2)+mask_size-int(mask_size/2),int(R/2)-int(mask_size/2):int(R/2)+mask_size-int(mask_size/2)] = 1
    # label_mask = None
    # index = np.where(label_mask == 1)
    # img_patch[index[0],index[1],2] = 255

    # if os.path.exists('./data1/trainImg') is False:
    #     os.makedirs('./data1/trainImg')
    #
    # if os.path.exists('./data1/mask') is False:
    #     os.makedirs('./data1/mask')
    #
    # cv2.imwrite('./data1/trainImg/{}'.format(img_name),img)
    # cv2.imwrite('./data1/mask/{}'.format(img_name), img_patch)




    return img_patch,new_labels,leftUp_point,img_name

class TestDataSet(Dataset):

    def __init__(self, x_img_name, y_label,step2Pre, transforms=None):
        self.x_img_name = x_img_name
        self.y_label = y_label
        self.transforms = transforms
        self.step2Pre = step2Pre

    def __getitem__(self, item):
        X_img,new_labels,leftUp_point,img_name = loadImgs(self.x_img_name[item], self.y_label[item].reshape(1,2),self.step2Pre[item].reshape(1,2))
        if self.transforms is not None:
            X_img = self.transforms(X_img)
        return X_img,new_labels,leftUp_point,img_name

    def __len__(self):
        return len(self.x_img_name)




def getDataLoader(batchSize):
    labelFileName = 'data.xls'
    labelDetail = np.array(getLabels(labelFileName))
    imgNames = labelDetail[:,0]
    labels = labelDetail[:, [2, 3]]


    step2Pred = np.array(pd.read_csv('./end_use_file/in subset2/end test in subset2.csv'))
    # step2Pred = np.array(pd.read_csv('./end_use_GT/subset_2_stage2Pre.csv'))
    pred_point = step2Pred[:,[2,3]]


    # random_list = random.sample(range(0, len(imgNames)), len(imgNames))
    # np_list = np.array(random_list)
    # imgNames = imgNames[np_list]
    # labels = labels[np_list]
    # rate = 0.5
    # train_imgNames = imgNames[:int(len(imgNames) * rate)]
    # train_labels = labels[:int(len(labels) * rate), :].astype(np.float32)
    # val_imgName = imgNames[int(len(imgNames) * rate):]
    # val_labels = labels[int(len(labels) * rate):, :].astype(np.float32)

    sub_set_1 = np.array([i for i in range(1136) if i % 2 == 0])
    sub_set_2 = np.array([i for i in range(1136) if i % 2 == 1])
    # sub_1_train = sub_set_1[[i for i in range(568) if i % 5 != 0]]
    # sub_1_val = sub_set_1[[i for i in range(568) if i % 5 == 0]]


    test_imgNames = imgNames[sub_set_2]
    test_labels = labels[sub_set_2]

    # 图片处理
    input_transform = transforms.Compose([
        # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
        transforms.ToPILImage(),
        transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
        #transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
    ])




    test_set = TestDataSet(test_imgNames,test_labels,pred_point,input_transform)
    test_loader = DataLoader(test_set,batch_size=batchSize,shuffle=False,num_workers=2)
    return test_loader,test_imgNames



if __name__ == '__main__':
    test_loader,test_imgNames = getDataLoader(4)
    for i,data in enumerate(test_loader):
        print(data)
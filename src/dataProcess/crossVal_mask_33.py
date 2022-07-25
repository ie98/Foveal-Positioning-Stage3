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




def loadImgs(img_name,label):
    R = 109
    patch_size = R+1
    mask_size = 40
    y = 1536
    x = 2304
    new_labels = np.ndarray(shape=[1, 2])
    img = cv2.imread(os.path.join(basePath,'images', img_name))

    # 计算图像下采样后，标签的位置
    rate_x = x / img.shape[1]
    rate_y = y / img.shape[0]
    new_label_x =  int(label[0,0] * rate_x)
    new_label_y =  int(label[0,1] * rate_y)
    new_labels[0,0] = new_label_x
    new_labels[0,1] = new_label_y

    # 图像下采样
    img.astype(np.uint8)
    img = cv2.resize(img, dsize=(x, y), interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite('./{}.jpg'.format(img_name),img)

    # 测试时使用
    # img[new_label_y,new_label_x,0] = 255

    label_mask = np.zeros(shape=[patch_size,patch_size]).astype(np.int)

    # 随机生成 -0.25R 到 0.25R 之间的随机整数
    random_x = random.randint(int(-0.3*R),int(0.3*R))
    random_y = random.randint(int(-0.3*R),int(0.3*R))

    patch_center_x = new_label_x + random_x
    patch_center_y = new_label_y + random_y

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

    label_mask[int(R/2)-random_y-int(mask_size/2):int(R/2)-random_y+mask_size-int(mask_size/2),int(R/2)-random_x-int(mask_size/2):int(R/2)-random_x+mask_size-int(mask_size/2)] = 1
    # index = np.where(label_mask == 1)
    # img_patch[index[0],index[1],2] = 255

    # if os.path.exists('./data/trainImg') is False:
    #     os.makedirs('./data/trainImg')
    #
    # if os.path.exists('./data/mask') is False:
    #     os.makedirs('./data/mask')
    #
    # cv2.imwrite('./data/trainImg/{}'.format(img_name),img)
    # cv2.imwrite('./data/mask/{}'.format(img_name), img_patch)




    return img_patch,label_mask,new_labels,leftUp_point

class MyDataSet(Dataset):

    def __init__(self, x_img_name, y_label, transforms=None):
        self.x_img_name = x_img_name
        self.y_label = y_label
        self.transforms = transforms

    def __getitem__(self, item):
        X_img, Y_label,new_labels,leftUp_point = loadImgs(self.x_img_name[item], self.y_label[item].reshape(1,2))
        if self.transforms is not None:
            X_img = self.transforms(X_img)
        return X_img, Y_label.astype(np.int),new_labels,leftUp_point

    def __len__(self):
        return len(self.x_img_name)




def getDataLoader(batchSize):
    labelFileName = 'data.xls'
    labelDetail = np.array(getLabels(labelFileName))
    imgNames = labelDetail[:,0]
    labels = labelDetail[:, [2, 3]]
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
    sub_1_train = sub_set_1[[i for i in range(568) if i % 5 != 0]]
    sub_1_val = sub_set_1[[i for i in range(568) if i % 5 == 0]]
    #random_list = random.sample(range(0, len(sub_set_1)), len(sub_set_1))
    #np_list = np.array(random_list)
    train_imgNames = imgNames[sub_1_train]
    train_labels = labels[sub_1_train].astype(np.float32)
    #train_imgNames = train_imgNames[np_list]
    #train_labels = train_labels[np_list]
    val_imgName = imgNames[sub_1_val]
    val_labels = labels[sub_1_val].astype(np.float32)

    test_imgNames = imgNames[sub_set_2]
    test_labels = labels[sub_set_2]



    #subset_1 = np.array(pd.read_csv('./subSet/subSet_1.csv'))[:,1:]
    #subset_2 = np.array(pd.read_csv('./subSet/subSet_2.csv'))[:,1:]
    #random_list = random.sample(range(0, len(subset_1)), len(subset_1))
    #rand_arry = np.array(random_list)
    #subset_1 = subset_1[rand_arry]
    #train_imgNames = subset_1[:,0]
    #train_labels = subset_1[:,[1,2]].astype(np.float32)
    #val_imgName = subset_2[:,0]
    #val_labels = subset_2[:,[1,2]].astype(np.float32)
    # 图片处理
    input_transform = transforms.Compose([
        # transforms.CenterCrop([256, 256]),  # 将图像裁剪成指定大小
        transforms.ToPILImage(),
        transforms.ToTensor(),  # 图像转 Tensor ， 同时会将每个像素点除以 255 ， 取值为 0 - 1
        #transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 标准化 ，两个list分别是 均值 和 标准差
    ])

    train_set = MyDataSet(train_imgNames,train_labels,input_transform)
    val_set = MyDataSet(val_imgName,val_labels,input_transform)
    train_loader = DataLoader(train_set,batch_size=batchSize,shuffle=True,num_workers=2)
    val_loader = DataLoader(val_set,batch_size=batchSize,shuffle=False,num_workers=2)

    test_set = MyDataSet(test_imgNames,test_labels,input_transform)
    test_loader = DataLoader(test_set,batch_size=batchSize,shuffle=False,num_workers=2)
    return train_loader,val_loader,test_loader,val_imgName,test_imgNames



if __name__ == '__main__':
    train_loader,val_loader,test_loader,val_imgName,test_imgNames = getDataLoader(4)
    for i,data in enumerate(train_loader):
        print(data)
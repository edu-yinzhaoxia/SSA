import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import os




def loadData(dataID=1):
    # 读入数据
    if dataID == 1:
        data = sio.loadmat('./Data/PaviaU.mat')['paviaU']
        labels = sio.loadmat('./Data/PaviaU_gt.mat')['paviaU_gt']

    elif dataID == 2:
        data = sio.loadmat('./Data/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('./Data/Salinas_gt.mat')['salinas_gt']


    elif dataID == 3:
        data = sio.loadmat('./Data/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat('./Data/Indian_pines_gt.mat')['indian_pines_gt']

    [row, col, n_feature] = data.shape
    K = row * col
    data = data.reshape(K, n_feature)
    data = featureNormalize(data, 2)
    data = np.reshape(data, (row, col, n_feature))

    return data, labels

def featureNormalize(X, type):
    # type==1 x = (x-mean)/std(x)
    # type==2 x = (x-max(x))/(max(x)-min(x))
    if type == 1:
        mu = np.mean(X, 0)
        X_norm = X - mu
        sigma = np.std(X_norm, 0)
        X_norm = X_norm / sigma
        return X_norm
    elif type == 2:
        minX = np.min(X, 0)
        maxX = np.max(X, 0)
        X_norm = X - minX
        X_norm = X_norm / (maxX - minX)
        # MAX = np.max(X.ravel()).astype('float32')
        # MIN = np.min(X.ravel())
        # X_norm = (X - MIN) / (MAX - MIN) - 0.5
        return X_norm

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype='float32')
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype='float32')
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


BATCH_SIZE_TRAIN = 256


def create_data_loader():
    # 地物类别
    # class_num = 16
    # 读入数据
    X, y = loadData(dataID=3)
    # 用于测试样本的比例
    test_ratio = 0.8
    # 每个像素周围提取 patch 的尺寸
    patch_size = 19

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)


    print('\n... ... create data cubes ... ...')
    X, y_all = createImageCubes(X, y, windowSize=patch_size)
    print('Data cube X shape: ', X.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)


    # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 3, 1, 2)
    Xtest = Xtest.transpose(0, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader
    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=BATCH_SIZE_TRAIN,
                                              shuffle=False,
                                              num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                  batch_size=BATCH_SIZE_TRAIN,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  )

    return train_loader, test_loader, all_data_loader, y


class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Testing dataset"""


class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len

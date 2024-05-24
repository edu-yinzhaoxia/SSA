from operator import truediv
import argparse
from HyperTools import *
from Models import *
import GAN_models
from Gensample import *
import time

DataName = {1: 'PaviaU', 2: 'Salinas', 3: 'Indian_pines'}



def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    target_names1 = ['Asphalt', 'Meadows', 'Gravel', 'Trees'
        , 'Mettal sheets', 'Bare soil', 'Bitumen',
                    'Bricks', 'Shadows']
    target_names2 = ['Brocoli-green-weeds-1', 'Brocoli-green-weeds-2', 'Fallow', 'Fallow-rough-plow', 'Fallow-smooth ', 'Stubble', 'Celery'
        , 'Grapes-untrained', 'Soil-vinyard-develop', 'Corn-senesced-green-weeds', 'Lettuce-romaine-4wk', 'Lettuce-romaine-5wk'
        , 'Lettuce-romaine-6wk', 'Lettuce-romaine-7wk', 'Vinyard-untrained', 'Vinyard-vertical-trellis ']
    target_names3 = ['Alfalfa', 'Corn-notill', 'corn-mintill', 'Corn', 'Grass-pasture ', 'Grass-tress', 'Grass-pasture-mowed'
        , 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill', 'Soybean-clean'
        , 'Wheat', 'Woods', 'Buildings-Grass-Tress-Drives', 'Stone-Steel-Towers']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names3)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, all_data_loader, y_all = create_data_loader()
    # N_PCA, PATCH_SIZE, class_nums = 103, 13, 9
    # N_PCA, PATCH_SIZE, class_nums = 204, 15, 16
    N_PCA, PATCH_SIZE, class_nums = 200, 19, 16
    # Model = HybridSN(N_PCA, PATCH_SIZE, class_nums)
    # Model = CNN_2D(N_PCA, PATCH_SIZE, class_nums)
    Model = SACNet(N_PCA, PATCH_SIZE, class_nums)

    # model_path = 'models/hybrid/data1/13_hybrid1_epoch100_99.88019.pth'
    # model_path = 'models/hybrid/data2/15_hybrid2_99.95381.pth'
    # model_path = 'models/hybrid/data3/19_hybrid3_epoch200_98.26829.pth'

    # model_path = 'models/2D-CNN/data1/CNN2D-1_epoch100_97.36711.pth'
    # model_path = 'models/2D-CNN/data2/CNN2D-2_epoch100_96.70007.pth'
    # model_path = 'models/2D-CNN/data3/CNN2D-3_epoch500_92.14634.pth'

    # model_path = './models/SACNet/data1/13_SAC-1_epoch100_99.72824.pth'
    # model_path = './models/SACNet/data2/15_SAC2_epoch100_99.59126.pth'
    model_path = './models/SACNet/data3/19_SAC3_epoch1200_95.29268.pth'
    Model.load_state_dict(torch.load(model_path))

    Model = Model.cuda()
    Model.eval()

    y_pred_test, y_test = test(device, Model, test_loader)
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    
    print('Kappa accuracy (%)', kappa)
    print('Overall accuracy (%)', oa)
    print('Average accuracy (%)', aa)
    print('Each accuracy (%)', each_acc)

    # image_nc = 15
    image_nc = 30
    gen_input_nc = image_nc

    # pretrained_generator_path = './GAN_models/hybrid/data1/15_DA1_netG_13_99.88_80.pth'
    # pretrained_generator_path = './GAN_models/hybrid/data2/DA2_netG_99.953_80.pth'
    # pretrained_generator_path = './GAN_models/hybrid/data3/DA2_netG_98.268_80.pth'

    # pretrained_generator_path = './GAN_models/2D-CNN/data1/DA1_netG_97.367_50.pth'
    # pretrained_generator_path = './GAN_models/2D-CNN/data2/DA2_netG_96.700_50.pth'
    # pretrained_generator_path = './GAN_models/2D-CNN/data3/DA2_netG_92.14_50.pth'

    # pretrained_generator_path = './GAN_models/SACNet/data1/15_DA1_netG_13_99.728_80.pth'
    # pretrained_generator_path = './GAN_models/SACNet/data2/DA2_netG_99.59_60.pth'
    pretrained_generator_path = './GAN_models/SACNet/data3/DA2_netG_95.292_60.pth'
    pretrained_G = GAN_models.Generator(gen_input_nc, image_nc).cuda()
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()

    count = 0
    # 模型测试
    y_pred_test = 0
    y_test = 0
    L2 = 0
    L0 = 0
    # band_num = 15
    band_num = 30
    I = len(test_loader)
    # data1:
    # band = [48, 3, 36, 47, 10, 55, 71, 44, 43, 4, 50, 52, 69, 77, 22]
    # data2:
    # band = [70, 76, 21, 49, 17, 48, 86, 71, 53, 32, 56, 45, 69, 31, 94, 28, 25, 44, 79, 37, 24, 42, 40, 29, 85, 35, 98, 65, 95, 46]
    # data3:
    band = [41, 129, 91, 90, 115, 49, 168, 172, 132, 127, 107, 43, 171, 51, 81, 123, 68, 120, 6, 44, 32, 48, 88, 137, 9, 22, 108, 141, 3, 67]

    tic1 = time.perf_counter()
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        [bath, n_feature, row, col] = inputs.shape
        images = torch.zeros([bath, band_num, row, col]).type_as(inputs).to(device)
        for i in range(band_num):
            images[:, i, :, :] = inputs[:, band[i]-1, :, :]
        perturbation = pretrained_G(images)
        # perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_images = perturbation + images
        for i in range(band_num):
            inputs[:, band[i]-1, :, :] = adv_images[:, i, :, :]
        adv_img = inputs.to(device)
        adv_img = torch.clamp(adv_img, 0, 1)
        l2 = l2_distance(images, adv_images)
        L2 = L2 + l2
        l0 = l0_distance(images, adv_images)
        L0 = L0 + l0
        outputs = Model(adv_img)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    toc1 = time.perf_counter()
    GenPGD_Time = toc1 - tic1
    classification_adv, oa_adv, confusion_adv, each_acc_adv, aa_adv, kappa_adv = acc_reports(y_test, y_pred_test)
    classification_adv = str(classification_adv)

    print('ASR(%)', 100-oa_adv)
    print('Kappa accuracy (%)', kappa_adv)
    print('Overall accuracy (%)', oa_adv)
    print('Average accuracy (%)', aa_adv)
    print('Each accuracy (%)', each_acc_adv)
    print('l2:', L2/I)
    print('l0:', L0/I)
    print('GenPGD_Time:', GenPGD_Time)





import torch.nn as nn
import torch
import numpy as np
import GAN_models
import torch.nn.functional as F
import torchvision
import os

models_path = './GAN_models/hybrid/data1/'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc
        self.netG = GAN_models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = GAN_models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        # self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
        #                                     lr=0.001)
        # self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
        #                                     lr=0.001)
        self.optimizer_G = torch.optim.Adadelta(self.netG.parameters(),
                                            lr=1)
        self.optimizer_D = torch.optim.Adadelta(self.netDisc.parameters(),
                                            lr=1)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, X, x, labels):
        # optimize D
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -1, 1) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            band1 = [48, 3, 36, 47, 10, 55, 71, 44, 43, 4, 50, 52, 69, 77, 22]
            # band2 = [70, 76, 21, 49, 17, 48, 86, 71, 53, 32, 56, 45, 69, 31, 94, 28, 25, 44, 79, 37, 24, 42, 40, 29,
            #          85, 35, 98, 65, 95, 46]
            # band3 = [41, 129, 91, 90, 115, 49, 168, 172, 132, 127, 107, 43, 171, 51, 81, 123, 68, 120, 6, 44, 32, 48,
            #          88, 137, 9, 22, 108, 141, 3, 67]
            [bath, band_num, row, col] = adv_images.shape
            for i in range(band_num):
                X[:, band1[i]-1, :, :] = adv_images[:, i, :, :]
            X = X.cuda()
            logits_model = self.model(X)
            if len(logits_model.shape)==1:
                logits_model = torch.unsqueeze(logits_model,0)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            # loss_adv = F.mse_loss(logits_model, onehot_labels)
            # loss_adv = F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs, band_num):
        for epoch in range(1, epochs+1):
            if epoch == 40:
                self.optimizer_G = torch.optim.Adadelta(self.netG.parameters(),
                                                    lr=0.5)
                self.optimizer_D = torch.optim.Adadelta(self.netDisc.parameters(),
                                                    lr=0.5)
            # if epoch == 80:
            #     self.optimizer_G = torch.optim.Adamax(self.netG.parameters(),
            #                                         lr=0.00002)
            #     self.optimizer_D = torch.optim.Adamax(self.netDisc.parameters(),
            #                                         lr=0.00002)

            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            band1 = [48, 3, 36, 47, 10, 55, 71, 44, 43, 4, 50, 52, 69, 77, 22]
            # band2 = [70, 76, 21, 49, 17, 48, 86, 71, 53, 32, 56, 45, 69, 31, 94, 28, 25, 44, 79, 37, 24, 42, 40, 29,
            #          85, 35, 98, 65, 95, 46]
            # band3 = [41, 129, 91, 90, 115, 49, 168, 172, 132, 127, 107, 43, 171, 51, 81, 123, 68, 120, 6, 44, 32, 48,
            #          88, 137, 9, 22, 108, 141, 3, 67]
            for i, data in enumerate(train_dataloader, start=0):
                X, labels = data
                [bath, n_feature, row, col] = X.shape
                images = torch.zeros([bath, band_num, row, col]).type_as(X)
                for i in range(band_num):
                    images[:, i, :, :] = X[:, band1[i]-1, :, :]
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(X, images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch==30:
                netG_file_name = models_path + 'DA2_netG_99.880_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
            if epoch==60:
                netG_file_name = models_path + 'DA2_netG_99.880_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
            if epoch==80:
                netG_file_name = models_path + 'DA2_netG_99.880_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)


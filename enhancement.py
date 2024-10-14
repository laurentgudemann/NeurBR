from glass_post.detail.ltm.neur_br.net.dip_model import BilateralDIP
from glass_post.detail.ltm.neur_br.net.noise import get_noise
from glass_post.detail.ltm.neur_br.utils.image_io import np_to_torch, pil_to_np, torch_to_np


import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import misc
from torchvision import transforms


import time


class Enhancement(object):
    def __init__(self, image_name, image, plot_during_training=False, show_every=200, num_iter=200):
        self.image = image
        self.img = image
        self.full_resolution = image.size
        self.sigma = 0.1
        self.image_np = None
        self.images_torch = None
        self.plot_during_training = plot_during_training
        # self.ratio = ratio
        self.show_every = show_every
        self.image_name = image_name
        self.num_iter = num_iter
        self.learning_rate = 0.03
        self.input_depth = (
            2  # if input_type = 'noise', set this value as 3, else if input_type = 'meshgrid', set this value as 2
        )
        self.data_type = torch.cuda.FloatTensor
        # self.data_type = torch.FloatTensor
        self.illumination_net_inputs = None
        self.original_illumination = None
        self.illumination_net = None
        self.total_loss = None
        self.illumination_out = None
        self.feature = None
        self.current_result = None
        self.best_result = None
        self.weight_map = None
        self.use_grid = True
        self.downsampling_size = (128, 128)
        self.noise_size = (128, 128)
        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_losses()
        # self._init_parameters()

    def _init_images(self):
        self.original_illumination = transforms.Compose(
            [transforms.Resize(self.downsampling_size), transforms.Grayscale(), transforms.ToTensor()]
        )(self.image).type(self.data_type)

        self.image = transforms.Resize(self.downsampling_size)(self.image)
        self.image_np = pil_to_np(self.image)  # pil image to numpy
        self.image_torch = np_to_torch(self.image_np).type(self.data_type)

    def _init_parameters(self):
        self.parameters = [p for p in self.illumination_net.parameters()]
        s = sum([np.prod(list(p.size())) for p in self.parameters])
        print("Number: %d" % s)

    def _init_inputs(self):
        if self.image_torch is not None:
            print((self.image_torch.shape[2], self.image_torch.shape[3]))
        # input_type = 'noise'
        input_type = "meshgrid"
        self.illumination_net_inputs = (
            get_noise(self.input_depth, input_type, self.noise_size).type(self.data_type).detach()
        )

    def _init_nets(self):
        self.illumination_net = BilateralDIP(in_channels=self.input_depth, use_gird=self.use_grid).type(self.data_type)

    def _init_losses(self):
        self.mse_loss = nn.MSELoss().type(self.data_type)

    def optimize(self, compute_layernorm=False):
        optimizer = torch.optim.Adam(self.illumination_net.parameters(), lr=self.learning_rate)
        print("Processing: {}".format(self.image_name.split("/")[-1]))
        norm_iter = {}
        nuclear_norm_iter = {}
        Fro_nuclear_ratio = {}
        start = time.time()
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure()
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()
        self.get_enhanced()
        end = time.time()
        print("time:%.4f" % (end - start))
        cv2.imwrite(self.image_name, self.best_result)
        if compute_layernorm:
            return norm_iter, nuclear_norm_iter, Fro_nuclear_ratio

    def _optimization_closure(self):
        reg_noise_std = 1 / 1000.0
        illumination_net_input = self.illumination_net_inputs + (
            self.illumination_net_inputs.clone().normal_() * reg_noise_std
        )

        self.feature, self.illumination_out = self.illumination_net(illumination_net_input, self.original_illumination)
        # grid = self.feature.cpu().clone().squeeze(0)#.permute(1,0,2,3)
        # for i in range(3):
        #     for j in range(8):
        #         tep=grid[i, j, :, :]
        #         imag = Image.fromarray((tep*255).byte().numpy(), mode="L")
        #         imag.save('output/grid-c{}-d{}.png'.format(str(i), str(j)))
        self.total_loss = self.mse_loss(self.illumination_out, self.image_torch)
        self.total_loss.backward()

    def _plot_closure(self, step):
        if step % self.show_every == self.show_every - 1:
            print("Iteration {:5d}    Loss {:5f}".format(step, self.total_loss.item()))
            self.get_enhanced()

    def gamma_trans(self, img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)

    def get_enhanced(self):
        (R, G, B) = self.img.split()
        ini_illumination = torch_to_np(self.illumination_out).transpose(1, 2, 0)
        # plot_image_grid("step-{}".format(step), [np.clip(self.image_np, 0, 1),
        #                                          np.clip(ini_illumination, 0, 1)])
        ini_illumination = misc.imresize(ini_illumination, (self.full_resolution[1], self.full_resolution[0]))
        ini_illumination = np.max(ini_illumination, axis=2)
        # # feature = torch_to_np(self.feature)#.transpose(1, 2, 0)
        # # cv2.imwrite('output/feature-{}.png'.format(step), feature.sum(axis=0))
        ini_illumination = self.gamma_trans(ini_illumination, 0.6)

        R = R / ini_illumination
        G = G / ini_illumination
        B = B / ini_illumination
        self.best_result = np.clip(cv2.merge([B, G, R]) * 255, 0.02, 255).astype(np.uint8)


def lowlight_enhancer(image_name, image):
    s = Enhancement(image_name, image)
    s.optimize()

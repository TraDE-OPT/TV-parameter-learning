# -*- coding: utf-8 -*-
#
#    Copyright (C) 2022 Kristian Bredies (kristian.bredies@uni-graz.at)
#                       Enis Chenchene (enis.chenchene@uni-graz.at)
#                       Alireza Hosseini (hosseini.alireza@ut.ac.ir)
#
#    This file is part of the example code repository for the paper:
#
#      K. Bredies, E. Chenchene, A. Hosseini.
#      A hybrid proximal generalized conditional gradient method and application
#      to total variation parameter learning, 2022.
#      Submitted to ECC23, within the EUCA Series of European Control Conferences,
#      To be held in Bucharest, Romania, from June 13 to June 16, 2023.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
This file contains all the experiments in Section III.D in:

K. Bredies, E. Chenchene, A. Hosseini.
A hybrid proximal generalized conditional gradient method and
application to total variation parameter learning, 2022.

Submitted to ECC23, within the EUCA Series of European Control Conferences
To be held in Bucharest, Romania, from June 13 to June 16, 2023.
"""

import random
import numpy as np

# plots and images
from matplotlib import rc
from PIL import Image

import structures as st
from plots import plot_exp1
from denoising import denoise, denoise_large
from data import apply_noise, create_sample
from train_quadratic_model import train_qm
from train_constant_model import train_cm
from single_patch import single_patch_best_par


def load_dataset(noise_level):

    print("\n \nLoading the dataset...")
    np.random.seed(1)
    dataset = create_sample(noise_level, "train_set")

    return dataset


def train_quadratic_model(dataset, maxit=25000):

    # train a quadratic model
    lam = 50
    A, _Sig, _Du_list = train_qm(dataset, lam, maxit)

    return A


def train_constant_model(dataset, maxit=20000):

    # Train a constant model
    lam = 50
    print("\n \nExtracting 1000 patches from training set...")
    dataset_small = random.sample(dataset, 1000)
    best_constant, _Sig, _Du_list = train_cm(dataset_small, lam, maxit)

    return best_constant


def experiment1(A, noise_level):

    np.random.seed(1)
    size_patches = 16

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 17})
    rc('text', usetex=True)

    objects = []

    for test in range(6):

        print(f"\n## Starting image {test+1}")
        img = Image.open(f"Figures/Figures_exp1/im{test+1}.png").convert('L')
        sxx, syy = np.shape(img)
        sx = int(((sxx/1.3)//size_patches)*size_patches)
        sy = int(((syy/1.3)//size_patches)*size_patches)
        true_image = 255-np.array(img.resize((sx, sy)), dtype=np.float64)
        true_image /= 255
        print(f"picture size is {sx} times {sy}")

        noisy_image = apply_noise(true_image, noise_level)
        denoised_image, params = denoise_large(true_image, noisy_image, A, size_patches)
        objects.append([true_image, noisy_image, denoised_image, params])

    plot_exp1(objects)
    print("Experiment 1 ended. Results have been saved as experiment1_comparisons.pdf\n")


def experiment2(A, best_constant, noise_level):

    np.random.seed(1)

    # structures
    Dx = st.grad_x(16)
    Dy = st.grad_y(16)
    M1 = -Dx.T
    M2 = -Dy.T

    print("\nLoading test set...")
    test_set = create_sample(noise_level, "test_set", size_patches=16)

    test_set = random.sample(test_set, 200)
    print("Test set reduced to 200.")

    consts = np.logspace(-4, -1, 8)
    consts = np.append(consts, best_constant)

    N = len(test_set)
    MSE_alpha = 0
    MSE_alpha_constants = np.zeros(len(consts))
    MSE_u = 0
    MSE_u_constants = np.zeros(len(consts))

    count = 1

    for (true_patch, noisy_patch) in test_set:

        if count % 50 == 1:
            print(f"\n ## Done {count} patches. {N-count} remain...\n")

        # predict parameter with quadratic model
        patch_noisy_b = np.append(noisy_patch, 1)
        a_learned = patch_noisy_b.T@A@patch_noisy_b

        # compute best parameter
        a, sig, _Du_list = single_patch_best_par(true_patch, noisy_patch, 10, maxit=50000,
                                                     show_details=False)

        # update mean squared error for the parameter
        MSE_alpha += (a-a_learned)**2/N

        if count % 50 == 1:
            print(f"MSE_alpha quadratic model \n   Equal to: {np.round(MSE_alpha*N/count,7)}")

        for i, const in enumerate(consts):
            MSE_alpha_constants[i] += (a-consts[i])**2/N

            if count % 50 == 1:
                print(f"MSE_alpha constant parameter {const}\n   \
Equal to: {np.round(MSE_alpha_constants[i]*N/count,7)}")

        # compute approximate solution wrt to best parameter
        denoised_patch = st.div(sig, M1, M2)+noisy_patch

        # update mean squared error for the image
        MSE_u += np.linalg.norm(true_patch-denoised_patch)**2/N
        if count % 50 == 1:
            print(f"\nMSE_u quadratic model \n   Equal to: {np.round(MSE_u*N/count,4)}")

        # denoise with respect to constant (non-adaptive) parameters
        for i, const in enumerate(consts):
            denoised_patch, _warning = denoise(noisy_patch, const, maxit=100000)
            MSE_u_constants[i] += np.linalg.norm(true_patch-denoised_patch)**2/N

            if count % 50 == 1:
                print(f"MSE_u constant parameter {const}\n   \
Equal to: {np.round(MSE_u_constants[i]*N/count,4)}")

        count += 1

    return MSE_alpha, MSE_u, MSE_alpha_constants, MSE_u_constants

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
This file contains useful functions to plot our numerical results.
For details and references, see Section III.D in:

K. Bredies, E. Chenchene, A. Hosseini.
A hybrid proximal generalized conditional gradient method and
application to total variation parameter learning, 2022.

Submitted to ECC23, within the EUCA Series of European Control Conferences
To be held in Bucharest, Romania, from June 13 to June 16, 2023.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import structures as st


def compare_results(Sig, dataset, num_plots):

    n, N, _ = np.shape(Sig)
    p = int(np.sqrt(n))

    Dx = st.grad_x(p)
    M1 = -Dx.T
    Dy = st.grad_y(p)
    M2 = -Dy.T

    # plot num_plots random patches
    indexes = random.sample(list(range(N)), num_plots)

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 17})
    rc('text', usetex=True)

    # plot
    _fig, axs = plt.subplots(num_plots, 3, figsize=(6, 10))
    axs = np.atleast_2d(axs)

    for i in range(num_plots):

        axs[i, 0].imshow(dataset[indexes[i]][0].reshape((p,p)), cmap="gray", vmin=0, vmax=1.0)
        if i == 0:
            axs[i, 0].set_title('Ground-truth')
        axs[i, 0].text(-4,15, f'patch {indexes[i]}', rotation='vertical')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(dataset[indexes[i]][1].reshape((p,p)), cmap="gray", vmin=0, vmax=1.0)
        if i == 0:
            axs[i, 1].set_title('Noisy')
        axs[i, 1].axis('off')

        debiased = st.div(Sig[:, indexes[i], :], M1, M2)+dataset[indexes[i]][1]
        axs[i, 2].imshow(debiased.reshape((p, p)), cmap="gray", vmin=0, vmax=1.0)
        if i == 0:
            axs[i, 2].set_title('Denoised')
        axs[i, 2].axis('off')

    plt.savefig("patches.pdf", bbox_inches='tight')
    plt.show()


def plot_D(Du_list, Save=True):

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})
    rc('text', usetex=True)

    plt.figure(figsize=(10, 5))
    plt.semilogy(list(range(len(Du_list))), Du_list, linewidth=0.25, color='k')
    plt.grid(True, which="both")
    plt.xlim([0, len(Du_list)])
    plt.xlabel(r"Iterations $(k)$")
    plt.ylabel(r"$D(A^k,v^k)$")
    if Save:
        plt.savefig("du.pdf", bbox_inches='tight')
    plt.show()


def plot_exp1(objects):

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 17})
    rc('text', usetex=True)

    # plot
    _fig, axs = plt.subplots(6, 4, figsize=(11, 16))
    axs = np.atleast_2d(axs)

    for (i, test) in enumerate(objects):

        true_image = test[0]
        noisy_image = test[1]
        denoised_image = test[2]
        params = test[3]

        axs[i, 0].imshow(true_image, cmap="gray_r", vmin=0, vmax=1.0)
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        if i == 0:
            axs[i, 0].set_title('Ground-truth')

        axs[i, 1].imshow(noisy_image, cmap="gray_r", vmin=0, vmax=1.0)
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])
        if i == 0:
            axs[i, 1].set_title('Noisy')

        axs[i, 2].imshow(denoised_image, cmap="gray_r", vmin=0, vmax=1.0)
        axs[i, 2].set_xticks([])
        axs[i, 2].set_yticks([])
        if i == 0:
            axs[i, 2].set_title('Denoised')

        pars_img = axs[i, 3].imshow(params, cmap="hot")
        axs[i, 3].set_xticks([])
        axs[i, 3].set_yticks([])
        im_ratio = params.shape[0]/params.shape[1]
        plt.colorbar(pars_img, ax=axs[i, 3], fraction=0.046*im_ratio, pad=0.04)
        if i == 0:
            axs[i, 3].set_title("Predicted parameters")

    plt.savefig("experiment1_comparisons.pdf", bbox_inches='tight')
    plt.savefig("experiment1_comparisons.png", bbox_inches='tight')
    plt.show()

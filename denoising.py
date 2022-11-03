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
This file contains the functions for TV denoising of a single patch "denoise" and
of a figure divided into several patches "denoise_large".
For details and references, see Section III.D in:

K. Bredies, E. Chenchene, A. Hosseini.
A hybrid proximal generalized conditional gradient method and
application to total variation parameter learning, 2022.

Submitted to ECC23, within the EUCA Series of European Control Conferences
To be held in Bucharest, Romania, from June 13 to June 16, 2023.
"""

import numpy as np
import structures as st


def denoise(f, par, maxit=100000, show_details=False):
    '''
    Implements primal-dual for TV-denoising with parameter "par", "f" being the noisy image.
    '''

    # structure
    n = len(f)
    p = int(np.sqrt(n))
    Dx = st.grad_x(p)
    Dy = st.grad_y(p)
    M1 = -Dx.T
    M2 = -Dy.T

    # initialize
    psi = np.zeros(n)
    sig = np.zeros((n, 2))

    # parameters
    tau = 1/np.sqrt(8)
    mu = 1/np.sqrt(8)

    res = 1
    it = 1
    while res > 1e-7 and it < maxit:

        psi_old = np.copy(psi)
        psi = st.prox_fidelity(tau/par, psi+tau*st.div(sig, M1, M2), f)
        sig_old = np.copy(sig)
        sig = st.proj_12(sig+mu*st.grad(2*psi-psi_old, Dx, Dy, n))

        # residual
        res_p = np.linalg.norm( (psi_old-psi)/tau + st.div(sig_old-sig, M1, M2))**2
        res_d = np.linalg.norm( (sig_old-sig)/mu - st.grad(psi_old-psi, Dx, Dy, n))**2
        res = res_p+res_d
        it += 1

        if show_details:
            if it % 10000 == 5 and it > 10000:
                print(f"##### iteration: {it}")
                print(f"primal residual: {res_p}")
                print(f"dual residual: {res_d}")
                print(f"residual: {res_p+res_d}")

    if it == maxit:
        warning = 1
    else:
        warning = 0

    return psi, warning


def denoise_large(true_image, noisy_image, A, size_patches):
    '''
    Splits "noisy_image" into several patches and denoises every patch individually.
    '''

    px, py = np.shape(noisy_image)

    tot_patches = int(px*py/size_patches**2)
    print(f"Splitting into {tot_patches} patches")

    denoised_image = np.zeros((px, py))
    params = np.zeros((int(px/size_patches), int(py/size_patches)))

    done = 0
    for i in np.arange(0, px, size_patches):
        for j in np.arange(0, py, size_patches):

            # extract patch
            noisy_patch = noisy_image[i:i+size_patches, j:j+size_patches]
            noisy_patch_vect = noisy_patch.reshape(size_patches**2)

            # compute best parameter
            noisy_patch_vect_b = np.append(noisy_patch_vect, 1)  # add bias
            best_par = noisy_patch_vect_b.T@A@noisy_patch_vect_b

            # debias with learned parameter
            denoised_patch, warning = denoise(noisy_patch_vect, best_par)

            if warning == 1:
                print(f"Warning: denoising in patch {(i,j)} failed.")

            denoised_image[i:i+size_patches, j:j+size_patches] = \
              denoised_patch.reshape((size_patches,size_patches))
            params[int(i/size_patches), int(j/size_patches)] = best_par

            done += 1
            if done % 50 == 0:
                print(f"Done {done} patches. Still {tot_patches-done} remain...")

    return denoised_image, params

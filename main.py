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
Run this script to reproduce all the numerical experiments obtained in Section III.D in:

K. Bredies, E. Chenchene, A. Hosseini.
A hybrid proximal generalized conditional gradient method and
application to total variation parameter learning, 2022.

Submitted to ECC23, within the EUCA Series of European Control Conferences
To be held in Bucharest, Romania, from June 13 to June 16, 2023.

Please note that the training of the quadratic model may take several hours.
"""

import zipfile
import io
import requests
from experiments import (load_dataset, train_quadratic_model, train_constant_model,
                             experiment1, experiment2)

if __name__ == "__main__":

    print("\n \n *** Downloading dataset from Zenodo. *** \nPlease wait...")
    # download the dataset from Zenodo, DOI:10.5281/zenodo.7267054
    zip_file_url = 'https://zenodo.org/record/7267054/files/Figures.zip?download=1'

    r = requests.get(zip_file_url, stream=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall()

    print("Dataset downloaded.")
    noise_level = 0.05

    # load the training set
    dataset = load_dataset(noise_level)

    # train the quadratic model
    A = train_quadratic_model(dataset)

    # train the constant model
    best_constant = train_constant_model(dataset)

    print("\n \n *** Starting experiment 1. ***")
    experiment1(A, noise_level)

    print("\n \n *** Starting experiment 2. ***")
    MSE_alpha, MSE_u, MSE_alpha_constants, MSE_u_constants = experiment2(A, best_constant,
                   noise_level)

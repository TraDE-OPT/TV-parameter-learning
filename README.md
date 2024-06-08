# A hybrid proximal generalized conditional gradient method and application to total variation parameter learning: Example code

This repository contains the experimental source code to reproduce the numerical experiments in:

* K. Bredies, E. Chenchene, A. Hosseini. A hybrid proximal generalized conditional gradient method and application to total variation parameter learning. 2022. [ArXiv preprint](https://arxiv.org/abs/2211.00997)

To reproduce the results of the numerical experiments in Section III.D, run:
```bash
python3 main.py
```

If you find this code useful, please cite the above-mentioned paper:
```BibTeX
@INPROCEEDINGS{chb23,
  author={Chenchene, Enis and Hosseini, Alireza and Bredies, Kristian},
  booktitle={2023 European Control Conference (ECC)}, 
  title={A hybrid proximal generalized conditional gradient method and application to total variation parameter learning}, 
  year={2023},
  volume={},
  number={},
  pages={1--6}
}
```

## Requirements

Please make sure to have the following Python modules installed, most of which should be standard.

* [numpy>=1.20.1](https://pypi.org/project/numpy/)
* [scipy>=1.6.2](https://pypi.org/project/scipy/)
* [matplotlib>=3.3.4](https://pypi.org/project/matplotlib/)
* [Pillow>=8.2.0](https://pypi.org/project/Pillow/)
* [glob2>=0.7](https://pypi.org/project/glob2/)
* [requests>=2.25.1](https://pypi.org/project/requests/)

## Acknowledgments  

* | ![](<euflag.png>) | K.B. and E.C. have received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement no. 861137. |
  |-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
* Training and test datasets have been downloaded from [Pixabay](https://pixabay.com/) with permission and are available on [Zenodo](https://doi.org/10.5281/zenodo.7267054).
  
## License  
This project is licensed under the GPLv3 license - see [LICENSE](LICENSE) for details.

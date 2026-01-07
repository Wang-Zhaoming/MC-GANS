# MC-GANS
This is the code of paper "[**Multi-Population Co-Evolutionary Generative Adversarial Network Architecture Search for Zero-Shot Learning**](https://doi.org/10.1109/TEVC.2026.3650926)", accepted to *IEEE Transactions on Evolutionary Computation (TEVC)*.

## Dependencies
- Environment: All of our experiments run and test in Python 3.10.6. The `environment.yml` file lists all the environment configurations. You can use the following conda command to create the MC-GANS environment:
```
conda env create -f environment.yml
```
- Dataset: Please download the [dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly), and change `--dataroot` in `config.py` to your local path. Please refer to [SDGZSL](https://github.com/uqzhichen/SDGZSL) for the finetuned features.
- Attribute: The attributes for AWA2, SUN, and APY are available in the datasets. Please download the 1024-D [CUB semantic](https://github.com/Hanzy1996/CE-GZSL) and save it to the data path.

# MC-GANS
This is the code of paper "[**Multi-Population Co-Evolutionary Generative Adversarial Network Architecture Search for Zero-Shot Learning**](https://doi.org/10.1109/TEVC.2026.3650926)", accepted to *IEEE Transactions on Evolutionary Computation (TEVC)*.

## Dependencies
- Environment: All of our experiments run and test in Python 3.10.6. The `environment.yml` file lists all the environment configurations. You can use the following conda command to create the MC-GANS environment:
```
conda env create -f environment.yml
```
- Dataset: Please download the [dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly), and change `--dataroot` in `config.py` to your local path. Please refer to [SDGZSL](https://github.com/uqzhichen/SDGZSL) for the finetuned features.
- Attribute: The attributes for AWA2 and SUN are available in the datasets. Please download the 1024-D [CUB semantic](https://github.com/Hanzy1996/CE-GZSL) and save it to the data path.

## Generative Adversarial Network Architecture Search
You can use the following commands to search for GAN models on the CUB, SUN, and AWA2 datasets, respectively:
- CUB
```
python search.py --syn_num 50 --nepoch 500 --critic_iter 8 --dataset CUB --class_embedding sent --attSize 1024 --nz 1024 --att_std 0.02 --lambda_1 2.5 --temperature 0.04 --w_up 10.0 --nepoch_classifier 150
```
- SUN
```
python search.py --syn_num 30 --nepoch 500 --critic_iter 8 --batch_size 512 --dataset SUN --class_embedding att --attSize 102 --nz 102 --att_std 0.02 --lambda_1 2.5 --temperature 0.04 --w_up 10.0 --nepoch_classifier 80
```
- AWA2
 ```
python search.py --syn_num 100 --nepoch 500 --test_data True --batch_size 256 --critic_iter 8 --dataset AWA2 --class_embedding att --attSize 85 --nz 85 --att_std 0.02 --lambda_1 0.005 --temperature 0.04 --w_up 10.0 --nepoch_classifier 80
```

## Retrain
Use the following commands for model retraining and zero-shot prediction:
- CUB
```
python retrain.py --syn_num 50 --nepoch 500 --critic_iter 5 --original True --lr 0.0001 --dataset CUB --class_embedding sent --attSize 1024 --nz 1024 --att_std 0.02 --lambda_1 2.5 --temperature 0.04 --w_up 10.0 --nepoch_classifier 150
```
- SUN
```
python retrain.py --syn_num 30 --nepoch 500 --critic_iter 5 --batch_size 512 --original False --dataset SUN --class_embedding att --attSize 102 --nz 102 --att_std 0.02 --lambda_1 2.5 --temperature 0.04 --w_up 10.0 --nepoch_classifier 80
```
- AWA2
 ```
python retrain.py --syn_num 100 --nepoch 500 --critic_iter 5 --batch_size 256 --original False --dataset AWA2 --class_embedding att --attSize 85 --nz 85 --att_std 0.02 --lambda_1 0.005 --temperature 0.04 --w_up 10.0 --nepoch_classifier 80
```

## Citation
If this work is helpful for you, please cite our paper.
```
@ARTICLE{11329048,
  author={Wang, Zhaoming and Xue, Yu and Neri, Ferrante},
  journal={IEEE Transactions on Evolutionary Computation}, 
  title={Multi-Population Co-Evolutionary Generative Adversarial Network Architecture Search for Zero-Shot Learning}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Zero shot learning;Generative adversarial networks;Computer architecture;Generators;Training;Optimization;Semantics;Evolutionary computation;Attention mechanisms;Computational modeling;Evolutionary computation;Zero-shot learning;Neural architecture search;Multi-population co-evolution;Generative adversarial network},
  doi={10.1109/TEVC.2026.3650926}}

```

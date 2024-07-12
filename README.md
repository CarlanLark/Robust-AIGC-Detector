# Robust-AIGC-Detector

Code for ACL 2024 long paper: Are AI-Generated Text Detectors Robust to Adversarial Perturbations?

### Environments

```bash
torch==1.11.0
transformers==4.30.2
textattack==0.3.9 
tensorflow==2.9.1 
tensorflow_hub==0.15.0
```


### Data Preparation

```bash
unzip data_in.zip
mkdir data_out
```

### Training
```bash
$ bash train.sh
```

### Checkpoints
The checkpoints of in-domain detector, cross-domain detector, and cross-genre detector can be found in <https://huggingface.co/CarlanLark/AIGT-detector-in-domain>. (These detectors are trained on the same training set and evaluated on different test sets.)

The checkpoint of mixed-source detector can be found in <https://huggingface.co/CarlanLark/AIGT-detector-mixed-source>.

### Robustness Evaluation
```bash
$ bash attack.sh
```

### Citation
If you find our work useful to your research, you can cite the paper below:
```bash
@article{huang2024ai,
  title={Are AI-Generated Text Detectors Robust to Adversarial Perturbations?},
  author={Huang, Guanhua and Zhang, Yuchen and Li, Zhe and You, Yongjian and Wang, Mingze and Yang, Zhouwang},
  journal={arXiv preprint arXiv:2406.01179},
  year={2024}
}
```
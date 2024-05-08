# ECOGEN: Bird Sounds Generation using Deep Learning

This repository contains the code for the paper [ECOGEN: Bird Sounds Generation using Deep Learning](https://doi.org/10.1111/2041-210X.14239). 
The paper proposes a novel method for generating bird sounds using deep learning by leveraging VQ-VAE2 network architecture.
The proposed method is able to generate bird sounds that aims to increase the dataset size for bird sound classification tasks.



## Dataset
The dataset used in this paper is the Xeno-Canto dataset from Kaggle. The dataset can be downloaded from [Part 1](https://www.kaggle.com/rohanrao/xeno-canto-bird-recordings-extended-a-m) and [Part 2](https://www.kaggle.com/rohanrao/xeno-canto-bird-recordings-extended-n-z).

## Model Checkpoint
MOdel checkpoints can be found in the [OSF Link](https://doi.org/10.17605/OSF.IO/YQDJ9) folder.

## API with sample_generator.SampleGenerator class

load model using a local checkpoint file
```{python}
m = SampleGenerator(model_path)
```

preprocess a .wav, .npy, or .png file into torch.tensor
```{python}
sample = m.load_sample(path_to_wav) # 
```

encoding and decoding
```{python}
quant_top, quant_bottom, diff, id_top, id_bottom = model.encode(sample)
reconstructed_sample = model.decode(quant_top,quant_bottom)'

# encode and decode in one step:
reconstructed_sample = encode_decode(sample)
```

convert generated sample (np.array formatted spectrogram) back to audio signal using Griffin-Lim algorithm to estimate phase
```{python}
signal, sr = model.to_audio_signal(reconstructed_sample)
```
Feature-space augmentation: generate novel samples by perturbing or interpolating feature vectors
```{python}
# feature-space noise addition
noise_sample = m.add_feature_space_noise(sample,strength=40)

# feature-space interpolation
s1 = m.load_sample(path_to_wav1)
s2 = m.load_sample(path_to_wav2)
interpolated_sample = m.feature_space_interpolation(s1,s2,ratio=.5)
```

## Requirements
The code is tested on Python 3.7.7 and PyTorch 1.13.1. The required packages can be installed using the following command:
```
git clone https://github.com/ixobert/birds-generation
cd ./birds-generation/
#Use this line for M1 series Mac
pip install -r mac-m1-requirements.txt

#Otherwise use this line
pip install -r requirements.txt
```

## Preprocessing
The preprocessing steps are as follows to train the ECOGEN VQ-VAE2 model:
1. Convert the audio files to mono channel
2. Resample the audio files to 22050 Hz
3. Trim the audio files to 5 seconds


## Usage
We heavily used Hydra to manage the configuration files. The configuration files can be found in the `src/configs` folder. See the [Hydra documentation](https://hydra.cc/docs/intro) for more details.

### ECOGEN Training
The ECOGEN training code is inspired from the [VQ-VAE2 implementation](https://github.com/rosinality/vq-vae-2-pytorch).
The training code can be found in the `src` folder.
The code expects the dataset to be in the following format:
```
./birds-songs/dataset/train.txt|test.txt
```

The train,test and validation text files contains the path to the audio files. See below an example of a train.txt file:

```
birds-song/1.wav
birds-song/2.wav
birds-song/3.wav
```


To train the ECOGEN model, run the following command:
```
python ./src/train_vqvae.py  dataset="xeno-canto" mode="train" lr=0.00002 nb_epochs=25000 log_frequency=1 dataset.batch_size=420 dataset.num_workers=8 run_name="ECOGEN Training on Xeno Canto"  tags=[vq-vae2,xeno-canto] +gpus=[1] debug=false
```
You will need to update the content of `configs/dataset` to point your custom dataset folder.


#### Sample Generation
The current version of ECOGEN supports 2 types of augmentation, interpolation and noise. The generate_samples script outputs the generated spectrograms in the `out_folder` folder as numpy files`.npy`.

To generate the bird songs spectrograms, run the following command:
#### Noise augmentation
```
python ./src/generate_samples.py  --data_paths="/folder_path/*.wav" --out_folder="./generated_samples"  --model_path="/path/to/model.ckpt"  --augmentations=noise --num_samples=3
```
### Interpolation augmentation
```
python ./src/generate_samples.py  --data_paths="/folder_path/*.wav" --out_folder="./generated_samples"  --model_path="/path/to/model.ckpt"  --augmentations=interpolation --num_samples=3
```

You can view the generated spectrograms using the following commands:
```python
import numpy as np
import matplotlib.pyplot as plt
spec = np.load("generated_spectrogram.npy")
plt.imshow(spec)
```

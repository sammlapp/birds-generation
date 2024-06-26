import os
from albumentations import RandomScale, Rotate, GaussNoise, GaussianBlur
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import torchvision
import numpy as np
from functools import partial
import albumentations as A
from torchvision.transforms import transforms as T
import torchaudio.transforms as AudioTransform
try:
    from imagedataset import ImageDataset
    from audiodataset import AudioDataset
    from rawaudiodataset import RawAudioDataset
    from codebookdataset import LMDBDataset
except ModuleNotFoundError:
    from .imagedataset import ImageDataset
    from .audiodataset import AudioDataset
    from .rawaudiodataset import RawAudioDataset
    from .codebookdataset import LMDBDataset


class CodeBookDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = kwargs.get('config')
        self.batch_size = self.config.get('batch_size') #TODO: DO the same in train_vqvae
    
    def setup(self, stage=None):
        self.dataset = LMDBDataset(data_path=self.config['lmdb_path'])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.config['num_workers'])


class ImagesDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = kwargs.get('config')
        self.batch_size = self.config.get('batch_size', 2)
    
    def setup(self, stage=None):
        #################
        #TODO: To delete: this code makes my life easier during development
        import sys
        platform = sys.platform.lower()
        print(f"Running on : {platform}")
        if platform == 'darwin':
            root_ = "/Users/test/Documents/Projects/Master/"
            self.config['root_dir'] = os.path.join(root_, "faces94/female")
            self.config['train_path'] = os.path.join(root_, "faces94/female_train.txt")
        #################
        self.dataset = ImageDataset(data_path=self.config['train_path'], root_dir=self.config['root_dir'], classes_name=self.config['classes_name'])
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.config['num_workers'])


class SpectrogramsDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = kwargs.get('config')
        self.batch_size = self.config.get('batch_size', 2)

    @classmethod
    def _custom_collate(self, batch_):
        def contains_none(b):
            for elem in b:
                if elem is None:
                    return False
            return True
        batch = list(filter(lambda x: contains_none(x), batch_))
        input = torch.stack([b[0] for b in batch])
        label = torch.stack([b[1] for b in batch])
        if len(batch[0]) == 2:
            return input, label
        filepath = [b[2] for b in batch]
        return input, label, filepath

    def setup(self, stage=None):
        transforms = self.config.get('augmentation_mode',[])
        print("Transforms function", transforms)
        transforms_ops = []
        if 'image_base' in transforms:
            transforms_ops = A.Compose([
                    RandomScale(),
                    Rotate(limit=3),
                    GaussNoise(10.0),
                    GaussianBlur(),
                ])
        else:
            transforms_ops = []
            # for op in transforms:
                # if op.lower() == "frequencymasking":
                    # transforms_ops.append(AudioTransform.FrequencyMasking(15))
                # elif op.lower() == "timemasking":
                    # transforms_ops.append(AudioTransform.TimeMasking(15))
                # elif op.lower() == "masking":
                    # transforms_ops.append(AudioTransform.FrequencyMasking(15))
                    # transforms_ops.append(AudioTransform.TimeMasking(15))
            transforms_ops = partial(self.custom_augment, transforms=transforms)
            print("Transforms ops", transforms_ops)
            # transforms_ops.append(T.Resize((512, 32)))
            # transforms_ops = T.Compose(transforms_ops)

        # self.dataset      = RawAudioDataset(data_path=self.config['train_path'], root_dir=self.config['root_dir'], classes_name=self.config['classes_name'], sr=self.config['sr'], window_length=self.config['sr']*4, spec=self.config['use_mel'], resize=self.config['resize'], return_tuple=self.config['return_tuple'], return_tuple_of3=self.config.get('return_tuple_of3', True), use_spectrogram=self.config.get('use_mel', False), use_cache=self.config.get('use_cache', True), use_rgb=self.config.get('use_rgb', False), transform=transforms_ops)
        # self.val_dataset  = RawAudioDataset(data_path=self.config['val_path'], root_dir=self.config['root_dir'], classes_name=self.config['classes_name'], sr=self.config['sr'], window_length=self.config['sr']*4, spec=self.config['use_mel'], resize=self.config['resize'], return_tuple=self.config['return_tuple'], return_tuple_of3=self.config.get('return_tuple_of3', True), use_spectrogram=self.config.get('use_mel', False), use_cache=self.config.get('use_cache', True), use_rgb=self.config.get('use_rgb', False))
        # self.test_dataset = RawAudioDataset(data_path=self.config['test_path'], root_dir=self.config['root_dir'], classes_name=self.config['classes_name'], sr=self.config['sr'], window_length=self.config['sr']*4, spec=self.config['use_mel'], resize=self.config['resize'], return_tuple=self.config['return_tuple'], return_tuple_of3=self.config.get('return_tuple_of3', True), use_spectrogram=self.config.get('use_mel', False), use_cache=self.config.get('use_cache', True), use_rgb=self.config.get('use_rgb', False))

        self.dataset      = AudioDataset(data_path=self.config['train_path'], root_dir=self.config['root_dir'], classes_name=self.config['classes_name'], sr=self.config['sr'], window_length=self.config['sr']*4, spec=self.config['use_mel'], resize=self.config['resize'], return_tuple=self.config['return_tuple'], return_tuple_of3=self.config.get('return_tuple_of3', True), use_spectrogram=self.config.get('use_mel', False), use_cache=self.config.get('use_cache', True), use_rgb=self.config.get('use_rgb', False), transforms=transforms_ops)
        self.val_dataset  = AudioDataset(data_path=self.config['val_path'], root_dir=self.config['root_dir'], classes_name=self.config['classes_name'], sr=self.config['sr'], window_length=self.config['sr']*4, spec=self.config['use_mel'], resize=self.config['resize'], return_tuple=self.config['return_tuple'], return_tuple_of3=self.config.get('return_tuple_of3', True), use_spectrogram=self.config.get('use_mel', False), use_cache=self.config.get('use_cache', True), use_rgb=self.config.get('use_rgb', False))
        self.test_dataset = AudioDataset(data_path=self.config['test_path'], root_dir=self.config['root_dir'], classes_name=self.config['classes_name'], sr=self.config['sr'], window_length=self.config['sr']*4, spec=self.config['use_mel'], resize=self.config['resize'], return_tuple=self.config['return_tuple'], return_tuple_of3=self.config.get('return_tuple_of3', True), use_spectrogram=self.config.get('use_mel', False), use_cache=self.config.get('use_cache', True), use_rgb=self.config.get('use_rgb', False))


    def custom_augment(self, image, transforms):
        for transform in transforms:
            transform = transform.lower()
            if "masking" in transform:
                # print("Before", image.shape)
                # remove a portion of max_size pixels at random position in the image along the frequency axis using numpy slicing
                max_mask_size = 7
                if transform == "frequency_masking":
                    max_size = np.random.randint(0, max_mask_size)
                    start = np.random.randint(0, image.shape[1] - max_size)
                    image[:, start:start + max_size, :] = 0
                elif transform == "time_masking":
                    max_size = np.random.randint(0, max_mask_size)
                    start = np.random.randint(0, image.shape[2] - max_size)
                    image[:, :, start:start + max_size] = 0
                elif transform == "masking":
                    max_size = np.random.randint(0, max_mask_size)
                    start = np.random.randint(0, image.shape[1] - max_size)
                    image[:, start:start + max_size, :] = 0
                    max_size = np.random.randint(0, max_mask_size)
                    start = np.random.randint(0, image.shape[2] - max_size)
                    image[:, :, start:start + max_size] = 0
                # print("After", image.shape)

        out = {"image":image}
        return out 


    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.config['num_workers'], pin_memory=True, collate_fn=self._custom_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.config['num_workers'], pin_memory=True, collate_fn=self._custom_collate)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.config['num_workers'], pin_memory=True, collate_fn=self._custom_collate)

if __name__ == "__main__":
    root_ = "/Users/test/Documents/Projects/Master/"
    # root_ = "/media/future/Rapido/"
    root_dir = os.path.join(root_, "udem-birds/classes")
    config = dict(
        root_dir = root_dir,
        train_path = os.path.join(root_, "udem-birds/samples/train_list.txt"), 
        classes_name=['AMGP_1(tu-tit)', 'AMGP_2(tuuut)', 'WRSA_1(boingboingboing)'],
        sr=16384,
        window_length=16384*4,
        spec=True,
        use_mel=True,
        num_workers=4,
    )
    module = SpectrogramsDataModule(config=config)
    module.setup()
    for data in module.train_dataloader():
        print(type(data))
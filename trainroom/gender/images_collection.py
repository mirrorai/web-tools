from loaders import load_images_from_csv, load_images_from_dir
from os.path import join, isfile
from image_sample import SampleWithCache

class ImagesCollection(object):

    def __init__(self, samples, cfg):

        self._samples = []
        for sample in samples:
            img_path = sample[0]
            label = sample[1]
            if not isfile(img_path):
                continue
            image_sample = SampleWithCache(img_path, label, cfg)
            self._samples.append(image_sample)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, key):
        return self._samples[key]

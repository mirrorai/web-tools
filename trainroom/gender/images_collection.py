from loaders import load_images_from_csv, load_images_from_dir
from os.path import join

class ImagesCollection(object):

    def __init__(self, params, cfg):

        assert params['TYPE'] in ['CSV_FILE', 'MS_COCO', 'IMAGES_DIR']

        self._params = params
        if self._params['TYPE'] == 'CSV_FILE':
            csv_path = join(self._params['DB_PATH'], self._params['CSV_FILE'])
            self._samples = load_images_from_csv(csv_path, cfg)
        elif self._params['TYPE'] == 'IMAGES_DIR':
            self._samples = load_images_from_dir(self._params['DB_PATH'], cfg)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, key):
        return self._samples[key]

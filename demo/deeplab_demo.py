import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec, image
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow.compat.v1 as tf
from deepLabModel import DeepLabModel
from deepLabModel import vis_segmentation


MODEL_NAME = 'mobilenetv2_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = 'deeplab_model.tar.gz'

model_dir = tempfile.mkdtemp()
tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model, this might take a while...')
urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                   download_path)
print('download completed! loading DeepLab model...')

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')


IMG_LOC='local' # @param ['url', 'local']

SAMPLE_IMAGE = '0'  # @param ['image1', 'image2', 'image3']
IMAGE_URL = ('./data/0.jpg')  #@param {type:"string"}

_SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
               'deeplab/g3doc/img/%s.jpg?raw=true')


def run_visualization(image_dir):
  """Inferences DeepLab model and visualizes result."""
  if IMG_LOC == 'url':
    try:
      f = urllib.request.urlopen(image_dir)
      jpeg_str = f.read()
      original_im = Image.open(BytesIO(jpeg_str))
    except IOError:
      print('Cannot retrieve image. Please check url: ' + image_dir)
      return
  
  elif IMG_LOC == 'local':
    original_im = Image.open(image_dir)

  print('running deeplab on image %s...' % image_dir)
  resized_im, seg_map = MODEL.run(original_im)

  vis_segmentation(resized_im, seg_map)

if IMG_LOC == 'url':
  image_dir = _SAMPLE_URL % SAMPLE_IMAGE
elif IMG_LOC == 'local':
  image_dir = IMAGE_URL #% SAMPLE_IMAGE
run_visualization(image_dir)
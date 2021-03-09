# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa/56062555
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

class DataManager(object):
  def load(self):
    # Load dataset
    dataset_zip = np.load('data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                          encoding = 'latin1')

    # print('Keys in the dataset:', dataset_zip.keys())
    #  ['metadata', 'imgs', 'latents_classes', 'latents_values']

    self.imgs       = dataset_zip['imgs']
    latents_values  = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    metadata        = dataset_zip['metadata'][()]

    # Define number of values per latents and functions to convert to indices
    latents_sizes = metadata['latents_sizes']
    # [ 1,  3,  6, 40, 32, 32]
    # color, shape, scale, orientation, posX, posY

    # --- CUT ONLY ELLIPSES - BEGIN
    latents_sizes[1] = 1 # only one shape is there
    elipses_idxs = np.where(latents_values[:,1] == 2)[0] # "== 2" - ellipses
    self.imgs = self.imgs[elipses_idxs]
    latents_values = latents_values[elipses_idxs]
    # --- CUT ONLY ELLIPSES - END

    self.n_samples = latents_sizes[::-1].cumprod()[-1]
    # 737280

    self.latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                         np.array([1,])))

    print("latents_sizes: %s" % str(latents_sizes))
    print("n_samples: %s" % str(self.n_samples))
    print("latents_bases: %s" % str(self.latents_bases))
    # [737280, 245760, 40960, 1024, 32, 1]
    
  @property
  def sample_size(self):
    return self.n_samples

  def get_image(self, shape=0, scale=0, orientation=0, x=0, y=0):
    latents = [0, shape, scale, orientation, x, y]
    index = np.dot(latents, self.latents_bases).astype(int)
    return self.get_images([index])[0]

  def get_images(self, indices):
    images = []
    for index in indices:
      img = self.imgs[index]
      img = img.reshape(4096)
      images.append(img)
    return images

  def get_random_images(self, size):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    return self.get_images(indices)

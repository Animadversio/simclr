# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data preprocessing and augmentation."""

import functools
from absl import flags

import tensorflow.compat.v2 as tf
import tensorflow_addons as tfa
import math

FLAGS = flags.FLAGS

CROP_PROPORTION = 0.875  # Standard for ImageNet.


def random_apply(func, p, x):
  """Randomly apply function func to x with probability p."""
  return tf.cond(
      tf.less(
          tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
          tf.cast(p, tf.float32)), lambda: func(x), lambda: x)


def random_brightness(image, max_delta, impl='simclrv2'):
  """A multiplicative vs additive change of brightness."""
  if impl == 'simclrv2':
    factor = tf.random.uniform([], tf.maximum(1.0 - max_delta, 0),
                               1.0 + max_delta)
    image = image * factor
  elif impl == 'simclrv1':
    image = tf.image.random_brightness(image, max_delta=max_delta)
  else:
    raise ValueError('Unknown impl {} for random brightness.'.format(impl))
  return image


def to_grayscale(image, keep_channels=True):
  image = tf.image.rgb_to_grayscale(image)
  if keep_channels:
    image = tf.tile(image, [1, 1, 3])
  return image


def color_jitter(image, strength, random_order=True, impl='simclrv2'):
  """Distorts the color of the image.

  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.

  Returns:
    The distorted image tensor.
  """
  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength
  if random_order:
    return color_jitter_rand(
        image, brightness, contrast, saturation, hue, impl=impl)
  else:
    return color_jitter_nonrand(
        image, brightness, contrast, saturation, hue, impl=impl)


def color_jitter_nonrand(image,
                         brightness=0,
                         contrast=0,
                         saturation=0,
                         hue=0,
                         impl='simclrv2'):
  """Distorts the color of the image (jittering order is fixed).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.

  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x, brightness, contrast, saturation, hue):
      """Apply the i-th transformation."""
      if brightness != 0 and i == 0:
        x = random_brightness(x, max_delta=brightness, impl=impl)
      elif contrast != 0 and i == 1:
        x = tf.image.random_contrast(
            x, lower=1-contrast, upper=1+contrast)
      elif saturation != 0 and i == 2:
        x = tf.image.random_saturation(
            x, lower=1-saturation, upper=1+saturation)
      elif hue != 0:
        x = tf.image.random_hue(x, max_delta=hue)
      return x

    for i in range(4):
      image = apply_transform(i, image, brightness, contrast, saturation, hue)
      image = tf.clip_by_value(image, 0., 1.)
    return image


def color_jitter_rand(image,
                      brightness=0,
                      contrast=0,
                      saturation=0,
                      hue=0,
                      impl='simclrv2'):
  """Distorts the color of the image (jittering order is random).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.

  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x):
      """Apply the i-th transformation."""
      def brightness_foo():
        if brightness == 0:
          return x
        else:
          return random_brightness(x, max_delta=brightness, impl=impl)

      def contrast_foo():
        if contrast == 0:
          return x
        else:
          return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
      def saturation_foo():
        if saturation == 0:
          return x
        else:
          return tf.image.random_saturation(
              x, lower=1-saturation, upper=1+saturation)
      def hue_foo():
        if hue == 0:
          return x
        else:
          return tf.image.random_hue(x, max_delta=hue)
      x = tf.cond(tf.less(i, 2),
                  lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                  lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
      return x

    perm = tf.random.shuffle(tf.range(4))
    for i in range(4):
      image = apply_transform(perm[i], image)
      image = tf.clip_by_value(image, 0., 1.)
    return image


def _compute_crop_shape(
    image_height, image_width, aspect_ratio, crop_proportion):
  """Compute aspect ratio-preserving shape for central crop.

  The resulting shape retains `crop_proportion` along one side and a proportion
  less than or equal to `crop_proportion` along the other side.

  Args:
    image_height: Height of image to be cropped.
    image_width: Width of image to be cropped.
    aspect_ratio: Desired aspect ratio (width / height) of output.
    crop_proportion: Proportion of image to retain along the less-cropped side.

  Returns:
    crop_height: Height of image after cropping.
    crop_width: Width of image after cropping.
  """
  image_width_float = tf.cast(image_width, tf.float32)
  image_height_float = tf.cast(image_height, tf.float32)

  def _requested_aspect_ratio_wider_than_image():
    crop_height = tf.cast(
        tf.math.rint(crop_proportion / aspect_ratio * image_width_float),
        tf.int32)
    crop_width = tf.cast(
        tf.math.rint(crop_proportion * image_width_float), tf.int32)
    return crop_height, crop_width

  def _image_wider_than_requested_aspect_ratio():
    crop_height = tf.cast(
        tf.math.rint(crop_proportion * image_height_float), tf.int32)
    crop_width = tf.cast(
        tf.math.rint(crop_proportion * aspect_ratio * image_height_float),
        tf.int32)
    return crop_height, crop_width

  return tf.cond(
      aspect_ratio > image_width_float / image_height_float,
      _requested_aspect_ratio_wider_than_image,
      _image_wider_than_requested_aspect_ratio)


def center_crop(image, height, width, crop_proportion):
  """Crops to center of image and rescales to desired size.

  Args:
    image: Image Tensor to crop.
    height: Height of image to be cropped.
    width: Width of image to be cropped.
    crop_proportion: Proportion of image to retain along the less-cropped side.

  Returns:
    A `height` x `width` x channels Tensor holding a central crop of `image`.
  """
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]
  crop_height, crop_width = _compute_crop_shape(
      image_height, image_width, height / width, crop_proportion)
  offset_height = ((image_height - crop_height) + 1) // 2
  offset_width = ((image_width - crop_width) + 1) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_height, offset_width, crop_height, crop_width)

  image = tf.image.resize([image], [height, width],
                          method=tf.image.ResizeMethod.BICUBIC)[0]

  return image


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: `Tensor` of image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf.name_scope(scope or 'distorted_bounding_box_crop'):
    shape = tf.shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, target_height, target_width)

    return image


def crop_and_resize(image, height, width):
  """Make a random crop and resize it to height `height` and width `width`.

  Args:
    image: Tensor representing the image.
    height: Desired image height.
    width: Desired image width.

  Returns:
    A `height` x `width` x channels Tensor holding a random crop of `image`.
  """
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  aspect_ratio = width / height
  image = distorted_bounding_box_crop(
      image,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
      area_range=(0.08, 1.0),
      max_attempts=100,
      scope=None)
  return tf.image.resize([image], [height, width],
                         method=tf.image.ResizeMethod.BICUBIC)[0]


def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
  """Blurs the given image with separable convolution.


  Args:
    image: Tensor of shape [height, width, channels] and dtype float to blur.
    kernel_size: Integer Tensor for the size of the blur kernel. This is should
      be an odd number. If it is an even number, the actual kernel size will be
      size + 1.
    sigma: Sigma value for gaussian operator.
    padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.

  Returns:
    A Tensor representing the blurred image.
  """
  radius = tf.cast(kernel_size / 2, dtype=tf.int32)
  kernel_size = radius * 2 + 1
  x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
  blur_filter = tf.exp(-tf.pow(x, 2.0) /
                       (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0)))
  blur_filter /= tf.reduce_sum(blur_filter)
  # One vertical and one horizontal filter.
  blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
  blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
  num_channels = tf.shape(image)[-1]
  blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
  blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
  expand_batch_dim = image.shape.ndims == 3
  if expand_batch_dim:
    # Tensorflow requires batched input to convolutions, which we can fake with
    # an extra dimension.
    image = tf.expand_dims(image, axis=0)
  blurred = tf.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
  blurred = tf.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
  if expand_batch_dim:
    blurred = tf.squeeze(blurred, axis=0)
  return blurred


def random_crop_with_resize(image, height, width, p=1.0):
  """Randomly crop and resize an image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: Probability of applying this transformation.

  Returns:
    A preprocessed image `Tensor`.
  """
  def _transform(image):  # pylint: disable=missing-docstring
    image = crop_and_resize(image, height, width)
    return image
  return random_apply(_transform, p=p, x=image)


def random_color_jitter(image, p=1.0, strength=1.0,
                        impl='simclrv2'):

  def _transform(image):
    color_jitter_t = functools.partial(
        color_jitter, strength=strength, impl=impl)
    image = random_apply(color_jitter_t, p=0.8, x=image)
    return random_apply(to_grayscale, p=0.2, x=image)
  return random_apply(_transform, p=p, x=image)


def random_blur(image, height, width, p=1.0):
  """Randomly blur an image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: probability of applying this transformation.

  Returns:
    A preprocessed image `Tensor`.
  """
  del width
  def _transform(image):
    sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
    return gaussian_blur(
        image, kernel_size=height//10, sigma=sigma, padding='SAME')
  return random_apply(_transform, p=p, x=image)


def batch_random_blur(images_list, height, width, blur_probability=0.5):
  """Apply efficient batch data transformations.

  Args:
    images_list: a list of image tensors.
    height: the height of image.
    width: the width of image.
    blur_probability: the probaility to apply the blur operator.

  Returns:
    Preprocessed feature list.
  """
  def generate_selector(p, bsz):
    shape = [bsz, 1, 1, 1]
    selector = tf.cast(
        tf.less(tf.random.uniform(shape, 0, 1, dtype=tf.float32), p),
        tf.float32)
    return selector

  new_images_list = []
  for images in images_list:
    images_new = random_blur(images, height, width, p=1.)
    selector = generate_selector(blur_probability, tf.shape(images)[0])
    images = images_new * selector + images * (1 - selector)
    images = tf.clip_by_value(images, 0., 1.)
    new_images_list.append(images)

  return new_images_list


pi = tf.constant(math.pi)
@tf.function
def cosfunc(x):
  """The cosine square smoothing function"""
  Lower = tf.square(tf.cos(pi*(x + 1/4)));
  Upper = 1 - tf.square(tf.cos(pi*(x - 3/4)));
  # print(tf.logical_and((x <= -1/4), (x > -3/4)).dtype)
  fval = tf.where(tf.logical_and((x <= -1/4), (x >-3/4)), Lower, tf.zeros(1)) + \
      tf.where(tf.logical_and((x >= 1/4), (x <= 3/4)), Upper, tf.zeros(1)) + \
      tf.where(tf.logical_and((x < 1/4), (x > -1/4)), tf.ones(1), tf.zeros(1))
  return fval


@tf.function
def rbf(ecc, N, spacing, e_o=1.0):
  """ Number N radial basis function
  ecc: eccentricities, tf array.  
  N: numbering of basis function, starting from 0. 
  spacing: log scale spacing of ring radius (deg), scalar.
  e_o: radius of 0 string, scalar. 
  """
  spacing = tf.convert_to_tensor(spacing, dtype=tf.float32)
  e_o = tf.convert_to_tensor(e_o, dtype=tf.float32)
  preinput = tf.divide(tf.math.log(ecc) - (tf.math.log(e_o) + (tf.cast(N, tf.float32) + 1) * spacing), spacing)
  ecc_basis = cosfunc(preinput);
  return ecc_basis


@tf.function
def fov_rbf(ecc, spacing, e_o=1.0):
  """Initial radial basis function
  """
  spacing = tf.convert_to_tensor(spacing,dtype="float32")
  e_o = tf.convert_to_tensor(e_o,dtype="float32")
  preinput = tf.divide(tf.math.log(ecc) - tf.math.log(e_o), spacing)
  preinput = tf.clip_by_value(preinput, tf.zeros(1), tf.ones(1)) # only clip 0 is enough.
  ecc_basis = cosfunc(preinput);
  return ecc_basis


# def random_foveation(img, height, width, 
  #                   kerW_coef=0.04, 
  #                   e_o=1, 
  #                   N_e=None, 
  #                   spacing=0.3, 
  #                   deg_per_pix = 0.03,
  #                   bdr=12):
  # """Randomly apply `pntN` foveation transform to `img`. points are sampled uniformly in the center of 
  # image after masking out the border `bdr` pixels.

  # Args: 
  #   kerW_coef: how gaussian filtering kernel std scale as a function of eccentricity 
  #   e_o: eccentricity of the initial ring belt
  #   spacing: log scale spacing between eccentricity of ring belts. 
  #   N_e: Number of ring belts in total. if None, it will calculate the N_e s.t. the whole image is covered by ring belts.
  #   bdr: width (in pixel) of border region that forbid sampling (bias foveation point to be in the center of img)
  # """
  # # H, W = img.shape[0], img.shape[1] # if this is fixed then these two steps could be saved
  # H, W = height, width
  # XX, YY = tf.meshgrid(tf.range(W),tf.range(H))
  # # deg_per_pix = 20/math.sqrt(H**2+W**2); # FIXME! degree to pixel transforms 
  # # xids = tf.random.uniform(shape=[1,], minval=bdr, maxval=W-bdr, dtype=tf.int32) # TF version
  # # yids = tf.random.uniform(shape=[1,], minval=bdr, maxval=H-bdr, dtype=tf.int32)
  # # xid, yid = xids[0], yids[0] # pixel coordinate of fixation point.
  # xid = random.randint(bdr, W-bdr)
  # yid = random.randint(bdr, H-bdr)
  # D2fov = tf.sqrt(tf.cast(tf.square(XX - xid) + tf.square(YY - yid), 'float32'))
  # D2fov_deg = D2fov * deg_per_pix
  # # maxecc = 10 $ fixed version
  # maxecc = math.sqrt(max(xid, W-xid)**2 + max(yid, H-yid)**2) * deg_per_pix # none tensor version
  # # maxecc = tf.sqrt(tf.cast(tf.square(tf.maximum(xid, W-xid)) + tf.square(tf.maximum(yid, H-yid)),tf.float32)) * deg_per_pix
  # e_r = maxecc; # 15
  # if N_e is None:
  #   N_e = int(math.ceil((math.log(maxecc)-math.log(e_o))/spacing)) #.astype("int32"
  #   # N_e = tf.cast(tf.math.ceil(\
  #   #     (tf.math.log(maxecc)-tf.math.log(tf.convert_to_tensor(e_o,dtype=tf.float32)))/spacing),tf.int32) # this is problematic
  # # spacing = tf.convert_to_tensor((math.log(e_r) - math.log(e_o)) / N_e);
  # # spacing = tf.convert_to_tensor(spacing, dtype="float32");
  # # e_o = tf.convert_to_tensor(e_o, dtype="float32");
  # rbf_basis = fov_rbf(D2fov_deg,spacing,e_o)
  # finalimg = tf.expand_dims(rbf_basis, -1)*img
  # for N in range(N_e):
  #   rbf_basis = rbf(D2fov_deg, N, spacing, e_o=e_o)
  #   mean_dev = math.exp(math.log(e_o) + (N + 1) * spacing)
  #   # mean_dev = tf.exp(tf.math.log(e_o) + (tf.cast(N, tf.float32) + 1) * spacing)
  #   kerW = kerW_coef * mean_dev / deg_per_pix
  #   kerSz = int(kerW * 3)
  #   # kerSz = tf.cast(kerW * 3, tf.int32)
  #   img_gsft = tfa.image.gaussian_filter2d(img, filter_shape=(kerSz, kerSz), sigma=kerW, padding='REFLECT')
  #   finalimg = finalimg + tf.expand_dims(rbf_basis, -1)*img_gsft
  # return finalimg


# def random_foveation_multiple(img, 
#                     pntN:int =1, 
#                     kerW_coef=0.04, 
#                     e_o=1, 
#                     N_e=None, 
#                     spacing=0.3, 
#                     bdr=12):
#   """Randomly apply `pntN` foveation transform to `img`. points are sampled uniformly in the center of 
#   image after masking out the border `bdr` pixels.

#   Args: 
#     kerW_coef: how gaussian filtering kernel std scale as a function of eccentricity 
#     e_o: eccentricity of the initial ring belt
#     spacing: log scale spacing between eccentricity of ring belts. 
#     N_e: Number of ring belts in total. if None, it will calculate the N_e s.t. the whole image is covered by ring belts.
#     bdr: width (in pixel) of border region that forbid sampling (bias foveation point to be in the center of img)
#   """
#   H, W = img.shape[0], img.shape[1] # if this is fixed then these two steps could be saved
#   XX, YY = tf.meshgrid(tf.range(W),tf.range(H))
#   deg_per_pix = 0.03 #20/math.sqrt(H**2+W**2); # FIXME! degree to pixel transforms 
#   finimg_list = []
#   xids = tf.random.uniform(shape=[pntN,], minval=bdr, maxval=W-bdr, dtype=tf.int32)
#   yids = tf.random.uniform(shape=[pntN,], minval=bdr, maxval=H-bdr, dtype=tf.int32)
#   for it in range(pntN):
#     xid, yid = xids[it], yids[it] # pixel coordinate of fixation point.
#     D2fov = tf.sqrt(tf.cast(tf.square(XX - xid) + tf.square(YY - yid), 'float32'))
#     D2fov_deg = D2fov * deg_per_pix
#     # maxecc = max(D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1])
#     # maxecc = max(D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1]) # maximal deviation at 4 corner
#     # maxecc = tf.reduce_max([D2fov_deg[0,0], D2fov_deg[-1,0], D2fov_deg[0,-1], D2fov_deg[-1,-1]]).eval() # just cannot get this work
#     maxecc = 10
#     e_r = maxecc; # 15
#     if N_e is None:
#       N_e = int(math.ceil((math.log(maxecc)-math.log(e_o))/spacing+1)) #.astype("int32"
#       # N_e = tf.cast(tf.math.ceil((tf.math.log(maxecc)-tf.math.log(e_o))/spacing+1),tf.int32) # this is problematic
#     # spacing = tf.convert_to_tensor((math.log(e_r) - math.log(e_o)) / N_e);
#     # spacing = tf.convert_to_tensor(spacing, dtype="float32");
#     rbf_basis = fov_rbf(D2fov_deg,spacing,e_o)
#     finalimg = tf.expand_dims(rbf_basis, -1)*img
#     for N in range(N_e):
#       rbf_basis = rbf(D2fov_deg, N, spacing, e_o=e_o)
#       mean_dev = math.exp(math.log(e_o) + (N + 1) * spacing)
#       kerW = kerW_coef * mean_dev / deg_per_pix
#       kerSz = int(kerW * 3)
#       img_gsft = tfa.image.gaussian_filter2d(img, filter_shape=(kerSz, kerSz), sigma=kerW, padding='REFLECT')
#       finalimg = finalimg + tf.expand_dims(rbf_basis, -1)*img_gsft
#     finimg_list.append(finalimg)
#   finimgs = tf.stack(finimg_list)
#   return finimgs


def FoveateAt(img, height:int, width:int, 
              pnt:tuple, 
              e_o=1, 
              spacing=0.3,
              kerW_coef=0.04, 
              N_e=None, 
              deg_per_pix=0.03,):
  """Apply foveation transform at (x,y) coordinate `pnt` to `img`

  Args: 
    kerW_coef: how gaussian filtering kernel std scale as a function of eccentricity 
    e_o: eccentricity of the initial ring belt
    spacing: log scale spacing between eccentricity of ring belts. 
    N_e: Number of ring belts in total. if None, it will calculate the N_e s.t. the whole image is covered by ring belts.
    bdr: width (in pixel) of border region that forbid sampling (bias foveation point to be in the center of img)
  """
  H, W = height, width
  xid, yid = pnt[0], pnt[1]
  XX, YY = tf.meshgrid(tf.range(W),tf.range(H))
  # deg_per_pix = 20/math.sqrt(H**2+W**2); # FIXME! degree to pixel transforms 
  D2fov = tf.sqrt(tf.cast(tf.square(XX - xid) + tf.square(YY - yid), 'float32'))
  D2fov_deg = D2fov * deg_per_pix
  maxecc = math.sqrt(max(xid, W-xid)**2 + max(yid, H-yid)**2) * deg_per_pix # none tensor version
  if N_e is None:
    N_e = int(math.ceil((math.log(maxecc)-math.log(e_o))/spacing)) 
  rbf_basis = fov_rbf(D2fov_deg,spacing,e_o)
  finalimg = tf.expand_dims(rbf_basis, -1)*img
  for N in range(N_e):
    rbf_basis = rbf(D2fov_deg, N, spacing, e_o=e_o)
    mean_dev = (math.log(e_o) + (N + 1) * spacing) # math.exp
    # mean_dev = tf.exp(tf.math.log(e_o) + (tf.cast(N, tf.float32) + 1) * spacing)
    kerW = kerW_coef * mean_dev / deg_per_pix
    kerSz = int(kerW * 3)
    # kerSz = tf.cast(kerW * 3, tf.int32)
    img_gsft = tfa.image.gaussian_filter2d(img, filter_shape=(kerSz, kerSz), sigma=kerW, padding='REFLECT')
    finalimg = finalimg + tf.expand_dims(rbf_basis, -1)*img_gsft
  return finalimg 


import random
def random_foveation(img, height, width, bdr=12, 
                    fov_area_ratio=0.1, kerW_coef=0.04, 
                    e_o=None, 
                    N_e=None, 
                    spacing=0.3,
                    deg_per_pix=0.03,):
  """Randomly apply `pntN` foveation transform to `img`. points are sampled uniformly in the center of 
  image after masking out the border `bdr` pixels.

  Args: 
    fov_area_ratio: tuple of range of area ratio; one float of ratio or None.
    kerW_coef: how gaussian filtering kernel std scale as a function of eccentricity 
    e_o: eccentricity of the initial ring belt
    spacing: log scale spacing between eccentricity of ring belts. 
    N_e: Number of ring belts in total. if None, it will calculate the N_e s.t. the whole image is covered by ring belts.
    bdr: width (in pixel) of border region that forbid sampling (bias foveation point to be in the center of img)
  """
  # H, W = img.shape[0], img.shape[1] # if this is fixed then these two steps could be saved
  H, W = height, width
  xid = random.randint(bdr, W-bdr)
  yid = random.randint(bdr, H-bdr)
  if fov_area_ratio is not None:
    if type(fov_area_ratio) in [tuple, list]:
      fov_area_r = random.uniform(fov_area_ratio[0], fov_area_ratio[1]) # should this ratio be uniform? 
    else: 
      fov_area_r = fov_area_ratio
    fov_rad = max(1,math.sqrt(H * W * fov_area_r / math.pi))
    fov_rad_deg = fov_rad * deg_per_pix
  else: 
    if e_o is not None:
      fov_rad_deg = e_o
    else:
      raise ValueError

  finalimg = FoveateAt(img, height, width, 
              pnt=(xid, yid), 
              e_o=fov_rad_deg, kerW_coef=kerW_coef, N_e=None, 
              spacing=spacing, deg_per_pix=deg_per_pix,)
  return finalimg


def preprocess_for_train(image,
                         height,
                         width,
                         foveation=True,
                         color_distort=True,
                         crop=True,
                         flip=True,
                         impl='simclrv2'):
  """Preprocesses the given image for training.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    foveation: Random foveation to the image. 
    color_distort: Whether to apply the color distortion.
    crop: Whether to crop the image.
    flip: Whether or not to flip left and right of an image.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.

  Returns:
    A preprocessed image `Tensor`.
  """
  if crop:
    image = random_crop_with_resize(image, height, width)
  if flip:
    image = tf.image.random_flip_left_right(image)
  if color_distort:
    image = random_color_jitter(image, strength=FLAGS.color_jitter_strength,
                                impl=impl)
  image = tf.reshape(image, [height, width, 3]) # this is single image augmentation
  if foveation:
    image = random_foveation(image, height, width, \
        kerW_coef=FLAGS.blur_scaling, fov_area_ratio=FLAGS.fov_area_range)
  image = tf.clip_by_value(image, 0., 1.)
  return image


def preprocess_for_eval(image, height, width, crop=True):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    crop: Whether or not to (center) crop the test images.

  Returns:
    A preprocessed image `Tensor`.
  """
  if crop:
    image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
  image = tf.reshape(image, [height, width, 3]) # this is single image augmentation
  image = tf.clip_by_value(image, 0., 1.)
  return image


def preprocess_image(image, height, width, is_training=False,
                     color_distort=True, test_crop=True):
  """Preprocesses the given image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    is_training: `bool` for whether the preprocessing is for training.
    color_distort: whether to apply the color distortion.
    test_crop: whether or not to extract a central crop of the images
        (as for standard ImageNet evaluation) during the evaluation.

  Returns:
    A preprocessed image `Tensor` of range [0, 1].
  """
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if is_training:
    return preprocess_for_train(image, height, width, color_distort, FLAGS.foveation)
  else:
    return preprocess_for_eval(image, height, width, test_crop)

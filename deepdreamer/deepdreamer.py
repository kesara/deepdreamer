###############################################################################
# Deep Dreamer
# Based on https://github.com/google/deepdream/blob/master/dream.ipynb
# Author: Kesara Rathnayake ( kesara [at] kesara [dot] lk )
###############################################################################

import numpy as np
from caffe import Classifier
from scipy.ndimage import affine_transform, zoom
from PIL.Image import fromarray as img_fromarray, open as img_open

NET_FN = "deploy.prototxt"  # Make sure force_backward: true
PARAM_FN = "bvlc_googlenet.caffemodel"
CHANNEL_SWAP = (2, 1, 0)
# ImageNet mean, training set dependent
CAFFE_MEAN = np.float32([104.0, 116.0, 122.0])


def _preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean["data"]


def _deprocess(net, img):
    return np.dstack((img + net.transformer.mean["data"])[::-1])


def _make_step(
        net, step_size=1.5, end="inception_4c/output", jitter=32, clip=True):
    """ Basic gradient ascent step. """

    src = net.blobs["data"]
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)

    # apply jitter shift
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)

    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g
    # unshift image
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)

    if clip:
        bias = net.transformer.mean["data"]
        src.data[:] = np.clip(src.data, -bias, 255-bias)


def _deepdream(
        net, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
        end="inception_4c/output", clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [_preprocess(net, base_img)]

    for i in xrange(octave_n-1):
        octaves.append(zoom(
            octaves[-1], (1, 1.0/octave_scale, 1.0/octave_scale), order=1))

    src = net.blobs["data"]

    # allocate image for network-produced details
    detail = np.zeros_like(octaves[-1])

    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = zoom(detail, (1, 1.0*h/h1, 1.0*w/w1), order=1)

        src.reshape(1, 3, h, w)  # resize the network's input image size
        src.data[0] = octave_base+detail

        for i in xrange(iter_n):
            _make_step(net, end=end, clip=clip, **step_params)

            # visualization
            vis = _deprocess(net, src.data[0])
            if not clip:  # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))

        # extract details produced on the current octave
        detail = src.data[0]-octave_base

    # returning the resulting image
    return _deprocess(net, src.data[0])


def deepdream(
        img_path, scale_coefficient=0.05, irange=100, iter_n=10, octave_n=4,
        octave_scale=1.4, end="inception_4c/output", clip=True):
    img = np.float32(img_open(img_path))
    s = scale_coefficient
    h, w = img.shape[:2]

    # Load DNN model
    net = Classifier(
        NET_FN, PARAM_FN, mean=CAFFE_MEAN, channel_swap=CHANNEL_SWAP)

    print("Dreaming...")
    for i in xrange(irange):
        img = _deepdream(
            net, img, iter_n=iter_n, octave_n=octave_n,
            octave_scale=octave_scale, end=end, clip=clip)
        img_fromarray(np.uint8(img)).save("{}_{}.jpg".format(
            img_path, i))
        print("Dream {} saved.".format(i))
        img = affine_transform(
            img, [1-s, 1-s, 1], [h*s/2, w*s/2, 0], order=1)

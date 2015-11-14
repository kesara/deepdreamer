# Deep Dreamer
Easy to configure Python program that make use of [Google's DeepDream](https://github.com/google/deepdream/)

* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Configuration options](#configuration-options)
* [Examples](#examples)

## Requirements
* Python 2.7
* [NumPy](https://pypi.python.org/pypi/numpy)
* [SciPy](https://pypi.python.org/pypi/scipy/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)
* [Caffe](http://caffe.berkeleyvision.org/)
* [FFmpeg](https://www.ffmpeg.org/) (Optional, required for videos.)

## Installation
1. Install [NumPy](https://pypi.python.org/pypi/numpy), [SciPy](https://pypi.python.org/pypi/scipy/), [Pillow](https://pypi.python.org/pypi/Pillow/) and [Caffe](http://caffe.berkeleyvision.org/). *NumPy, SciPy & Pillow can be installed via PIP.*
2. Download **deploy.prototxt** from [bvlc_googlenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet).
3. Add line `force_backward: true` to **deploy.prototxt** file.
4. Download **bvlc_googlenet.caffemodel** from [bvlc_googlenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet).
5. (Optional) If instead you want to incept using MIT's "Places" neural net, download the **Places205-GoogLeNet** from [their website](http://places.csail.mit.edu/downloadCNN.html). You need the **deploy_places205.protxt** and **googlelet_places205_train_iter_2400000.caffemodel** files from the archive.
6. Make sure the files are in the root directory of DeepDreamer.

## Usage
* Just deep dreaming
`python deepdreamer.py image.jpg`
* Create a deepdream gif
`python deepdreamer.py --gif true image.jpg`
* Create a deepdream video (requires ffmpeg)
`python deepdreamer.py --video video.mp4`

## Configuration options
```
usage: deepdreamer.py [-h] [--zoom {true,false}] [--scale SCALE]
                      [--dreams DREAMS] [--itern ITERN] [--octaves OCTAVES]
                      [--octave-scale OCTAVE_SCALE] [--layers LAYERS]
                      [--clip {true,false}]
                      [--network {bvlc_googlenet,googlenet_place205}]
                      [--gif {true,false}] [--reverse {true,false}]
                      [--duration DURATION] [--loop {true,false}]
                      [--framerate FRAMERATE] [--list-layers] [--video VIDEO]
                      [image]

positional arguments:
  image

optional arguments:
  -h, --help            show this help message and exit
  --zoom {true,false}   zoom dreams (default: true)
  --scale SCALE         scale coefficient for zoom (default: 0.05)
  --dreams DREAMS       number of images (default: 100)
  --itern ITERN         dream iterations (default: 10)
  --octaves OCTAVES     dream octaves (default: 4)
  --octave-scale OCTAVE_SCALE
                        dream octave scale (default: 1.4)
  --layers LAYERS       dream layers (default: inception_4c/output)
  --clip {true,false}   clip dreams (default: true)
  --network {bvlc_googlenet,googlenet_place205}
                        choose the network to use (default: bvlc_googlenet)
  --gif {true,false}    make a gif (default: false)
  --reverse {true,false}
                        make a reverse gif (default: false)
  --duration DURATION   gif frame duration in seconds (default: 0.1)
  --loop {true,false}   enable gif loop (default: false)
  --framerate FRAMERATE
                        framerate for video (default: 24)
  --list-layers         list layers
  --video VIDEO         video file
```

## Examples
Original Image|Deepdream
--------|---------
![Original](https://farm8.staticflickr.com/7233/7167040599_cf7c835c77_z_d.jpg)|![Deepdream](http://i.imgur.com/Auikelk.jpg)
From complete black (#000000)|![Deepdream](http://i.imgur.com/Ox1B8wf.gif)
From complete black (#000000)|![Deepdream](http://i.imgur.com/llUZ7Ll.gif)
![Original](https://farm9.staticflickr.com/8084/8361122341_183dd4a7e3_z_d.jpg)|![Deepdream](http://i.imgur.com/1YpPJVK.jpg)
![Original](https://farm8.staticflickr.com/7016/6736252139_e979e45b8c_z_d.jpg)|![Deepdream](http://i.imgur.com/E8cO7zk.jpg)

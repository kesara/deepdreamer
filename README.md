# [Deep Dreamer](https://deepdreamer.fq.nz/)
Easy to configure Python program that make use of [Google's DeepDream](https://github.com/google/deepdream/)

* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Configuration options](#configuration-options)
* [Examples](#examples)

## Requirements
* Python 3
* [NumPy](https://pypi.python.org/pypi/numpy)
* [SciPy](https://pypi.python.org/pypi/scipy/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)
* [Caffe](http://caffe.berkeleyvision.org/)
* [FFmpeg](https://www.ffmpeg.org/) (Optional, required for videos.)

## Installation
1. Install [NumPy](https://pypi.python.org/pypi/numpy), [SciPy](https://pypi.python.org/pypi/scipy/), [Pillow](https://pypi.python.org/pypi/Pillow/) and [Caffe](http://caffe.berkeleyvision.org/). *On Ubuntu 17.10 installing caffe will usually install all other dependencies.*
2. Clone this project. `git clone https://github.com/kesara/deepdreamer.git`
3. Go to project directory. `cd deepdeamer`
4. Download **deploy.prototxt** from [bvlc_googlenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) into the project directory.
5. Add line `force_backward: true` to **deploy.prototxt** file.
6. Download **bvlc_googlenet.caffemodel** from [bvlc_googlenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) into the project directory.
7. (Optional) Download MIT's "Places" neural net, download the **Places205-GoogLeNet** from [their website](http://places.csail.mit.edu/downloadCNN.html). You need the **deploy_places205.protxt** and **googlelet_places205_train_iter_2400000.caffemodel** files from the archive.

## Usage
* Just deep dreaming
`python3 deepdreamer.py image.jpg`
* Create a deepdream gif
`python3 deepdreamer.py --gif true image.jpg`
* Create a deepdream video (requires ffmpeg)
`python3 deepdreamer.py --video video.mp4`

## Configuration options
```
usage: deepdreamer.py [-h] [--zoom {true,false}] [--scale SCALE]
                      [--dreams DREAMS] [--itern ITERN] [--octaves OCTAVES]
                      [--octave-scale OCTAVE_SCALE] [--layers LAYERS]
                      [--clip {true,false}] [--gpuid GPUID]
                      [--network {bvlc_googlenet,googlenet_place205}]
                      [--gif {true,false}] [--reverse {true,false}]
                      [--duration DURATION] [--loop {true,false}]
                      [--framerate FRAMERATE] [--list-layers] [--video VIDEO]
                      [image]

positional arguments:
  image

optional arguments:
  -h, --help            show this help message and exit
  --gpuid GPUID         enable GPU with id GPUID (default: disabled)
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
![Deepdream](https://i.imgur.com/Auikelk.jpg)
![Deepdream](https://i.imgur.com/Ox1B8wf.gif)
![Deepdream](https://i.imgur.com/llUZ7Ll.gif)
![Deepdream](https://i.imgur.com/41GVLNC.gif)

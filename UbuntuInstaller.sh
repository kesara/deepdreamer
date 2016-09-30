#clone the repo
git clone https://github.com/kesara/deepdreamer.git deepdreamer
cd deepdreamer

#install basic deps
sudo apt-get install build-essential python-dev python-setuptools
#install PIP
sudo easy_install pip
#install numpy scipy Pillow

sudo -H pip install numpy scipy Pillow

#TODO install Caffe

#download the prototxt file
wget https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt

#add the line
echo "force_backward: true">>deploy.prototxt

#download the caffemodel
wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
                                   

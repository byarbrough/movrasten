# movrasten
Train a Tensorflow Keras Deep Neural Network model and conduct inference on a an edge device (Raspberry Pi) with hardware acceleration from the Intel Nueral Compute Stick 2. How's that for buzzword bingo!

## Installation
This is a little bit complicated becasue there are several things that need to be installed, potentially on multiple devices.
*TODO: add diagram*

## movrasten
Get the code
```
git clone https://github.com/byarbrough/movrasten.git && cd movrasten
```
Install python requiremets
```
pip install -r requirements.txt
```
This may require `sudo` and `--user` options.

## OpenVINO Installation
[Intel's OpenVINO Toolkit](https://software.intel.com/en-us/openvino-toolkit) is s unified AI framework for computer vision designed to work with with CPU, GPU, NCS, and FPGA with a single API. Neat!

Depending on the version of the neural compute stick in use, the installation changes. I tested on NCS 2 which requires OpenVINO, *not* the sdk.
According to [Intel Support](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html) the version can be determined with 
```
lsusb | grep 03e7
```
Where `2150` corresponds to version 1 and `2485` means you are using NCS 2.

This guide follows [Get Started with Intel NCS 2](https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick)

#### Desktop
1. Download OpenVINO from https://software.intel.com/en-us/openvino-toolkit/choose-download
2. Follow the instructions to install it. [For Linux:](https://docs.openvinotoolkit.org/2019_R2/_docs_install_guides_installing_openvino_linux.html)
```
cd ~/Downloads
tar xvf l_openvino_toolkit_<VERSION>.tgz
cd l_openvino_toolkit_<VERSION>
sudo -E ./install_openvino_dependencies.sh
./install_GUI.sh
# Follow the GUI to install
```
Install dependencies
```
cd /opt/intel/openvino/install_dependencies
sudo -E ./install_openvino_dependencies.sh
```
For all others see [Intel's getting started](https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick#inpage-nav-2)
3. Set the enviornment variable
```
source /opt/intel/openvino/bin/setupvars.sh
```
This will need to be done every time a terminal is opened. I like to create a symbolic link within movrasten or its parent directory `ln -s  /opt/intel/openvino/bin/setupvars.sh env-ncs` so that I can call `source env-ncs` and don't have to remember where `setupvars.sh` is.

Alternatively, `.bashrc` can be modified to include that line.

4. More things to install
USB Driver
```
cd /opt/intel/openvino/install_dependencies
./install_NCS_udev_rules.sh
```
Prerequisites
```
cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/
```
This can be done in a virtual environment with the optional venv tag.
*TODO: verify if this is the right way or if it should be install_prerequisites_tf.sh*
```
./install_prerequisites.sh venv tf
```
5. Demo!
Make sure the NCS is plugged in!
```
cd /opt/intel/openvino/deployment_tools/demo
./demo_squeezenet_download_convert_run.sh -d MYRIAD
```
You should get a great *Demo completed successfully.*

#### Raspberry Pi Installation
_NOTE: Inference on the Pi doesn't work yet._
For NCS version 1, I suspect that following the SDK basic installation at https://movidius.github.io/ncsdk/install.html will work, being sure to use `make install api` since the full sdk ought not to go on the Pi.

For version 2, use https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html

Apparently $PYTHONPATH has a bug so needs this added to it
`${PYTHONPATH}:/opt/intel/openvino/python/python3.5/armv7l/`
https://software.intel.com/en-us/forums/computer-vision/topic/807560#comment-1937618%20%20Attachments

## Run
1. Train the model
2. Freeze the model
3. Convert the model to OpenVINO format
4. Run inference on the model

### Train the model
A model can be trained with `basic/tr_image.py`
```
cd basic
python tr_image.py <path to training directory>
```
This will output a .h5 file in the Keras format and a frozen .pb file.

### Freeze the model
If you trained with the provided file then you already have a frozen .pb file. If you are using a pretrained model then consider [freezing it]()

### Convert to OpenVINO Format
This requries [prerequisites to be installed](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Config_Model_Optimizer.html)

[Convert a frozen Tensorflow Model](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html#loading-nonfrozen-models)
```
 python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model model.pb -b 1 --data_type FP16 --scale 255 --reverse_input_channels
```
`-b` is the batch size. The rescale must match whatever rescale was done in training.
Input channels are reversed (bgr) and that caused me a lot of suffering until I saw it somewhere in the docs!!!

### Conduct Inference
Then inference can happen on your Ubuntu machie with one of the samples
```
python /opt/intel/openvino/deployment_tools/inference_engine/samples/python_samples/classification_sample/classification_sample.py  -m inference_graph.xml -nt 2 -i <path to test image> -d MYRIAD
```
Inference should work with MYRIAD or CPU for the `-d` option
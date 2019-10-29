<img src="https://images.hellogiggles.com/uploads/2017/03/13095105/Rasputin_2.png" width="398" height="230">

# Movrasten
*Mo*idius *Ras*berry Pi and *Ten*sorflow
Train a Tensorflow Keras Deep Neural Network model and conduct inference on a an edge device (Raspberry Pi) with hardware acceleration from the Intel Nueral Compute Stick 2. How's that for buzzword bingo!

# Run with Docker
By using Docker, training and conversion of models can be done with simple `make` commands.

### Setup
Download the repository
```
git clone https://github.com/byarbrough/movrasten.git && cd movrasten
```
Place your images in to the `data/train` folder. Each class of images should be in its own folder. For example, `tree` on a three-class dataset should yield.
```
data
├── test
│   ├── 01
│   ├── 02
│   └── 03
└── train
    ├── 01
    ├── 02
    └── 03
```
Makefile does not yet implement automated testing, so it is fine if `data/test` is empty.

### Run
To build the Docker image, train a model, and convert that model to a 32-bit OpenVINO format, simply call
```
make all
```
These stages can also be run independently. To show all options use:
```
make help
```
Keep in mind that an image must first be running (`make run`) before it can be used for training or conversion.

The Keras models are saved in `models/` as both `.h5` and `.pb`. The models in OpenVINO format are saved in `modesl/openvino`. If using `make convert_16` for inference on the Raspberry Pi, make sure to copy all three files (`bin`, `.mapping`, `.xml`) to the Pi.

## Raspberry Pi Installation
Minimum version `2019.2.242`. The latest version can be found at the [Intel® Open Source Technology Center](https://download.01.org/opencv/2019/openvinotoolkit/)
Follow [Install OpenVINO™ toolkit for Raspbian* OS](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html)

Or use these consolidated steps
```
cd ~/Downloads/
wget https://download.01.org/opencv/2019/openvinotoolkit/R2/l_openvino_toolkit_runtime_raspbian_p_2019.2.242.tgz
sudo mkdir -p /opt/intel/openvino
sudo tar -xf l_openvino_toolkit_runtime_raspbian_p_2019.2.242.tgz --strip 1 -C /opt/intel/openvino
sudo apt install cmake
source /opt/intel/openvino/bin/setupvars.sh
echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc
```
You should see the output
```
[setupvars.sh] OpenVINO environment initialized
```
Then setup USB rules
```
sudo usermod -a -G users "$(whoami)"
```
Log out and back in
```
sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh
```

At this point, the following should execute cleanly:
```
python3 -c "from openvino.inference_engine import IENetwork, IEPlugin"
```

### Conduct Inference
Once the files have been copeid from `models/openvino` to the Raspberry Pi, conduct inference on a single image with:
```
python infer/classification_sample.py -m <path to model>.xml -nt 5 -i <path to test image> -d MYRIAD
```

# Manual Installation
This is a little bit complicated becasue there are several things that need to be installed, potentially on multiple devices.

## Download
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
1. Download OpenVINO from https://software.intel.com/en-us/openvino-toolkit/choose-download. This has been tested with 2019.3.376
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
Inference requires both the `.bin` and `.xml` files to be in the same directory.
```
python infer/classification_sample.py -m <path to model>.xml -nt 5 -i <path to test image> -d MYRIAD
```
On the desktop inference should work with MYRIAD or CPU for the `-d` option

#### Raspbery Pi
Copy *both* the `.bin` and `.xml` files from the desktop to the Pi. Then the same inference command will work!
On the Raspberry Pi inference will only work for MYRIAD.

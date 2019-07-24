# movrasten
Demo of Movidius NCS on Raspberry Pi using Tensor Flow

## Installation
1. Install Intel Movidius Neural Compute Stick 2 SDK
For using sdk inside a viraulenv: prior to running `make install`,  open `ncsdk/ncsdk.conf` for editing and set _USE_VIRTUALENV=yes_ (by default it will say _USE_VIRTUALENV=no_).
```
git clone -b ncsdk2 http://github.com/Movidius/ncsdk && cd ncsdk && make install
```
Activate the virtualenv with 
```
source /opt/movidius/virtualenv-python/bin/activate
```
More information can be found in the [NCSDK Documentation](https://movidius.github.io/ncsdk/index.html).

Update, for NCS 2 things are different and this doesn't work because it cannot open the hardware. Will try this getting started guide next.
https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick

2. Download movrasten files
May want to change directories ahead of time to seperate movrasten from your NCS install.
```
git clone https://github.com/byarbrough/movrasten.git && cd movrasten
```

3. Install python requiremets
```
pip install -r requirements.txt
```
This may require `sudo` and `--user` options.

4. Get to training!

### Raspberry Pi Installation
Depending on the version of the neural compute stick in use, the installation changes. I tested on NCS 2 which requires OpenVINO, *not* the sdk.
According to [Intel Support](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html) the version can be determined with 
```
lsusb | grep 03e7
```
Where `2150` corresponds to version 1 and `2485` means you are using NCS 2.

For version 1, I suspect that following the SDK basic installation at https://movidius.github.io/ncsdk/install.html will work, being sure to use `make install api` since the full sdk ought not to go on the Pi.
For version 2, use https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html



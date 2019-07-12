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

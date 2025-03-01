#!/usr/bin/env bash


# ====================================================
# import the bash utils 
. bash_utils.sh 

# ====================================================


print_blue '================================================'
print_blue "Configuring and installing python packages ..."


pip3 install --upgrade pip

# pip3 packages 
install_pip_packages pygame numpy matplotlib pyopengl Pillow pybind11 
install_pip_package numpy>=1.18.3
install_pip_package scipy==1.4.1
install_pip_package scikit-image==0.16.2
install_pip_packages pyyaml termcolor tqdm yacs
install_pip_package opencv-python  

install_pip_package opencv-contrib-python #==3.4.2.16 

install_pip_package torch 
install_pip_package torchvision
install_pip_package ordered-set # from https://pypi.org/project/ordered-set/
install_pip_package tensorflow-gpu==1.14.0  # 1.14.0 works with all the modules contained in pyslam2


 

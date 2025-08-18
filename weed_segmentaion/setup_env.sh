#!/bin/bash
set -e 

# Create enviornment with Python specific version - 3.7. here alongside with necessary packages
conda create -y -n rbt_env python=3.9 numpy opencv pip matplotlib

# Activate the enviornment 
source /home/biosense-ms/psi4conda/etc/profile.d/conda.sh
conda activate rbt_env

# Install depthai 
pip install depthai==2.22.0.0

echo "Virtual environment created and ready to use."

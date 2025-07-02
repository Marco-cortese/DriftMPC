#!/bin/bash

# This script installs acados following the instructions from the official acados documentation.

echo "Installing acados..."
if [ -d "acados" ]; then
    echo "Removing existing acados directory..."
    rm -rf acados
fi

# Clone the acados repository
git clone https://github.com/acados/acados.git


cd acados
ACADOS_ROOT=$(pwd)
echo -e "\n\n acados root directory: $ACADOS_ROOT \n\n"

# Checkout the latest stable version
git submodule update --recursive --init

# Create a build directory
if [ -d build ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi
mkdir -p build
cd build

# Configure the build
cmake \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release \
    -DACADOS_WITH_QPOASES=ON \
    -DACADOS_WITH_DAQP=ON \
    -DACADOS_WITH_PYTHON=ON \
    ..
# Build acados
make install -j$(nproc)

cd $ACADOS_ROOT # return to the root directory of acados

# fix the dynamic library loading on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Fixing dynamic library loading on macOS..."
    install_name_tool -add_rpath $ACADOS_ROOT/lib $ACADOS_ROOT/lib/libacados.dylib
fi

## python interface

# # activate the virtual environment
# if [ -d aca ]; then
#     echo "Removing existing virtual environment 'aca'..."
#     rm -rf aca
# fi
# python -m venv aca
# source aca/bin/activate

source ~/ml/bin/activate
echo -e "Installing Python interface for acados...\n\n\n"
which python # check if the correct python is used
echo -e "Python version: $(python --version), Python executable: $(which python)\n\n\n\n"

pip install -e $ACADOS_ROOT/interfaces/acados_template


# Add the path to the compiled shared libraries libacados.so, libblasfeo.so, libhpipm.so to
# LD_LIBRARY_PATH (default path is <acados_root/lib>) by running: 
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:"$ACADOS_ROOT/lib"
export ACADOS_SOURCE_DIR="$ACADOS_ROOT"



# test the installation
cd $ACADOS_ROOT/examples/acados_python/getting_started
python3 minimal_example_ocp.py
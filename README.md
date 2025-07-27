# erl_covariance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ROS1](https://img.shields.io/badge/ROS1-noetic-blue)](http://wiki.ros.org/)
[![ROS2](https://img.shields.io/badge/ROS2-humble-blue)](https://docs.ros.org/)

**`erl_covariance` provides a collection of kernel functions.**

## Kernel Functions

- [Covariance Base Class](include/erl_covariance/covariance.hpp) - Base class for all covariance functions
- [Matern 3/2](include/erl_covariance/matern32.hpp) - Matern 3/2 kernel function
- [Ornstein-Uhlenbeck](include/erl_covariance/ornstein_uhlenbeck.hpp) - Ornstein-Uhlenbeck kernel function
- [Radial Basis Function (RBF)](include/erl_covariance/radial_bias_function.hpp) - Radial basis function kernel
- [Rational Quadratic](include/erl_covariance/rational_quadratic.hpp) - Rational quadratic kernel function
- [Reduced Rank Covariance](include/erl_covariance/reduced_rank_covariance.hpp) - Base class of reduced rank covariance approximation
- [Reduced Rank Matern 3/2](include/erl_covariance/reduced_rank_matern32.hpp) - Reduced rank Matern 3/2 kernel

## Getting Started

### Create Workspace

```bash
cd <your_workspace>
mkdir -p src
vcs import --input https://raw.githubusercontent.com/ExistentialRobotics/erl_covariance/refs/head/main/erl_covariance.repos src
```

### Install Dependencies

- CMake >= 3.16
- C++17 compatible compiler
- [erl_cmake_tools](https://github.com/ExistentialRobotics/erl_cmake_tools)
- [erl_common](https://github.com/ExistentialRobotics/erl_common)

```bash
# Ubuntu 20.04
wget -qO - https://raw.githubusercontent.com/ExistentialRobotics/erl_common/refs/heads/main/scripts/setup_ubuntu_20.04.bash | bash
# Ubuntu 22.04, 24.04
wget -qO - https://raw.githubusercontent.com/ExistentialRobotics/erl_common/refs/heads/main/scripts/setup_ubuntu_22.04_24.04.bash | bash
```

### Use as a standard CMake package

```bash
cd <your_workspace>
touch CMakeLists.txt
```

Add the following lines to your `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(<your_project_name>)
add_subdirectory(src/erl_cmake_tools)
add_subdirectory(src/erl_common)
add_subdirectory(src/erl_covariance)
```

Then run the following commands:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j`nproc`
```

### Use as a ROS package

```bash
cd <your_workspace>/src
source /opt/ros/<ros_distro>/setup.bash
# for ROS1
catkin build erl_covariance
source devel/setup.bash
# for ROS2
colcon build --packages-up-to erl_covariance
source install/setup.bash
```

### Install as a Python package

**We also provide the Python bindings!**

- Make sure you have installed all dependencies.
- Make sure you have the correct Python environment activated, `pipenv` is recommended.

```bash
cd <your_workspace>
for package in erl_cmake_tools erl_common erl_covariance; do
    cd src/$package
    pip install . --verbose
    cd ../..
done
```

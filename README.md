erl_covariance
==============

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ROS1](https://img.shields.io/badge/ROS1-noetic-blue)](http://wiki.ros.org/)
[![ROS2](https://img.shields.io/badge/ROS2-humble-blue)](https://docs.ros.org/)

`erl_covariance` provides a collection of kernel functions.

## Kernel Functions

- [Covariance Base Class](include/erl_covariance/covariance.hpp) - Base class for all covariance functions
- [Matern 3/2](include/erl_covariance/matern32.hpp) - Matern 3/2 kernel function
- [Ornstein-Uhlenbeck](include/erl_covariance/ornstein_uhlenbeck.hpp) - Ornstein-Uhlenbeck kernel function
- [Radial Basis Function (RBF)](include/erl_covariance/radial_bias_function.hpp) - Radial basis function kernel
- [Rational Quadratic](include/erl_covariance/rational_quadratic.hpp) - Rational quadratic kernel function
- [Reduced Rank Covariance](include/erl_covariance/reduced_rank_covariance.hpp) - Base class of reduced rank covariance approximation
- [Reduced Rank Matern 3/2](include/erl_covariance/reduced_rank_matern32.hpp) - Reduced rank Matern 3/2 kernel

# Install Dependencies

- CMake >= 3.16
- C++17 compatible compiler
- [erl_cmake_tools](https://github.com/ExistentialRobotics/erl_cmake_tools)
- [erl_common](https://github.com/ExistentialRobotics/erl_common)

# Getting Started

## Create Workspace

```bash
cd <your_workspace>
mkdir -p src
vcs import --input https://raw.githubusercontent.com/ExistentialRobotics/erl_covariance/main/erl_covariance.repos src
```

## Use as a standard CMake package

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

## Use as a ROS package

```bash
cd <your_workspace>/src
catkin build erl_covariance # for ROS1
colcon build --packages-up-to erl_covariance # for ROS2
```

## Install as a Python package

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

# Usage

The library provides various kernel functions that can be used for Gaussian process modeling and covariance estimation. Each kernel function implements the base `Covariance` interface and provides methods for computing covariance matrices, derivatives, and other kernel-specific operations.

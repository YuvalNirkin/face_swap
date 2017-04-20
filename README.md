# Nirkin Face Swap - Swapping faces in extremely hard conditions
Created by Yuval Nirkin.

[nirkin.com](http://www.nirkin.com/)

## Overview

![alt text](https://yuvalnirkin.github.io/face_swap/images/face_swap_samples.jpg "Samples")

## Dependencies
| Library                                                            | Minimum Version | Notes                                    |
|--------------------------------------------------------------------|-----------------|------------------------------------------|
| [Boost](http://www.boost.org/)                                     | 1.47            |                                          |
| [OpenCV](http://opencv.org/)                                       | 3.0             |                                          |
| [find_face_landmarks](https://github.com/YuvalNirkin/find_face_landmarks) | 1.1      |                                          |
| [face_segmentation](https://github.com/YuvalNirkin/face_segmentation) | 0.9          |                                          |
| [Caffe](https://github.com/BVLC/caffe)                             | 1.0             |☕️                                        |
| [Eigen](http://eigen.tuxfamily.org)                                | 3.0.0           |                                          |
| [GLEW](http://glew.sourceforge.net/)                               | 2.0.0           |                                          |
| [Qt](https://www.qt.io/)                                           | 5.4.0           |                                          |
| [HDF5](https://support.hdfgroup.org/HDF5/)                         | 1.8.18          |                                          |

## Installation
- Use CMake and your favorite compiler to build and install the library.
- Download the [landmarks model file](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract to "data" in the installation directory.
- Download the [face_seg_fcn8s.zip](https://github.com/YuvalNirkin/face_segmentation/releases/download/0.9/face_seg_fcn8s.zip) and extract to "data" in the installation directory.
- Download the [3dmm_cnn_resnet_101.zip](https://github.com/YuvalNirkin/face_swap/releases/download/0.9/3dmm_cnn_resnet_101.zip) and extract to "data" in the installation directory.
- Add face_swap/bin to path.

## Usage
- For using the library's C++ interface, please take a look at the [Doxygen generated documentation](https://yuvalnirkin.github.io/face_swap/).

## Bibliography

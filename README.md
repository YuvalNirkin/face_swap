# End-to-end, automatic face swapping pipeline
![alt text](https://yuvalnirkin.github.io/assets/img/projects/face_swap/joker_teaser.jpg "Samples")
The Joker (Heath Ledger) swapped using our method onto very different subjects and images.

[Yuval Nirkin](http://www.nirkin.com/), [Iacopo Masi](http://www-bcf.usc.edu/~iacopoma/), [Anh Tuan Tran](https://sites.google.com/site/anhttranusc/), [Tal Hassner](http://www.openu.ac.il/home/hassner/), and [Gerard Medioni](http://iris.usc.edu/people/medioni/index.html).

# Overview
Code for the automatic, image-to-image face swapping method described in the paper:

Yuval Nirkin, Iacopo Masi, Anh Tuan Tran, Tal Hassner, Gerard Medioni, "[On Face Segmentation, Face Swapping, and Face Perception](https://arxiv.org/abs/1704.06729)", IEEE Conference on Automatic Face and Gesture Recognition (FG), Xi'an, China on May 2018

Please see [project page](http://www.openu.ac.il/home/hassner/projects/faceswap/) for more details, more resources and updates on this project.

If you find this code useful, please make sure to cite our paper in your work.

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
- Download the [face_seg_fcn8s.zip](https://github.com/YuvalNirkin/face_segmentation/releases/download/1.0/face_seg_fcn8s.zip) and extract to "data" in the installation directory.
- Download the [3dmm_cnn_resnet_101.zip](https://github.com/YuvalNirkin/face_swap/releases/download/0.9/3dmm_cnn_resnet_101.zip) and extract to "data" in the installation directory.
- Add face_swap/bin to path.

## Usage
- For using the library's C++ interface, please take a look at the [Doxygen generated documentation](https://yuvalnirkin.github.io/docs/face_swap/).
- For running the face swap on a pair of images, it's best to first create a configuration file because of the large number of parameters. For this example create a configuration file "test.cfg" under "bin" in the installation directory with the following parameters:
```Ini
landmarks = ../data/shape_predictor_68_face_landmarks.dat       # path to landmarks model file
model_3dmm_h5 = ../data/BaselFaceModel_mod_wForehead_noEars.h5  # path to 3DMM file (.h5)
model_3dmm_dat = ../data/BaselFace.dat                          # path to 3DMM file (.dat)
reg_model = ../data/3dmm_cnn_resnet_101.caffemodel              # path to 3DMM regression CNN model file (.caffemodel)
reg_deploy = ../data/3dmm_cnn_resnet_101_deploy.prototxt        # path to 3DMM regression CNN deploy file (.prototxt)
reg_mean = ../data/3dmm_cnn_resnet_101_mean.binaryproto         # path to 3DMM regression CNN mean file (.binaryproto)
seg_model = ../data/face_seg_fcn8s.caffemodel                   # path to face segmentation CNN model file (.caffemodel)
seg_deploy = ../data/face_seg_fcn8s_deploy.prototxt             # path to face segmentation CNN deploy file (.prototxt)
generic = 0                                 # use generic mode (disable shape regression)
expressions = 1                             # use expression regression
gpu = 1                                     # toggle GPU / CPU
gpu_id = 0                                  # GPU's device id
verbose = 1                                 # 1 = before blend image, 2 += projected meshes, 3 += landmarks, 4 += meshes ply
input = ../data/images/brad_pitt_01.jpg     # source image
input = ../data/images/bruce_willis_01.jpg  # target image
output = out.jpg                            # output image or directory
```
Now run the following commands:
```Bash
cd path/to/face_swap/bin
face_swap_image --cfg test.cfg
```
- For running the face swap on a list of images, prepare a csv file in which each line contains the paths to a pair of images, separated by a comma. For this example create a file "img_list.csv" like the following:
```
../data/images/brad_pitt_01.jpg,../data/images/bruce_willis_01.jpg
../data/images/bruce_willis_01.jpg,../data/images/brad_pitt_01.jpg
```
Replace the input and output parameters from "test.cfg" with the following:
```Ini
log = test.log                              # path to log file
input = img_list.csv                        # list file or directory
output = .                                  # output directory
```
Now run the following commands:
```Bash
cd path/to/face_swap/bin
face_swap_batch --cfg test.cfg
```
- It's also possible to run on entire image directories. In that case all possible pairs will be processed. Just specify a directory in the input parameter for face_swap_batch.

## Citation

Please cite our paper with the following bibtex if you use our face renderer:

``` latex
@inproceedings{nirkin2018_faceswap,
      title={On Face Segmentation, Face Swapping, and Face Perception},
      booktitle = {IEEE Conference on Automatic Face and Gesture Recognition},
      author={Nirkin, Yuval and Masi, Iacopo and Tran, Anh Tuan and Hassner, Tal and Medioni, G\'{e}rard Medioni},
      year={2018},
    }
```

## Related projects
- [Deep face segmentation](https://github.com/YuvalNirkin/face_segmentation), used to segment face regions in the face swapping pipeline.
- [Interactive system for fast face segmentation ground truth labeling](https://github.com/YuvalNirkin/face_video_segment), used to produce the training set for our deep face segmentation.
- [CNN3DMM](http://www.openu.ac.il/home/hassner/projects/CNN3DMM/), used to estimate 3D face shapes from single images.
- [ResFace101](http://www.openu.ac.il/home/hassner/projects/augmented_faces/), deep face recognition used in the paper to test face swapping capabilities. 

## Copyright
Copyright 2017, Yuval Nirkin, Iacopo Masi, Anh Tuan Tran, Tal Hassner, and Gerard Medioni 

The SOFTWARE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use.


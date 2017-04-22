![alt text](https://yuvalnirkin.github.io/face_swap/images/face_swap_samples.jpg "Samples")

# End-to-end, automatic face swapping pipeline
[Yuval Nirkin](http://www.nirkin.com/), [Iacopo Masi](http://www-bcf.usc.edu/~iacopoma/), [Anh Tuan Tran](https://sites.google.com/site/anhttranusc/), [Tal Hassner](http://www.openu.ac.il/home/hassner/), and [Gerard Medioni](http://iris.usc.edu/people/medioni/index.html).

Code for the automatic, image-to-image face swapping method described in the paper:


Yuval Nirkin, Iacopo Masi, Anh Tuan Tran, Tal Hassner, Gerard Medioni, "On Face Segmentation, Face Swapping, and Face Perception," arXiv preprint.


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
- Download the [face_seg_fcn8s.zip](https://github.com/YuvalNirkin/face_segmentation/releases/download/0.9/face_seg_fcn8s.zip) and extract to "data" in the installation directory.
- Download the [3dmm_cnn_resnet_101.zip](https://github.com/YuvalNirkin/face_swap/releases/download/0.9/3dmm_cnn_resnet_101.zip) and extract to "data" in the installation directory.
- Add face_swap/bin to path.

## Usage
- For using the library's C++ interface, please take a look at the [Doxygen generated documentation](https://yuvalnirkin.github.io/face_swap/).

## Related projects
- [Deep face segmentation](https://github.com/YuvalNirkin/face_segmentation), used to segment face regions in the face swapping pipeline.
- [Interactive system for fast face segmentation ground truth labeling](https://github.com/YuvalNirkin/face_video_segment), used to produce the training set for our deep face segmentation.
- [CNN3DMM](http://www.openu.ac.il/home/hassner/projects/CNN3DMM/), used in the tests reported in the paper to estimate 3D face shapes from single images.
- [ResFace101](http://www.openu.ac.il/home/hassner/projects/augmented_faces/), deep face recognition used in the paper to test face swapping capabilities. 

## Copyright
Copyright 2017, Yuval Nirkin, Iacopo Masi, Anh Tuan Tran, Tal Hassner, and Gerard Medioni 

The SOFTWARE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use.


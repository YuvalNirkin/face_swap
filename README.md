# End-to-end, automatic face swapping pipeline
![alt text](https://yuvalnirkin.github.io/assets/img/projects/face_swap/joker_teaser.jpg "Samples")
The Joker (Heath Ledger) swapped using our method onto very different subjects and images.

[Yuval Nirkin](http://www.nirkin.com/), [Iacopo Masi](http://www-bcf.usc.edu/~iacopoma/), [Anh Tuan Tran](https://sites.google.com/site/anhttranusc/), [Tal Hassner](http://www.openu.ac.il/home/hassner/), and [Gerard Medioni](http://iris.usc.edu/people/medioni/index.html).

# Overview
Code for the automatic, image-to-image face swapping method described in the paper:

Yuval Nirkin, Iacopo Masi, Anh Tuan Tran, Tal Hassner, Gerard Medioni, "[On Face Segmentation, Face Swapping, and Face Perception](https://arxiv.org/abs/1704.06729)", IEEE Conference on Automatic Face and Gesture Recognition (FG), Xi'an, China on May 2018

Please see [project page](http://www.openu.ac.il/home/hassner/projects/faceswap/) for more details, more resources and updates on this project.

If you find this code useful, please make sure to cite our paper in your work.

## News
Version 1.0 is released:
- Better performance and results.
- Python binding.
- Easier to build: removed many of the dependencies.
- Software renderer: OpenGL is no longer required, the project can now be built on gui-less servers without any problems. 
- Added low resolution segmentation model for GPUs with low memory.

## Installation
Both Linux and Windows are supported.
- [Ubuntu installation guide](https://github.com/YuvalNirkin/face_swap/wiki/Ubuntu-Installation-Guide)
- [Windows installation guide](https://github.com/YuvalNirkin/face_swap/wiki/Windows-Installation-Guide)

## Usage
- For Python follow the [Python guide](https://github.com/YuvalNirkin/face_swap/wiki/Python-Guide)
- For running the applications follow the [applications guide](https://github.com/YuvalNirkin/face_swap/wiki/Applications-Guide).
- For using the library's C++ interface, please take a look at the [Doxygen generated documentation](https://yuvalnirkin.github.io/docs/face_swap/).

## Citation
Please cite our paper with the following bibtex if you use our face renderer:

``` latex
@inproceedings{nirkin2018_faceswap,
      title={On Face Segmentation, Face Swapping, and Face Perception},
      booktitle = {IEEE Conference on Automatic Face and Gesture Recognition},
      author={Nirkin, Yuval and Masi, Iacopo and Tran, Anh Tuan and Hassner, Tal and Medioni, G\'{e}rard},
      year={2018},
    }
```

## Related projects
- [Deep face segmentation](https://github.com/YuvalNirkin/face_segmentation), used to segment face regions in the face swapping pipeline.
- [Interactive system for fast face segmentation ground truth labeling](https://github.com/YuvalNirkin/face_video_segment), used to produce the training set for our deep face segmentation.
- [CNN3DMM](http://www.openu.ac.il/home/hassner/projects/CNN3DMM/), used to estimate 3D face shapes from single images.
- [ResFace101](http://www.openu.ac.il/home/hassner/projects/augmented_faces/), deep face recognition used in the paper to test face swapping capabilities.
- [Hand augmentations](https://github.com/YuvalNirkin/egohands_augmentations), used to generate the hand augmentations for training the face segmentation.

## Copyright
Copyright 2018, Yuval Nirkin, Iacopo Masi, Anh Tuan Tran, Tal Hassner, and Gerard Medioni 

The SOFTWARE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use.

## License
This project is released under the GPLv3 license.

For a more permitting license please contact the [author](mailto:yuval.nirkin@gmail.com).

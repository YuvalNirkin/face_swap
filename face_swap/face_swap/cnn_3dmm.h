#ifndef FACE_SWAP_CNN_3DMM_H
#define FACE_SWAP_CNN_3DMM_H

// Includes
#include <opencv2/core.hpp>

// Caffe
#include <caffe/caffe.hpp>

#include <string>

namespace face_swap
{
	/**	This class provided face shape and texture estimation using Caffe with
	a convolutional neural network.
	The CNN estimates shape and texture coefficients for a PCA model that is based on
	Basel's 3D Morphable Model.
	This is an implementation of the following papers:
	-# Regressing Robust and Discriminative 3D Morphable Models with a very Deep
	Neural Network, Anh Tuan Tran, Tal Hassner, Iacopo Masi and Gerard Medioni.
	-# A 3D Face Model for Pose and Illumination Invariant Face Recognition, 
	P. Paysan and R. Knothe and B. Amberg and S. Romdhani and T. Vetter.
	*/
    class CNN3DMM
    {
    public:

		/** Creates an instance of CNN3DMM.
		@param deploy_file Path to 3DMM regression CNN deploy file (.prototxt).
		@param caffe_model_file Path to 3DMM regression CNN model file (.caffemodel).
		@param mean_file Path to 3DMM regression CNN mean file (.binaryproto).
		@param init_cnn if true the CNN will be initialized.
		@param with_gpu Toggle GPU\CPU execution.
		@param gpu_device_id Set the GPU's device id.
		*/
        CNN3DMM(const std::string& deploy_file, const std::string& caffe_model_file,
            const std::string& mean_file, bool init_cnn = true,
			bool with_gpu = true, int gpu_device_id = 0);

		/** Destructor.
		*/
        ~CNN3DMM();

		/** Estimate face shape and texture from image.
		@param[in] img The image to process.
		@param[out] shape_coefficients PCA shape coefficients.
		@param[out] tex_coefficients PCA texture coefficients.
		*/
        void process(const cv::Mat& img, 
            cv::Mat& shape_coefficients, cv::Mat& tex_coefficients);

    private:
        void wrapInputLayer(std::vector<cv::Mat>& input_channels);

        cv::Mat readMean(const std::string& mean_file) const;

        cv::Mat preprocess(const cv::Mat& img);

        void copyInputData(cv::Mat& img);

    protected:
        std::shared_ptr<caffe::Net<float> > m_net;
        int m_num_channels;
        cv::Size m_input_size;
        cv::Mat m_mean;
        bool m_with_gpu;
    };

}   // namespace face_swap

#endif // FACE_SWAP_CNN_3DMM_H

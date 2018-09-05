#ifndef FACE_SWAP_FACE_SEG_H
#define FACE_SWAP_FACE_SEG_H

#include "face_swap/face_swap_export.h"

// std
#include <string>

// OpenCV
#include <opencv2/core.hpp>

// Caffe
#include <caffe/caffe.hpp>

namespace face_swap
{
	/**	This class provided deep face segmentation using Caffe with a fully connected
		convolutional neural network.
	*/
    class FACE_SWAP_EXPORT FaceSeg
    {
    public:
		/**	Construct FaceSeg instance.
			@param deploy_file Network definition file for deployment (.prototxt).
			@param model_file Network weights model file (.caffemodel).
			@param with_gpu Toggle GPU\CPU.
			@param gpu_device_id Set the GPU's device id.
			@param scale Scale image to the network's maximum size (depicted by the prototxt file).
			@param postprocess_seg Toggle postprocessing of the segmentation.
		*/
		FaceSeg(const std::string& deploy_file, const std::string& model_file,
            bool with_gpu = true, int gpu_device_id = 0,
			bool scale = true, bool postprocess_seg = false);

        ~FaceSeg();

		/**	Do face segmentation.
			@param img BGR color image.
			@return 8-bit segmentation mask, 255 for face pixels and 0 for
			background pixels.
		*/
        cv::Mat process(const cv::Mat& img);

    private:

		/** Wrap the input layer of the network in separate cv::Mat objects
			(one per channel). This way we save one memcpy operation and we
			don't need to rely on cudaMemcpy2D. The last preprocessing operation 
			will write the separate channels directly to the input layer.
			@param input_channels Input image channels.
		*/
        void wrapInputLayer(std::vector<cv::Mat>& input_channels);

		/**	Preprocess image for network.
			@param img BGR color image.
			@param input_channels Input image channels.
		*/
        void preprocess(const cv::Mat& img, std::vector<cv::Mat>& input_channels);

		/**	Substract mean color from image.
			@param img BGR float image.
		*/
        void subtractMean_c3(cv::Mat& img);

    protected:
        std::shared_ptr<caffe::Net<float>> m_net;
        int m_num_channels;
        cv::Size m_input_size;
        bool m_with_gpu;
		bool m_scale;
		bool m_postprocess_seg;
		int m_foreground_channel = 1;

		// Mean pixel color
		const float MB = 104.00699f, MG = 116.66877f, MR = 122.67892f;
    };

}   // namespace face_swap

#endif // FACE_SWAP_FACE_SEG_H

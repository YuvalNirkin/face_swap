#ifndef FACE_SWAP_CNN_3DMM_EXPR_H
#define FACE_SWAP_CNN_3DMM_EXPR_H


#include "cnn_3dmm.h"

class FaceServices2;

namespace face_swap
{
	/**	This class provided face shape, texture, expression and pose estimations using
	Caffe with a convolutional neural network.
	The CNN estimates shape and texture coefficients for a PCA model that is based on
	Basel's 3D Morphable Model. The pose and expression are then estimated using epnp
	optimization. This is an implementation of the following papers:
	-# Regressing Robust and Discriminative 3D Morphable Models with a very Deep
	Neural Network, Anh Tuan Tran, Tal Hassner, Iacopo Masi and Gerard Medioni.
	-# A 3D Face Model for Pose and Illumination Invariant Face Recognition,
	P. Paysan and R. Knothe and B. Amberg and S. Romdhani and T. Vetter.
	*/
    class CNN3DMMExpr : public CNN3DMM
    {
    public:

		/** Creates an instance of CNN3DMMExpr.
		@param deploy_file Path to 3DMM regression CNN deploy file (.prototxt).
		@param caffe_model_file Path to 3DMM regression CNN model file (.caffemodel).
		@param mean_file Path to 3DMM regression CNN mean file (.binaryproto).
		@param model_file Path to 3DMM file (.dat).
		@param generic Use generic model without shape regression.
		@param with_expr Toggle fitting face expressions.
		@param with_gpu Toggle GPU\CPU execution.
		@param gpu_device_id Set the GPU's device id.
		*/
        CNN3DMMExpr(const std::string& deploy_file, const std::string& caffe_model_file,
            const std::string& mean_file, const std::string& model_file,
            bool generic = false, bool with_expr = true,
			bool with_gpu = true, int gpu_device_id = 0);

		/** Destructor.
		*/
        ~CNN3DMMExpr();

		/** Estimate face pose and shape, texture, expression coefficients from image.
		@param[in] img The image to process.
		@param[in] landmarks The face landmarks detected on the specified image.
		@param[out] shape_coefficients PCA shape coefficients.
		@param[out] tex_coefficients PCA texture coefficients.
		@param[out] expr_coefficients PCA expression coefficients.
		@param[out] vecR Face's rotation vector [Euler angles].
		@param[out] vecT Face's translation vector.
		@param[out] K Camera intrinsic parameters.
		*/
        void process(const cv::Mat& img, const std::vector<cv::Point>& landmarks,
            cv::Mat& shape_coefficients, cv::Mat& tex_coefficients,
            cv::Mat& expr_coefficients, cv::Mat& vecR, cv::Mat& vecT, cv::Mat& K);
    private:
        std::unique_ptr<FaceServices2> fservice;
        bool m_generic, m_with_expr;
    };

}   // namespace face_swap

#endif // FACE_SWAP_CNN_3DMM_EXPR_H

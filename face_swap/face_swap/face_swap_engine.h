#ifndef FACE_SWAP_FACE_SWAP_ENGINE_H
#define FACE_SWAP_FACE_SWAP_ENGINE_H

#include "face_swap/face_swap_export.h"
#include "face_swap/basel_3dmm.h"

// std
#include <memory>

// OpenCV
#include <opencv2/core.hpp>


namespace face_swap
{
	struct FaceData
	{
		// Input
		cv::Mat img;
		cv::Mat seg;

		// Intermediate pipeline data
		cv::Mat scaled_img;
		cv::Mat scaled_seg;
		cv::Mat cropped_img;
		cv::Mat cropped_seg;
		std::vector<cv::Point> scaled_landmarks;
		std::vector<cv::Point> cropped_landmarks;
		cv::Rect bbox;
		cv::Rect scaled_bbox;
		cv::Mat shape_coefficients, tex_coefficients, expr_coefficients;
		cv::Mat vecR, vecT, K;

		// Flipped image data
		cv::Mat shape_coefficients_flipped, tex_coefficients_flipped, expr_coefficients_flipped;
		cv::Mat vecR_flipped, vecT_flipped;

		// Processing parameters
		bool enable_seg = true;
		int max_bbox_res = 0;
	};

	/** Face swap interface.
	*/
	class FACE_SWAP_EXPORT FaceSwapEngine
	{
	public:

		/**	Transfer the face in the source image onto the face in the target image.
		@param[in] src_data Includes all the images and intermediate data for the specific face.
		@param[in] tgt_data Includes all the images and intermediate data for the specific face.
		@return The output face swapped image.
		*/
		virtual cv::Mat swap(FaceData& src_data, FaceData& tgt_data) = 0;

		/** Process a single image and save the intermediate face data.
		@param[in] face_data Includes all the images and intermediate data for the specific face.
		@param[in] process_flipped Toggle processing of flipped image.
		@return true for success and false for failure.
		*/
		virtual bool process(FaceData& face_data, bool process_flipped = false) = 0;

		virtual cv::Mat renderFaceData(const FaceData& face_data, float scale = 1.0f) = 0;

		/**	Construct FaceSwapEngine instance.
		@param landmarks_path Path to the landmarks model file.
		@param model_3dmm_h5_path Path to 3DMM file (.h5).
		@param model_3dmm_dat_path Path to 3DMM file (.dat).
		@param reg_model_path Path to 3DMM regression CNN model file (.caffemodel).
		@param reg_deploy_path Path to 3DMM regression CNN deploy file (.prototxt).
		@param reg_mean_path Path to 3DMM regression CNN mean file (.binaryproto).
		@param seg_model_path Path to face segmentation CNN model file (.caffemodel).
		@param seg_deploy_path Path to face segmentation CNN deploy file (.prototxt).
		@param generic Use generic model without shape regression.
		@param with_expr Toggle fitting face expressions.
		@param with_gpu Toggle GPU\CPU execution.
		@param gpu_device_id Set the GPU's device id.
		*/
		static std::shared_ptr<FaceSwapEngine> createInstance(
			const std::string& landmarks_path, const std::string& model_3dmm_h5_path,
			const std::string& model_3dmm_dat_path, const std::string& reg_model_path,
			const std::string& reg_deploy_path, const std::string& reg_mean_path,
			const std::string& seg_model_path, const std::string& seg_deploy_path,
			bool generic = false, bool with_expr = true, bool with_gpu = true,
			int gpu_device_id = 0);
	};

}   // namespace face_swap

#endif // FACE_SWAP_FACE_SWAP_ENGINE_H
    
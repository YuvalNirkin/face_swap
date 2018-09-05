#ifndef FACE_SWAP_FACE_SWAP_ENGINE_IMPL_H
#define FACE_SWAP_FACE_SWAP_ENGINE_IMPL_H

#include "face_swap_engine.h"
#include "face_swap/cnn_3dmm_expr.h"
#include "face_swap/basel_3dmm.h"
#include "face_swap/face_detection_landmarks.h"
#include "face_swap/face_seg.h"

// std
#include <memory>


namespace face_swap
{
	/** Face swap interface.
	*/
	class FaceSwapEngineImpl : public FaceSwapEngine
	{
	public:
		FaceSwapEngineImpl(
			const std::string& landmarks_path, const std::string& model_3dmm_h5_path,
			const std::string& model_3dmm_dat_path, const std::string& reg_model_path,
			const std::string& reg_deploy_path, const std::string& reg_mean_path,
			const std::string& seg_model_path, const std::string& seg_deploy_path,
			bool generic = false, bool with_expr = true, bool with_gpu = true,
			int gpu_device_id = 0);

		/**	Transfer the face in the source image onto the face in the target image.
		@param[in] src_data Includes all the images and intermediate data for the specific face.
		@param[in] tgt_data Includes all the images and intermediate data for the specific face.
		@return The output face swapped image.
		*/
		cv::Mat swap(FaceData& src_data, FaceData& tgt_data);

		/** Process a single image and save the intermediate face data.
		@param[in] face_data Includes all the images and intermediate data for the specific face.
		@param[in] process_flipped Toggle processing of flipped image.
		@return true for success and false for failure.
		*/
		bool process(FaceData& face_data, bool process_flipped = false);

		cv::Mat renderFaceData(const FaceData& img_data, float scale = 1.0f);

	private:

		/** Crops the image and it's corresponding segmentation according
		to the detected face landmarks. Optinally scale all the images and segmetations.
		@param[in] face_data Includes all the images and intermediate data for the specific face.
		@return true for success and false for failure.
		*/
		bool preprocessImages(FaceData& face_data);

	private:
		//std::shared_ptr<sfl::SequenceFaceLandmarks> m_sfl;
		std::shared_ptr<FaceDetectionLandmarks> m_lms;
		std::unique_ptr<CNN3DMMExpr> m_cnn_3dmm_expr;
		std::unique_ptr<Basel3DMM> m_basel_3dmm;
		std::unique_ptr<FaceSeg> m_face_seg;

		bool m_with_gpu;
		int m_gpu_device_id;
	};

}   // namespace face_swap

#endif // FACE_SWAP_FACE_SWAP_ENGINE_IMPL_H

#ifndef FACE_SWAP_FACE_SWAP_H
#define FACE_SWAP_FACE_SWAP_H

#include "face_swap/cnn_3dmm_expr.h"
#include "face_swap/basel_3dmm.h"
#include "face_swap/face_renderer.h"

// OpenCV
#include <opencv2/core.hpp>

// sfl
#include <sfl/sequence_face_landmarks.h>
#include <sfl/utilities.h>

// face_seg
#include <face_seg/face_seg.h>

namespace face_swap
{
    class FaceSwap
    {
    public:
		/**	Construct FaceSwap instance.
		@param landmarks_path Path to the landmarks model file.
		@param model_3dmm_h5_path Path to 3DMM file (.h5).
		@param model_3dmm_dat_path Path to 3DMM file (.dat).
		@param reg_model_path Path to 3DMM regression CNN model file (.caffemodel).
		@param reg_deploy_path Path to 3DMM regression CNN deploy file (.prototxt).
		@param reg_mean_path Path to 3DMM regression CNN mean file (.binaryproto).
		@param generic Use generic model without shape regression.
		@param with_expr Toggle fitting face expressions.
		@param with_gpu Toggle GPU\CPU execution.
		@param gpu_device_id Set the GPU's device id.
		*/
        FaceSwap(const std::string& landmarks_path, const std::string& model_3dmm_h5_path,
            const std::string& model_3dmm_dat_path, const std::string& reg_model_path,
            const std::string& reg_deploy_path, const std::string& reg_mean_path,
            bool generic = false, bool with_expr = true, bool with_gpu = true, 
			int gpu_device_id = 0);

		/**	Set segmentation model.
		Source and Target segmentations will be calculated using this model
		if not specified. GPU\CPU execution and the GPU's device id are determined
		in the constructor.
		@param seg_model_path Path to face segmentation CNN model file (.caffemodel).
		@param seg_deploy_path Path to face segmentation CNN deploy file (.prototxt).
		*/
		void setSegmentationModel(const std::string& seg_model_path, 
			const std::string& seg_deploy_path);

		/**	Clear previously set segmentation model.
		Source and Target segmentation will not longer be calculated if not specified.
		*/
		void clearSegmentationModel();

		/**	Check whether the segmentation model is initialized.
		*/
		bool isSegmentationModelInit();

		/**	Set source image and segmentation.
		*/
        bool setSource(const cv::Mat& img, const cv::Mat& seg = cv::Mat());

		/**	Set target image and segmentation.
		*/
        bool setTarget(const cv::Mat& img, const cv::Mat& seg = cv::Mat());

		/**	Transfer the face from the source image onto the face in the target image.
		*/
        cv::Mat swap();

		/**	Get the 3D reconstruced mesh of the source face.
		If the "generic" option is enabled than this will return the generic shape.
		*/
        const Mesh& getSourceMesh() const;

		/**	Get the 3D reconstruced mesh of the target face.
		If the "generic" option is enabled than this will return the generic shape.
		*/
        const Mesh& getTargetMesh() const;

    private:

		/** Crops the image and it's corresponding segmentation according
		to the detected face landmarks.
		@param[in] img The image to crop.
		@param[in] seg The segmentation to crop (must be the same size as the image).
		@param[out] landmarks The detected face landmarks for the original image.
		@param[out] cropped_landmarks The detected face landmarks for the cropped image.
		@param[out] cropped_img The cropped image.
		@param[out] cropped_seg The cropped segmentation.
		@param[out] bbox The cropping bounding box.
		@return true for success and false for failure.
		*/
        bool preprocessImages(const cv::Mat& img, const cv::Mat& seg,
            std::vector<cv::Point>& landmarks, std::vector<cv::Point>& cropped_landmarks,
            cv::Mat& cropped_img, cv::Mat& cropped_seg, cv::Rect& bbox);

		/** Crops the image and it's corresponding segmentation according
		to the detected face landmarks.
		@param[in] img The image to crop.
		@param[in] seg The segmentation to crop (must be the same size as the image).
		@param[out] landmarks The detected face landmarks for the original image.
		@param[out] cropped_landmarks The detected face landmarks for the cropped image.
		@param[out] cropped_img The cropped image.
		@param[out] cropped_seg The cropped segmentation.
		@return true for success and false for failure.
		*/
        bool preprocessImages(const cv::Mat& img, const cv::Mat& seg,
            std::vector<cv::Point>& landmarks, std::vector<cv::Point>& cropped_landmarks,
            cv::Mat& cropped_img, cv::Mat& cropped_seg);

		/**	Generate texture for the mesh based on the image size, intrinsic and
		extrinsic transformations.
		@param[in] mesh The mesh to generate the texture for.
		@param[in] img The image for the texture
		@param[in] seg The segmentation for the texture (will be used as the
		texture's alpha channel).
		@param[in] vecR Mesh's rotation vector [Euler angles].
		@param[in] vecT Mesh's translation vector.
		@param[in] K Camera intrinsic parameters.
		@param[out] tex Generated texture image.
		@param[out] uv Generated texture coordinates.
		*/
        void generateTexture(const Mesh& mesh, const cv::Mat& img, const cv::Mat& seg,
            const cv::Mat& vecR, const cv::Mat& vecT, const cv::Mat& K,
            cv::Mat& tex, cv::Mat& uv);

		/**	Generate texture coordinates for the mesh based on the image size,
		intrinsic and extrinsic transformations.
		@param[in] mesh The mesh to generate the texture coordinates for.
		@param[in] img_size The image size that the texture coordinates will be
		relative to.
		@param[in] vecR Mesh's rotation vector [Euler angles].
		@param[in] vecT Mesh's translation vector.
		@param[in] K Camera intrinsic parameters.
		@return n X 2 matrix where n is the number of vertices. Each row contain the
		xy texture coordinate of the corresponding vertex.
		*/
        cv::Mat generateTextureCoordinates(const Mesh& mesh, const cv::Size& img_size,
            const cv::Mat& vecR, const cv::Mat& vecT, const cv::Mat& K);

		/**	Blend the source and destination images based on a mask for the soure
		image. The mask is calculated by first removing black background pixels and
		then doing bitwise and with the destination's segmentation if it is specified.
		@param[in] src The source image.
		@param[in] dst The destination image.
		@param[in] dst_seg The segmentation for the destination image.
		@return The src and dst blended image.
		*/
        cv::Mat blend(const cv::Mat& src, const cv::Mat& dst,
            const cv::Mat& dst_seg = cv::Mat());

    private:
        std::shared_ptr<sfl::SequenceFaceLandmarks> m_sfl;
        std::unique_ptr<CNN3DMMExpr> m_cnn_3dmm_expr;
        std::unique_ptr<Basel3DMM> m_basel_3dmm;
        std::unique_ptr<FaceRenderer> m_face_renderer;
		std::unique_ptr<face_seg::FaceSeg> m_face_seg;

		bool m_with_gpu;
		int m_gpu_device_id;

        Mesh m_src_mesh, m_dst_mesh;
        cv::Mat m_vecR, m_vecT, m_K;
        cv::Mat m_tex, m_uv;
        cv::Mat m_tgt_cropped_img, m_tgt_cropped_seg;
        cv::Mat m_target_img, m_target_seg;
        cv::Rect m_target_bbox;

        /// Debug ///
        cv::Mat m_source_img;
        cv::Mat m_src_cropped_img, m_src_cropped_seg;
        cv::Mat m_tgt_rendered_img;
        std::vector<cv::Point> m_src_cropped_landmarks;
        std::vector<cv::Point> m_tgt_cropped_landmarks;
        cv::Mat m_src_vecR, m_src_vecT, m_src_K;
        std::vector<cv::Point> m_src_landmarks, m_tgt_landmarks;

    public:
        cv::Mat debugSource();
        cv::Mat debugTarget();
        cv::Mat debug();
        cv::Mat debugSourceMesh();
        cv::Mat debugTargetMesh();
        cv::Mat debugMesh(const cv::Mat& img, const cv::Mat& seg, 
            const cv::Mat& uv, const Mesh& mesh,
            const cv::Mat& vecR, const cv::Mat& vecT, const cv::Mat& K);
        cv::Mat debugSourceLandmarks();
        cv::Mat debugTargetLandmarks();
        cv::Mat debugRender();
        /////////////
    };
}   // namespace face_swap

#endif // FACE_SWAP_FACE_SWAP_H
    
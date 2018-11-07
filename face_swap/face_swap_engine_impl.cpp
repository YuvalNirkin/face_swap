#include "face_swap/face_swap_engine_impl.h"
#include "face_swap/utilities.h"
#include "face_swap/landmarks_utilities.h"

// std
#include <limits>
#include <iostream> // debug

// OpenCV
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  // Debug

namespace face_swap
{
	std::shared_ptr<FaceSwapEngine> FaceSwapEngine::createInstance(
		const std::string& landmarks_path, const std::string& model_3dmm_h5_path,
		const std::string& model_3dmm_dat_path, const std::string& reg_model_path,
		const std::string& reg_deploy_path, const std::string& reg_mean_path,
		const std::string& seg_model_path, const std::string& seg_deploy_path,
		bool generic, bool with_expr, bool with_gpu, int gpu_device_id)
	{
		return std::make_shared<FaceSwapEngineImpl>(
			landmarks_path, model_3dmm_h5_path,
			model_3dmm_dat_path, reg_model_path,
			reg_deploy_path, reg_mean_path,
			seg_model_path, seg_deploy_path,
			generic, with_expr, with_gpu, gpu_device_id);
	}

	FaceSwapEngineImpl::FaceSwapEngineImpl(
		const std::string& landmarks_path, const std::string& model_3dmm_h5_path,
		const std::string& model_3dmm_dat_path, const std::string& reg_model_path,
		const std::string& reg_deploy_path, const std::string& reg_mean_path,
		const std::string& seg_model_path, const std::string& seg_deploy_path,
		bool generic, bool with_expr, bool with_gpu, int gpu_device_id) :
		m_with_gpu(with_gpu),
		m_gpu_device_id(gpu_device_id)
	{
		// Initialize Sequence Face Landmarks
		//m_sfl = sfl::SequenceFaceLandmarks::create(landmarks_path);

		// Initialize detection and landmarks
		m_lms = FaceDetectionLandmarks::create(landmarks_path);

		// Initialize CNN 3DMM with exression
		m_cnn_3dmm_expr = std::make_unique<CNN3DMMExpr>(
			reg_deploy_path, reg_model_path, reg_mean_path, model_3dmm_dat_path,
			generic, with_expr, with_gpu, gpu_device_id);

		// Initialize segmentation model
		if (!(seg_model_path.empty() || seg_deploy_path.empty()))
			m_face_seg = std::make_unique<FaceSeg>(seg_deploy_path,
				seg_model_path, with_gpu, gpu_device_id, true, true);

		// Load Basel 3DMM
		m_basel_3dmm = std::make_unique<Basel3DMM>();
		*m_basel_3dmm = Basel3DMM::load(model_3dmm_h5_path);
	}

	cv::Mat FaceSwapEngineImpl::swap(FaceData& src_data, FaceData& tgt_data)
	{
		// Process images
		process(src_data);
		process(tgt_data);

		// Check if horizontal flip is required
		float src_angle = getFaceApproxHorAngle(src_data.cropped_landmarks);
		float tgt_angle = getFaceApproxHorAngle(tgt_data.cropped_landmarks);
		cv::Mat cropped_src, cropped_src_seg;
		std::vector<cv::Point> cropped_src_landmarks = src_data.cropped_landmarks;
		cv::Mat src_shape_coefficients, src_tex_coefficients, src_expr_coefficients;
		cv::Mat src_vecR, src_vecT;
		cv::Mat src_K = src_data.K;
		if ((src_angle * tgt_angle) < 0 && std::abs(src_angle - tgt_angle) > (CV_PI / 18.0f) &&
            std::abs(src_angle) > (CV_PI / 36.0f))
		{
			// Horizontal flip the source image
			cv::flip(src_data.cropped_img, cropped_src, 1);
			if (!src_data.cropped_seg.empty())
				cv::flip(src_data.cropped_seg, cropped_src_seg, 1);

			// Horizontal flip the source landmarks
			horFlipLandmarks(cropped_src_landmarks, cropped_src.cols);

			// Recalculate source coefficients
			if (src_data.shape_coefficients_flipped.empty() || src_data.expr_coefficients_flipped.empty())
			{
				m_cnn_3dmm_expr->process(cropped_src, cropped_src_landmarks,
					src_data.shape_coefficients_flipped,
					src_data.tex_coefficients_flipped, src_data.expr_coefficients_flipped,
					src_data.vecR_flipped, src_data.vecT_flipped, src_data.K);
			}

			src_shape_coefficients = src_data.shape_coefficients_flipped;
			src_tex_coefficients = src_data.tex_coefficients_flipped;
			src_expr_coefficients = src_data.expr_coefficients_flipped;
			src_vecR = src_data.vecR_flipped;
			src_vecT = src_data.vecT_flipped;
		}
		else
		{
			cropped_src = src_data.cropped_img;
			cropped_src_seg = src_data.cropped_seg;
			src_shape_coefficients = src_data.shape_coefficients;
			src_tex_coefficients = src_data.tex_coefficients;
			src_expr_coefficients = src_data.expr_coefficients;
			src_vecR = src_data.vecR;
			src_vecT = src_data.vecT;
		}

		// Source mesh
		Mesh src_mesh;
		cv::Mat src_tex, src_uv;
		{
			// Create source mesh
			src_mesh = m_basel_3dmm->sample(src_shape_coefficients, src_tex_coefficients,
				src_expr_coefficients);

			// Texture source mesh
			generateTexture(src_mesh, cropped_src, cropped_src_seg, src_vecR, src_vecT, src_K,
				src_tex, src_uv);
		}


		// Create target mesh
		Mesh tgt_mesh = m_basel_3dmm->sample(tgt_data.shape_coefficients,
			tgt_data.tex_coefficients, tgt_data.expr_coefficients);
		tgt_mesh.tex = src_tex;
		tgt_mesh.uv = src_uv;

		////////////////////////////////////////
		// Actual swap
		////////////////////////////////////////

		// Render
		cv::Mat rendered_img = tgt_data.cropped_img.clone();
		cv::Mat depthbuf;
		renderMesh(rendered_img, tgt_mesh, tgt_data.vecR, tgt_data.vecT, tgt_data.K, depthbuf);

		// Copy back to original target image
		cv::Mat tgt_rendered_img = tgt_data.scaled_img.clone();
		rendered_img.copyTo(tgt_rendered_img(tgt_data.scaled_bbox));
		cv::Mat tgt_depthbuf(tgt_data.scaled_img.size(), CV_32F, std::numeric_limits<float>::max());
		depthbuf.copyTo(tgt_depthbuf(tgt_data.scaled_bbox));

		// Create binary mask from the rendered depth buffer
		cv::Mat mask(tgt_depthbuf.size(), CV_8U);
		unsigned char* mask_data = mask.data;
		float* tgt_depthbuf_data = (float*)tgt_depthbuf.data;
		for (int i = 0; i < tgt_depthbuf.total(); ++i)
		{
			if ((*tgt_depthbuf_data++ - 1e-6f) < std::numeric_limits<float>::max())
				*mask_data++ = 255;
			else *mask_data++ = 0;
		}

		// Combine the segmentation with the mask
		if (!tgt_data.scaled_seg.empty())
			cv::bitwise_and(mask, tgt_data.scaled_seg, mask);

		// Blend images
		return blend(tgt_rendered_img, tgt_data.scaled_img, mask);
	}

	bool FaceSwapEngineImpl::process(FaceData& face_data, bool process_flipped)
	{
		// Preprocess input image
		if (face_data.scaled_landmarks.empty())
		{
			if (!preprocessImages(face_data))
				return false;
		}

		// If segmentation was not specified and we have a segmentation model then
		// calculate the segmentation
		bool compute_seg = face_data.scaled_seg.empty() && face_data.enable_seg && m_face_seg != nullptr;
		if (compute_seg)
		{
			face_data.cropped_seg = m_face_seg->process(face_data.cropped_img);
			face_data.scaled_seg = cv::Mat::zeros(face_data.scaled_img.size(), CV_8U);
			face_data.cropped_seg.copyTo(face_data.scaled_seg(face_data.scaled_bbox));
		}

		// Calculate coefficients and pose
		if (face_data.shape_coefficients.empty() || face_data.expr_coefficients.empty())
		{
			m_cnn_3dmm_expr->process(face_data.cropped_img, face_data.cropped_landmarks,
				face_data.shape_coefficients, face_data.tex_coefficients,
				face_data.expr_coefficients, face_data.vecR, face_data.vecT, face_data.K);
		}

		// Calculate flipped coefficients and pose
		if (process_flipped && (face_data.shape_coefficients_flipped.empty() ||
			face_data.expr_coefficients_flipped.empty()))
		{
			// Horizontal flip the cropped image
			cv::Mat cropped_img_flipped;
			cv::flip(face_data.cropped_img, cropped_img_flipped, 1);

			// Horizontal flip the cropped landmarks
			std::vector<cv::Point> cropped_landmarks_flipped = face_data.cropped_landmarks;
			horFlipLandmarks(cropped_landmarks_flipped, cropped_img_flipped.cols);

			// Recalculate source coefficients
			m_cnn_3dmm_expr->process(cropped_img_flipped, cropped_landmarks_flipped,
				face_data.shape_coefficients_flipped,
				face_data.tex_coefficients_flipped, face_data.expr_coefficients_flipped,
				face_data.vecR_flipped, face_data.vecT_flipped, face_data.K);
		}
			
		return true;
	}

	cv::Mat FaceSwapEngineImpl::renderFaceData(const FaceData& face_data, float scale)
	{
		cv::Mat out = face_data.scaled_img.clone();
		if(scale != 1.0f)
			cv::resize(out, out, cv::Size(), scale, scale, cv::INTER_CUBIC);

		// Check if a face was detected
		if (face_data.scaled_landmarks.empty())
		{
			cv::Mat temp = out.clone();
			cv::hconcat(out, temp, out);
			cv::hconcat(out, temp, out);
			return out;
		}

		// Render landmarks and bounding box
		std::vector<cv::Point> landmarks = face_data.scaled_landmarks;
		cv::Rect bbox = face_data.scaled_bbox;
		if (scale != 1.0f)
		{
			// Scale landmarks
			for (auto&& p : landmarks)
			{
				p.x = (int)std::round(p.x * scale);
				p.y = (int)std::round(p.y * scale);
			}

			// Scale bounding box
			bbox.x = (int)std::round(face_data.scaled_bbox.x * scale);
			bbox.y = (int)std::round(face_data.scaled_bbox.y * scale);
			bbox.width = (int)std::round(face_data.scaled_bbox.width * scale);
			bbox.height = (int)std::round(face_data.scaled_bbox.height * scale);
		}
		cv::Mat landmarks_render = face_data.scaled_img.clone();
		if (scale != 1.0f)
			cv::resize(landmarks_render, landmarks_render, cv::Size(), scale, scale, cv::INTER_CUBIC);
		int thickness = int(bbox.width / 200.0f);
		render(landmarks_render, landmarks, false, cv::Scalar(0, 255, 0), thickness);
		render(landmarks_render, bbox, cv::Scalar(0, 0, 255), thickness);
		out = landmarks_render;

		// Render mesh wireframe
		cv::Mat wireframe_render = face_data.scaled_img.clone();
		if (scale != 1.0f)
			cv::resize(wireframe_render, wireframe_render, cv::Size(), scale, scale, cv::INTER_CUBIC);
		cv::Mat wireframe_render_cropped = face_data.cropped_img.clone();
		cv::Mat P = createPerspectiveProj3x4(face_data.vecR, face_data.vecT, face_data.K);
		Mesh mesh = m_basel_3dmm->sample(face_data.shape_coefficients,
			face_data.tex_coefficients, face_data.expr_coefficients);
		renderWireframe(wireframe_render_cropped, mesh, P, scale);
		wireframe_render_cropped.copyTo(wireframe_render(bbox));
		cv::hconcat(out, wireframe_render, out);

		// Render segmentation
		cv::Mat seg_render = face_data.scaled_img.clone();
		if (!face_data.scaled_seg.empty())
			renderSegmentation(seg_render, face_data.scaled_seg);
		if (scale != 1.0f)
			cv::resize(seg_render, seg_render, cv::Size(), scale, scale, cv::INTER_CUBIC);
		cv::hconcat(out, seg_render, out);

		return out;
	}

	bool FaceSwapEngineImpl::preprocessImages(FaceData& face_data)
	{
		// Calculate landmarks
		//m_sfl->clear();
		//const sfl::Frame& lmsFrame = m_sfl->addFrame(face_data.img);
		//if (lmsFrame.faces.empty()) return false;
		////std::cout << "faces found = " << lmsFrame.faces.size() << std::endl;    // Debug
		//const sfl::Face* face = lmsFrame.getFace(sfl::getMainFaceID(m_sfl->getSequence()));

		std::vector<Face> faces;
		m_lms->process(face_data.img, faces);
		if (faces.empty()) return false;
		Face& main_face = faces[getMainFaceID(faces, face_data.img.size())];
		face_data.scaled_landmarks = main_face.landmarks;

		// Calculate crop bounding box
		face_data.bbox = getFaceBBoxFromLandmarks(face_data.scaled_landmarks, face_data.img.size(), true);
		face_data.bbox.width = face_data.bbox.width / 4 * 4;    // Make sure cropped image is dividable by 4
		face_data.bbox.height = face_data.bbox.height / 4 * 4;
		
		// Inforce maximum bounding box resolution
		if (face_data.max_bbox_res > 0 && face_data.max_bbox_res < face_data.bbox.width)
		{
			float scale = (float)face_data.max_bbox_res / (float)face_data.bbox.width;

			// Scale landmarks
			for (cv::Point& p : face_data.scaled_landmarks)
			{
				p.x = (int)std::round((float)p.x * scale);
				p.y = (int)std::round((float)p.y * scale);
			}

			// Scale bounding box
			face_data.scaled_bbox.x = (int)std::round((float)face_data.bbox.x * scale);
			face_data.scaled_bbox.y = (int)std::round((float)face_data.bbox.y * scale);
			face_data.scaled_bbox.width = (int)std::round((float)face_data.bbox.width * scale);
			face_data.scaled_bbox.height = (int)std::round((float)face_data.bbox.height * scale);
			face_data.scaled_bbox.width = face_data.scaled_bbox.width / 4 * 4;    // Make sure cropped image is dividable by 4
			face_data.scaled_bbox.height = face_data.scaled_bbox.height / 4 * 4;

			// Scale images
			cv::resize(face_data.img, face_data.scaled_img, cv::Size(), scale, scale, cv::INTER_CUBIC);
			if (!face_data.seg.empty())
				cv::resize(face_data.seg, face_data.scaled_seg, cv::Size(), scale, scale, cv::INTER_CUBIC);
		}
		else
		{
			face_data.scaled_bbox = face_data.bbox;
			face_data.scaled_img = face_data.img;
			face_data.scaled_seg = face_data.seg;
		}

		// Crop landmarks
		face_data.cropped_landmarks = face_data.scaled_landmarks;
		for (cv::Point& p : face_data.cropped_landmarks)
		{
			p.x -= face_data.scaled_bbox.x;
			p.y -= face_data.scaled_bbox.y;
		}

		// Crop images
		face_data.cropped_img = face_data.scaled_img(face_data.scaled_bbox);
		if (!face_data.scaled_seg.empty()) 
			face_data.cropped_seg = face_data.scaled_seg(face_data.scaled_bbox);

		return true;
	}

}   // namespace face_swap
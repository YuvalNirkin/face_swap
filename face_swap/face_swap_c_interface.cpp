#include "face_swap/face_swap_c_interface.h"
#include <face_swap/face_swap_engine.h>

#include <iostream>	// Debug

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>	// Debug

using namespace face_swap;

std::shared_ptr<FaceSwapEngine> instance = nullptr;

int init(const char* landmarks_path, const char* model_3dmm_h5_path,
	const char * model_3dmm_dat_path, const char * reg_model_path,
	const char * reg_deploy_path, const char * reg_mean_path,
	const char * seg_model_path, const char * seg_deploy_path,
	bool generic, bool with_expr, bool with_gpu, int gpu_device_id)
{
	try
	{
		instance = FaceSwapEngine::createInstance(
			std::string(landmarks_path), std::string(model_3dmm_h5_path),
			std::string(model_3dmm_dat_path), std::string(reg_model_path),
			std::string(reg_deploy_path), std::string(reg_mean_path),
			std::string(seg_model_path), std::string(seg_deploy_path),
			generic, with_expr, with_gpu, gpu_device_id);
	}
	catch (std::exception& e)
	{
		std::cout << "face_swap error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}

cv::Mat rgba2bgr(unsigned char* img, int w, int h)
{
	cv::Mat img_rgba(h, w, CV_8UC4, img);
	cv::Mat img_bgr;
	cv::cvtColor(img_rgba, img_bgr, cv::COLOR_RGBA2BGR);
	return img_bgr;
}

void bgr2rgba(const cv::Mat& bgr, unsigned char*& rgba, int& w, int& h)
{
	// Initialize rgba array
	if (rgba != nullptr && (w != bgr.cols || h != bgr.rows))
	{
		delete[] rgba;
		rgba = nullptr;
	}
	if (rgba == nullptr)
	{
		rgba = new unsigned char[bgr.total() * 4];
		w = bgr.cols;
		h = bgr.rows;
	}

	// Convert color
	cv::Mat img_rgba(h, w, CV_8UC4, rgba);
	cv::cvtColor(bgr, img_rgba, cv::COLOR_BGR2RGBA);
}

void convert_face_data(FaceDataInterface* in_face_data, FaceData& out_face_data)
{
	if (in_face_data->img != nullptr)
		out_face_data.img = rgba2bgr(in_face_data->img, in_face_data->w, in_face_data->h);
	if (in_face_data->seg != nullptr)
		out_face_data.seg = cv::Mat(in_face_data->h, in_face_data->w, CV_8U, in_face_data->seg);
	if (in_face_data->scaled_img != nullptr)
		out_face_data.scaled_img = rgba2bgr(in_face_data->scaled_img, in_face_data->scaled_w, in_face_data->scaled_h);
	if (in_face_data->scaled_seg != nullptr)
		out_face_data.scaled_seg = cv::Mat(in_face_data->scaled_h, in_face_data->scaled_w, CV_8U, in_face_data->scaled_seg);
	if (in_face_data->cropped_img != nullptr)
		out_face_data.cropped_img = rgba2bgr(in_face_data->cropped_img, in_face_data->cropped_w, in_face_data->cropped_h);
	if (in_face_data->cropped_seg != nullptr)
		out_face_data.cropped_seg = cv::Mat(in_face_data->cropped_h, in_face_data->cropped_w, CV_8U, in_face_data->cropped_seg);
	if (in_face_data->scaled_landmarks != nullptr)
		out_face_data.scaled_landmarks.assign(((cv::Point*)in_face_data->scaled_landmarks), ((cv::Point*)in_face_data->scaled_landmarks) + 68);
	if (in_face_data->cropped_landmarks != nullptr)
		out_face_data.cropped_landmarks.assign(((cv::Point*)in_face_data->cropped_landmarks), ((cv::Point*)in_face_data->cropped_landmarks) + 68);
	if (in_face_data->bbox != nullptr)
		memcpy(&(out_face_data.bbox.x), in_face_data->bbox, 4 * sizeof(int));
	if (in_face_data->scaled_bbox != nullptr)
		memcpy(&(out_face_data.scaled_bbox.x), in_face_data->scaled_bbox, 4 * sizeof(int));
	if (in_face_data->shape_coefficients != nullptr)
		out_face_data.shape_coefficients = cv::Mat(99, 1, CV_32F, in_face_data->shape_coefficients);
	if (in_face_data->tex_coefficients != nullptr)
		out_face_data.tex_coefficients = cv::Mat(99, 1, CV_32F, in_face_data->tex_coefficients);
	if (in_face_data->expr_coefficients != nullptr)
		out_face_data.expr_coefficients = cv::Mat(29, 1, CV_32F, in_face_data->expr_coefficients);
	if (in_face_data->vecR != nullptr)
		out_face_data.vecR = cv::Mat(3, 1, CV_32F, in_face_data->vecR);
	if (in_face_data->vecT != nullptr)
		out_face_data.vecT = cv::Mat(3, 1, CV_32F, in_face_data->vecT);
	if (in_face_data->K != nullptr)
		out_face_data.K = cv::Mat(3, 3, CV_32F, in_face_data->K);
	if (in_face_data->shape_coefficients_flipped != nullptr)
		out_face_data.shape_coefficients_flipped = cv::Mat(99, 1, CV_32F, in_face_data->shape_coefficients_flipped);
	if (in_face_data->tex_coefficients_flipped != nullptr)
		out_face_data.tex_coefficients_flipped = cv::Mat(99, 1, CV_32F, in_face_data->tex_coefficients_flipped);
	if (in_face_data->expr_coefficients_flipped != nullptr)
		out_face_data.expr_coefficients_flipped = cv::Mat(29, 1, CV_32F, in_face_data->expr_coefficients_flipped);
	if (in_face_data->vecR_flipped != nullptr)
		out_face_data.vecR_flipped = cv::Mat(3, 1, CV_32F, in_face_data->vecR_flipped);
	if (in_face_data->vecT_flipped != nullptr)
		out_face_data.vecT_flipped = cv::Mat(3, 1, CV_32F, in_face_data->vecT_flipped);
	out_face_data.enable_seg = in_face_data->enable_seg;
	out_face_data.max_bbox_res = in_face_data->max_bbox_res;
}

void convert_face_data(FaceData& in_face_data, FaceDataInterface* out_face_data)
{
	if (!in_face_data.img.empty())
		bgr2rgba(in_face_data.img, out_face_data->img, out_face_data->w, out_face_data->h);
	if (!in_face_data.seg.empty())
	{
		if (out_face_data->seg == nullptr) out_face_data->seg = new unsigned char[in_face_data.seg.total()];
		memcpy(out_face_data->seg, in_face_data.seg.data, in_face_data.seg.total());
	}	
	if (!in_face_data.scaled_img.empty())
		bgr2rgba(in_face_data.scaled_img, out_face_data->scaled_img, out_face_data->scaled_w, out_face_data->scaled_h);
	if (!in_face_data.scaled_seg.empty())
	{
		if (out_face_data->scaled_seg == nullptr) out_face_data->scaled_seg = new unsigned char[in_face_data.scaled_seg.total()];
		memcpy(out_face_data->scaled_seg, in_face_data.scaled_seg.data, in_face_data.scaled_seg.total());
	}
	if (!in_face_data.cropped_img.empty())
		bgr2rgba(in_face_data.cropped_img, out_face_data->cropped_img, out_face_data->cropped_w, out_face_data->cropped_h);
	if (!in_face_data.cropped_seg.empty())
	{
		if (out_face_data->cropped_seg == nullptr) out_face_data->cropped_seg = new unsigned char[in_face_data.cropped_seg.total()];
		memcpy(out_face_data->cropped_seg, in_face_data.cropped_seg.data, in_face_data.cropped_seg.total());
	}
	if (!in_face_data.scaled_landmarks.empty())
	{
		if (out_face_data->scaled_landmarks == nullptr) out_face_data->scaled_landmarks = new int[68 * 2];
		memcpy(out_face_data->scaled_landmarks, in_face_data.scaled_landmarks.data(), 68 * 2 *sizeof(int));
	}
	if (!in_face_data.cropped_landmarks.empty())
	{
		if (out_face_data->cropped_landmarks == nullptr) out_face_data->cropped_landmarks = new int[68 * 2];
		memcpy(out_face_data->cropped_landmarks, in_face_data.cropped_landmarks.data(), 68 * 2 *sizeof(int));
	}
	if (in_face_data.bbox.width > 0 && in_face_data.bbox.height > 0)
	{
		if (out_face_data->bbox == nullptr) out_face_data->bbox = new int[4];
		memcpy(out_face_data->bbox, &(in_face_data.bbox.x), 4 * sizeof(int));
	}
	if (in_face_data.scaled_bbox.width > 0 && in_face_data.scaled_bbox.height > 0)
	{
		if (out_face_data->scaled_bbox == nullptr) out_face_data->scaled_bbox = new int[4];
		memcpy(out_face_data->scaled_bbox, &(in_face_data.scaled_bbox.x), 4 * sizeof(int));
	}
	if (!in_face_data.shape_coefficients.empty())
	{
		if (out_face_data->shape_coefficients == nullptr) out_face_data->shape_coefficients = new float[99];
		memcpy(out_face_data->shape_coefficients, in_face_data.shape_coefficients.data, 99 * sizeof(float));
	}
	if (!in_face_data.tex_coefficients.empty())
	{
		if (out_face_data->tex_coefficients == nullptr) out_face_data->tex_coefficients = new float[99];
		memcpy(out_face_data->tex_coefficients, in_face_data.tex_coefficients.data, 99 * sizeof(float));
	}
	if (!in_face_data.expr_coefficients.empty())
	{
		if (out_face_data->expr_coefficients == nullptr) out_face_data->expr_coefficients = new float[29];
		memcpy(out_face_data->expr_coefficients, in_face_data.expr_coefficients.data, 29 * sizeof(float));
	}
	if (!in_face_data.vecR.empty())
	{
		if (out_face_data->vecR == nullptr) out_face_data->vecR = new float[3];
		memcpy(out_face_data->vecR, in_face_data.vecR.data, 3 * sizeof(float));
	}
	if (!in_face_data.vecT.empty())
	{
		if (out_face_data->vecT == nullptr) out_face_data->vecT = new float[3];
		memcpy(out_face_data->vecT, in_face_data.vecT.data, 3 * sizeof(float));
	}
	if (!in_face_data.K.empty())
	{
		if (out_face_data->K == nullptr) out_face_data->K = new float[9];
		memcpy(out_face_data->K, in_face_data.K.data, 9 * sizeof(float));
	}
	if (!in_face_data.shape_coefficients_flipped.empty())
	{
		if (out_face_data->shape_coefficients_flipped == nullptr) out_face_data->shape_coefficients_flipped = new float[99];
		memcpy(out_face_data->shape_coefficients_flipped, in_face_data.shape_coefficients_flipped.data, 99 * sizeof(float));
	}
	if (!in_face_data.tex_coefficients_flipped.empty())
	{
		if (out_face_data->tex_coefficients_flipped == nullptr) out_face_data->tex_coefficients_flipped = new float[99];
		memcpy(out_face_data->tex_coefficients_flipped, in_face_data.tex_coefficients_flipped.data, 99 * sizeof(float));
	}
	if (!in_face_data.expr_coefficients_flipped.empty())
	{
		if (out_face_data->expr_coefficients_flipped == nullptr) out_face_data->expr_coefficients_flipped = new float[29];
		memcpy(out_face_data->expr_coefficients_flipped, in_face_data.expr_coefficients_flipped.data, 29 * sizeof(float));
	}
	if (!in_face_data.vecR_flipped.empty())
	{
		if (out_face_data->vecR_flipped == nullptr) out_face_data->vecR_flipped = new float[3];
		memcpy(out_face_data->vecR_flipped, in_face_data.vecR_flipped.data, 3 * sizeof(float));
	}
	if (!in_face_data.vecT_flipped.empty())
	{
		if (out_face_data->vecT_flipped == nullptr) out_face_data->vecT_flipped = new float[3];
		memcpy(out_face_data->vecT_flipped, in_face_data.vecT_flipped.data, 3 * sizeof(float));
	}
	out_face_data->enable_seg = in_face_data.enable_seg;
	out_face_data->max_bbox_res = in_face_data.max_bbox_res;
}

int process(FaceDataInterface* face_data)
{
	try
	{
		// Validate initialization
		if (instance == nullptr)
		{
			std::cout << "Error: Face swap must be initialized first!" << std::endl;
			return 1;
		}

		// Process face data
		FaceData out_face_data;
		convert_face_data(face_data, out_face_data);
		instance->process(out_face_data);
		convert_face_data(out_face_data, face_data);
	}
	catch (std::exception& e)
	{
		std::cout << "face_swap error: " << e.what() << std::endl;
		return 1;
	}
	
	return 0;
}

int swap(FaceDataInterface* src_data, FaceDataInterface* tgt_data, unsigned char* out)
{
	try
	{
		FaceData out_src_data, out_tgt_data;
		convert_face_data(src_data, out_src_data);
		convert_face_data(tgt_data, out_tgt_data);
		cv::Mat render_img = instance->swap(out_src_data, out_tgt_data);
		convert_face_data(out_src_data, src_data);
		convert_face_data(out_tgt_data, tgt_data);
		bgr2rgba(render_img, out, render_img.cols, render_img.rows);
	}
	catch (std::exception& e)
	{
		std::cout << "face_swap error: " << e.what() << std::endl;
		return 1;
	}
	
	return 0;
}

#ifndef FACE_SWAP_C_INTERFACE_H
#define FACE_SWAP_C_INTERFACE_H

#include "face_swap/face_swap_export.h"

extern "C" {

	struct FaceDataInterface
	{
		// Input
		unsigned char* img;
		unsigned char* seg;
		int w, h;

		// Intermediate pipeline data
		unsigned char* scaled_img;
		unsigned char* scaled_seg;
		int scaled_w, scaled_h;
		unsigned char* cropped_img;
		unsigned char* cropped_seg;
		int cropped_w, cropped_h;
		int* scaled_landmarks;
		int* cropped_landmarks;
		int* bbox;
		int* scaled_bbox;
		float *shape_coefficients, *tex_coefficients, *expr_coefficients;
		float *vecR, *vecT, *K;

		// Flipped image data
		float *shape_coefficients_flipped, *tex_coefficients_flipped, *expr_coefficients_flipped;
		float *vecR_flipped, *vecT_flipped;

		// Processing parameters
		bool enable_seg = true;
		int max_bbox_res = 0;
	};

	FACE_SWAP_EXPORT int init(const char* landmarks_path, const char* model_3dmm_h5_path,
		const char* model_3dmm_dat_path, const char* reg_model_path,
		const char* reg_deploy_path, const char* reg_mean_path,
		const char* seg_model_path, const char* seg_deploy_path,
		bool generic = false, bool with_expr = true, bool with_gpu = true,
		int gpu_device_id = 0);

	FACE_SWAP_EXPORT int process(FaceDataInterface* face_data);

	FACE_SWAP_EXPORT int swap(FaceDataInterface* src_data, FaceDataInterface* tgt_data,
		unsigned char* out);
}


#endif // FACE_SWAP_C_INTERFACE_H

/** @file
@brief Face segmentation utility functions.
*/

#ifndef FACE_SWAP_SEGMENTATION_UTILITIES_H
#define FACE_SWAP_SEGMENTATION_UTILITIES_H

#include "face_swap/face_swap_export.h"

// OpenCV
#include <opencv2/core.hpp>

namespace face_swap
{
	FACE_SWAP_EXPORT void removeSmallerComponents(cv::Mat& seg);

	FACE_SWAP_EXPORT void smoothFlaws(cv::Mat& seg, int smooth_iterations = 1, int smooth_kernel_radius = 2);

	FACE_SWAP_EXPORT void fillHoles(cv::Mat& seg);

	FACE_SWAP_EXPORT void postprocessSegmentation(cv::Mat& seg, bool disconnected = true,
		bool holes = true, bool smooth = true, int smooth_iterations = 1,
		int smooth_kernel_radius = 2);

}   // namespace face_swap

#endif	// FACE_SWAP_SEGMENTATION_UTILITIES_H

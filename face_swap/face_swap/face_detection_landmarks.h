#ifndef FACE_SWAP_FACE_DETECTION_LANDMARKS_H
#define FACE_SWAP_FACE_DETECTION_LANDMARKS_H

#include "face_swap/face_swap_export.h"

// std
#include <string>
#include <list>
#include <memory>

// OpenCV
#include <opencv2/core.hpp>


namespace face_swap
{
	/** @brief Represents a face detected in a frame.
	*/
    struct Face
    {
		cv::Rect bbox;						///< Bounding box.
        std::vector<cv::Point> landmarks;	///< Face landmarks.
    };

	/** @brief This class provide face detection and landmarks extraction
	functionality for single images.
	*/
	class FACE_SWAP_EXPORT FaceDetectionLandmarks
	{
	public:
		/** @brief Process a frame.
		@param frame The frame to process [BGR].
		@param faces The output faces detected in the frame.
		*/
		virtual void process(const cv::Mat& frame, std::vector<Face>& faces) = 0;

		/** @brief Create an instance initialized with a landmarks model file.
		@param landmarks_path Path to the landmarks model file (.dat).
		*/
		static std::shared_ptr<FaceDetectionLandmarks> create(const std::string& landmarks_path);
	};

}   // namespace face_swap

#endif	// FACE_SWAP_FACE_DETECTION_LANDMARKS_H

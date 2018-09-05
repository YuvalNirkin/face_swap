#include "face_swap/face_detection_landmarks.h"

// std
#include <exception>

// OpenCV
#include <opencv2/imgproc.hpp>

// dlib
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>

using std::string;
using std::runtime_error;

namespace face_swap
{
	void dlib_obj_to_points(const dlib::full_object_detection& obj,
		std::vector<cv::Point>& points)
	{
		points.resize(obj.num_parts());
		for (unsigned long i = 0; i < obj.num_parts(); ++i)
		{
			cv::Point& p = points[i];
			const dlib::point& obj_p = obj.part(i);
			p.x = (float)obj_p.x();
			p.y = (float)obj_p.y();
		}
	}

	class FaceDetectionLandmarksImpl : public FaceDetectionLandmarks
	{
	private:
		// dlib
		dlib::frontal_face_detector m_detector;
		dlib::shape_predictor m_landmarks_model;

	public:
		FaceDetectionLandmarksImpl(const std::string& landmarks_path)
		{
			// Face detector for finding bounding boxes for each face in an image
			m_detector = dlib::get_frontal_face_detector();

			// Shape predictor for finding landmark positions given an image and face bounding box.
			if (landmarks_path.empty()) return;
			dlib::deserialize(landmarks_path) >> m_landmarks_model;
		}
		
		void process(const cv::Mat& frame, std::vector<Face>& faces)
		{
			// Convert OpenCV's mat to dlib format 
			dlib::cv_image<dlib::bgr_pixel> dlib_frame(frame);

			// Detect bounding boxes around all the faces in the image.
			std::vector<dlib::rectangle> dlib_rects = m_detector(dlib_frame);

			// Extract landmarks for each face we detected.
			std::vector<dlib::full_object_detection> shapes;
			for (size_t i = 0; i < dlib_rects.size(); ++i)
			{
				faces.push_back(Face());
				Face& curr_face = faces.back();
				dlib::rectangle& dlib_rect = dlib_rects[i];

				// Set landmarks
				dlib::full_object_detection shape = m_landmarks_model(dlib_frame, dlib_rect);
				dlib_obj_to_points(shape, curr_face.landmarks);

				// Set face bounding box
				curr_face.bbox.x = (int)dlib_rect.left();
				curr_face.bbox.y = (int)dlib_rect.top();
				curr_face.bbox.width = (int)dlib_rect.width();
				curr_face.bbox.height = (int)dlib_rect.height();
			}
		}
	};

	std::shared_ptr<FaceDetectionLandmarks> FaceDetectionLandmarks::create(
		const std::string& landmarks_path)
	{
		return std::make_shared<FaceDetectionLandmarksImpl>(landmarks_path);
	}

}   // namespace face_swap
#include "face_swap/landmarks_utilities.h"

// std
#include <map>

// OpenCV
#include <opencv2/imgproc.hpp>

using std::runtime_error;

const float MAX_FACE_ANGLE = 75.0f;

namespace face_swap
{
	void render(cv::Mat & img, const std::vector<cv::Point>& landmarks,
		bool drawLabels, const cv::Scalar & color, int thickness)
	{
		if (landmarks.size() == 68)
		{
			for (size_t i = 1; i <= 16; ++i)
				cv::line(img, landmarks[i], landmarks[i - 1], color, thickness);

			for (size_t i = 28; i <= 30; ++i)
				cv::line(img, landmarks[i], landmarks[i - 1], color, thickness);

			for (size_t i = 18; i <= 21; ++i)
				cv::line(img, landmarks[i], landmarks[i - 1], color, thickness);

			for (size_t i = 23; i <= 26; ++i)
				cv::line(img, landmarks[i], landmarks[i - 1], color, thickness);

			for (size_t i = 31; i <= 35; ++i)
				cv::line(img, landmarks[i], landmarks[i - 1], color, thickness);
			cv::line(img, landmarks[30], landmarks[35], color, thickness);

			for (size_t i = 37; i <= 41; ++i)
				cv::line(img, landmarks[i], landmarks[i - 1], color, thickness);
			cv::line(img, landmarks[36], landmarks[41], color, thickness);

			for (size_t i = 43; i <= 47; ++i)
				cv::line(img, landmarks[i], landmarks[i - 1], color, thickness);
			cv::line(img, landmarks[42], landmarks[47], color, thickness);

			for (size_t i = 49; i <= 59; ++i)
				cv::line(img, landmarks[i], landmarks[i - 1], color, thickness);
			cv::line(img, landmarks[48], landmarks[59], color, thickness);

			for (size_t i = 61; i <= 67; ++i)
				cv::line(img, landmarks[i], landmarks[i - 1], color, thickness);
			cv::line(img, landmarks[60], landmarks[67], color, thickness);
		}
		else
		{
			for (size_t i = 0; i < landmarks.size(); ++i)
				cv::circle(img, landmarks[i], thickness, color, -1);
		}

		if (drawLabels)
		{
			// Add labels
			for (size_t i = 0; i < landmarks.size(); ++i)
				cv::putText(img, std::to_string(i), landmarks[i],
					cv::FONT_HERSHEY_PLAIN, 0.5, color, thickness);
		}
	}

	void render(cv::Mat& img, const cv::Rect& bbox, const cv::Scalar& color,
		int thickness)
	{
		cv::rectangle(img, bbox, color, thickness);
	}

	void render(cv::Mat& img, const Face& face, bool drawLabels,
		const cv::Scalar& bbox_color, const cv::Scalar& landmarks_color, int thickness,
		double fontScale)
	{
		render(img, face.bbox, bbox_color, thickness);
		render(img, face.landmarks, drawLabels, landmarks_color, thickness);
	}

	void render(cv::Mat& img, const std::vector<Face>& faces, bool drawLabels,
		const cv::Scalar& bbox_color, const cv::Scalar& landmarks_color, int thickness,
		double fontScale)
	{
		for (auto& face :faces)
			render(img, face, drawLabels, bbox_color, landmarks_color, thickness,
				fontScale);
	}

	int getMainFaceID(const std::vector<Face>& faces, const cv::Size& frame_size)
	{
		if (faces.empty()) return -1;

		std::vector<float> face_scores;
		cv::Point2f center(frame_size.width*0.5f, frame_size.height*0.5f);
		float face_dist, face_size;
		float central_ratio, size_ratio;
		float score;

		// Calculate frame max distance and size
		float max_dist = 0.25f*std::sqrt(frame_size.width * frame_size.width +
			frame_size.height * frame_size.height);
		float max_size = 0.25f*(frame_size.width + frame_size.height);

		// For each face detected in the frame
		for (const Face& face : faces)
		{
			// Calculate center distance
			cv::Point tl = face.bbox.tl();
			cv::Point br = face.bbox.br();
			cv::Point2f pos((tl.x + br.x)*0.5f, (tl.y + br.y)*0.5f);
			face_dist = (float)cv::norm(pos - center);

			// Calculate face size
			face_size = (face.bbox.width + face.bbox.height)*0.5f;

			// Calculate central ratio
			if (max_dist < 1e-6f) central_ratio = 1.0f;
			else central_ratio = (1 - face_dist / max_dist);
			central_ratio = std::min(std::max(0.0f, central_ratio), 1.0f);

			// Calculate size ratio
			if (max_size < 1e-6f) size_ratio = 1.0f;
			else size_ratio = face_size / max_size;
			size_ratio = std::min(std::max(0.0f, size_ratio), 1.0f);

			// Add face score
			score = (central_ratio + size_ratio) * 0.5f;
			face_scores.push_back(score);
		}

		return std::distance(face_scores.begin(),
			std::max_element(face_scores.begin(), face_scores.end()));
	}

    cv::Point2f getFaceLeftEye(const std::vector<cv::Point>& landmarks)
    {
        if (landmarks.size() != 68) return cv::Point2f();

        cv::Point2f left_eye(0, 0);
        for (size_t i = 42; i <= 47; ++i)
            left_eye += cv::Point2f(landmarks[i]);

        return (left_eye / 6);
    }

    cv::Point2f getFaceRightEye(const std::vector<cv::Point>& landmarks)
    {
        if (landmarks.size() != 68) return cv::Point2f();

        cv::Point2f right_eye(0, 0);
        for (size_t i = 36; i <= 41; ++i)
            right_eye += cv::Point2f(landmarks[i]);

        return (right_eye / 6);
    }

    float getFaceApproxVertAngle(const std::vector<cv::Point>& landmarks)
    {
        if (landmarks.size() != 68) return 0;
        cv::Point2f left_eye = getFaceLeftEye(landmarks);
        cv::Point2f right_eye = getFaceRightEye(landmarks);
        cv::Point2f x1 = landmarks[0], x2 = landmarks[16];
        cv::Point2f v = x2 - x1;
        cv::Point2f right_eye_dir = x1 - right_eye;
        cv::Point2f left_eye_dir = x1 - left_eye;
        float x12_dist = cv::norm(v);
        float d1 = v.cross(right_eye_dir) / x12_dist;
        float d2 = v.cross(left_eye_dir) / x12_dist;
        float d = (d1 + d2)*0.5f / cv::norm(left_eye - right_eye);
        return d * (2 * MAX_FACE_ANGLE) * (CV_PI / 180.0f);
    }

    float getFaceApproxHorAngle(const std::vector<cv::Point>& landmarks)
    {
        if (landmarks.size() != 68) return 0;
        const float max_angle = 75.0f;

        const cv::Point& center = landmarks[27];
        const cv::Point& left_eye = landmarks[42];
        const cv::Point& right_eye = landmarks[39];
        float left_dist = cv::norm(center - left_eye);
        float right_dist = cv::norm(center - right_eye);
        float d = (left_dist / (left_dist + right_dist) - 0.5f);

        return d * (2 * MAX_FACE_ANGLE) * (CV_PI / 180.0f);
    }

    float getFaceApproxTiltAngle(const std::vector<cv::Point>& landmarks)
    {
        if (landmarks.size() != 68) return 0;

        cv::Point2f left_eye = getFaceLeftEye(landmarks);
        cv::Point2f right_eye = getFaceRightEye(landmarks);
        cv::Point2f v = left_eye - right_eye;
        return atan2(v.y, v.x);
    }

    cv::Point3f getFaceApproxEulerAngles(const std::vector<cv::Point>& landmarks)
    {
        float x = getFaceApproxVertAngle(landmarks);
        float y = getFaceApproxHorAngle(landmarks);
        float z = getFaceApproxTiltAngle(landmarks);

        return cv::Point3f(x, y, z);
    }

    cv::Rect getFaceBBoxFromLandmarks(const std::vector<cv::Point>& landmarks,
        const cv::Size& frameSize, bool square)
    {
        int xmin(std::numeric_limits<int>::max()), ymin(std::numeric_limits<int>::max()),
            xmax(-1), ymax(-1), sumx(0), sumy(0);
        for (const cv::Point& p : landmarks)
        {
            xmin = std::min(xmin, p.x);
            ymin = std::min(ymin, p.y);
            xmax = std::max(xmax, p.x);
            ymax = std::max(ymax, p.y);
            sumx += p.x;
            sumy += p.y;
        }

        int width = xmax - xmin + 1;
        int height = ymax - ymin + 1;
        int centerx = (xmin + xmax) / 2;
        int centery = (ymin + ymax) / 2;
        int avgx = (int)std::round(sumx / landmarks.size());
        int avgy = (int)std::round(sumy / landmarks.size());
        int devx = centerx - avgx;
        int devy = centery - avgy;
        int dleft = (int)std::round(0.1*width) + abs(devx < 0 ? devx : 0);
        int dtop = (int)std::round(height*(std::max(float(width) / height, 1.0f) * 2 - 1)) + abs(devy < 0 ? devy : 0);
        int dright = (int)std::round(0.1*width) + abs(devx > 0 ? devx : 0);
        int dbottom = (int)std::round(0.1*height) + abs(devy > 0 ? devy : 0);

        // Limit to frame boundaries
        xmin = std::max(0, xmin - dleft);
        ymin = std::max(0, ymin - dtop);
        xmax = std::min((int)frameSize.width - 1, xmax + dright);
        ymax = std::min((int)frameSize.height - 1, ymax + dbottom);

        // Make square
        if (square)
        {
            int sq_width = std::max(xmax - xmin + 1, ymax - ymin + 1);
            centerx = (xmin + xmax) / 2;
            centery = (ymin + ymax) / 2;
            xmin = centerx - ((sq_width - 1) / 2);
            ymin = centery - ((sq_width - 1) / 2);
            xmax = xmin + sq_width - 1;
            ymax = ymin + sq_width - 1;

            // Limit to frame boundaries
            xmin = std::max(0, xmin);
            ymin = std::max(0, ymin);
            xmax = std::min((int)frameSize.width - 1, xmax);
            ymax = std::min((int)frameSize.height - 1, ymax);
        }

        return cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
    }

}   // namespace face_swap


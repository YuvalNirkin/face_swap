#include "face_swap/utilities.h"
#include <iostream>	// Debug
#include <fstream>

// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>	// Debug


#ifdef WITH_PROTOBUF
// Boost
#include <boost/filesystem.hpp>
#include <boost/timer/timer.hpp>	// Debug

// Protobuf
#include "face_data.pb.h"

using namespace boost::filesystem;

#endif // WITH_PROTOBUF

namespace face_swap
{
    cv::Mat euler2RotMat(float x, float y, float z)
    {
        // Calculate rotation about x axis
        cv::Mat R_x = (cv::Mat_<float>(3, 3) <<
            1, 0, 0,
            0, cos(x), -sin(x),
            0, sin(x), cos(x));

        // Calculate rotation about y axis
        cv::Mat R_y = (cv::Mat_<float>(3, 3) <<
            cos(y), 0, sin(y),
            0, 1, 0,
            -sin(y), 0, cos(y));

        // Calculate rotation about z axis
        cv::Mat R_z = (cv::Mat_<float>(3, 3) <<
            cos(z), -sin(z), 0,
            sin(z), cos(z), 0,
            0, 0, 1);

        // Combined rotation matrix
        return  (R_z * R_x * R_y);
    }

    cv::Mat euler2RotMat(const cv::Mat& euler)
    {
        return euler2RotMat(euler.at<float>(0), euler.at<float>(1), euler.at<float>(2));
    }

    cv::Mat createModelView(const cv::Mat & euler, const cv::Mat & translation)
    {
        cv::Mat MV = cv::Mat_<float>::eye(4, 4);
        cv::Mat R = euler2RotMat(euler);
        R.copyTo(MV(cv::Rect(0, 0, 3, 3)));
        //MV(cv::Rect(0, 0, 3, 3)) = euler2RotMat(euler);
        MV.at<float>(0, 3) = translation.at<float>(0);
        MV.at<float>(1, 3) = translation.at<float>(1);
        return MV;
    }

    cv::Mat createOrthoProj4x4(const cv::Mat & euler, const cv::Mat & translation, 
        int width, int height)
    {
        cv::Mat M = createModelView(euler, translation);
        cv::Mat V = (cv::Mat_<float>(4, 4) << 
            width / 2.0f, 0.0f, 0.0f, width / 2.0f,
            0.0f, -height / 2.0f, 0.0f, -height / 2.0f + height,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f);

        return (V * M);
    }

    cv::Mat createOrthoProj3x4(const cv::Mat & euler, const cv::Mat & translation, 
        int width, int height)
    {
        cv::Mat P_4x4 = createOrthoProj4x4(euler, translation, width, height);
        cv::Mat P_3x4 = P_4x4.rowRange(0, 3);
        P_3x4.at<float>(2, 0) = 0.0f;
        P_3x4.at<float>(2, 1) = 0.0f;
        P_3x4.at<float>(2, 2) = 0.0f;
        P_3x4.at<float>(2, 3) = 1.0f;

        return P_3x4;
    }

    cv::Mat createPerspectiveProj3x4(const cv::Mat & euler,
        const cv::Mat & translation, const cv::Mat & K)
    {
        cv::Mat R(3, 3, CV_32F);
        cv::Rodrigues(euler, R);
        cv::Mat RT;
        cv::hconcat(R, translation, RT);
        cv::Mat P = K*RT;
        return P;
    }

	/*cv::Mat createPerspectiveProj4x4(const cv::Mat& euler,
		const cv::Mat& translation, const cv::Mat& K)
	{
		cv::Mat R(3, 3, CV_32F);
		cv::Rodrigues(euler, R);
		cv::Mat P = cv::Mat::eye(4, 4, CV_32F);
		R.copyTo(P(cv::Rect(0, 0, 3, 3)));
		translation.copyTo(P(cv::Rect(3, 0, 1, translation.total())));
		return P;
	}*/

	cv::Mat refineMask(const cv::Mat& img, const cv::Mat& mask)
	{
		// Erode mask
		cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3));
		cv::Mat eroded_mask;
		cv::erode(mask, eroded_mask, kernel, cv::Point(-1, -1), 5);

		// Create markers
		cv::Mat markers = cv::Mat::zeros(mask.size(), CV_32S);
		int* markers_data = (int*)markers.data;
		unsigned char* eroded_mask_data = eroded_mask.data;
		for (int r = 0; r < mask.rows; ++r)
		{
			const unsigned char* mask_data = mask.ptr<uchar>(r);
			for (int c = 0; c < mask.cols; ++c)
			{
				if (*eroded_mask_data++ >= 128)
					*markers_data = 1;	// 1
				else if (*mask_data < 128)
					*markers_data = 2;	// 2
				++markers_data;
				++mask_data;
			}
		}

		/// Debug ///
		/*cv::Mat debug_eroded_mask = img.clone();
		renderSegmentation(debug_eroded_mask, eroded_mask);
		cv::Mat debug_mask = img.clone();
		renderSegmentation(debug_mask, mask);
		cv::imshow("debug_eroded_mask", debug_eroded_mask);
		cv::imshow("debug_mask", debug_mask);*/
		//cv::waitKey(0);
		/////////////

		/// Debug ///
		/*{
			cv::Mat debug_markers = cv::Mat::zeros(markers.size(), CV_8UC1);
			markers.convertTo(debug_markers, CV_8UC1);
			cv::bitwise_not(debug_markers, debug_markers);
			cv::imshow("debug_markers_before", debug_markers);
		}*/
		/////////////

		// Apply watershed
		cv::watershed(img, markers);

		/// Debug ///
		/*{
			cv::Mat debug_markers = cv::Mat::zeros(markers.size(), CV_8UC1);
			markers.convertTo(debug_markers, CV_8UC1);
			cv::bitwise_not(debug_markers, debug_markers);
			cv::imshow("debug_markers_after", debug_markers);
		}*/
		/////////////

		// Output refined mask
		cv::Mat out_mask(mask.size(), CV_8U);
		markers_data = (int*)markers.data;
		unsigned char* out_mask_data = out_mask.data;
		for (int i = 0; i < out_mask.total(); ++i)
		{
			if (*markers_data++ == 1)
				*out_mask_data++ = 255;
			else *out_mask_data++ = 0;
		}

		/// Debug ///
		/*cv::Mat debug_out_mask = img.clone();
		renderSegmentation(debug_out_mask, out_mask);
		cv::imshow("debug_out_mask", debug_out_mask);
		cv::waitKey(0);*/
		/////////////

		return out_mask;
	}

	void horFlipLandmarks(std::vector<cv::Point>& landmarks, int width)
	{
		// Invert X coordinates
		for (cv::Point& p : landmarks)
			p.x = width - p.x;

		// Jaw
		for (int i = 0; i <= 7; ++i)
			std::swap(landmarks[i], landmarks[16 - i]);

		// Eyebrows
		for (int i = 17; i <= 21; ++i)
			std::swap(landmarks[i], landmarks[43 - i]);

		// Nose
		std::swap(landmarks[31], landmarks[35]);
		std::swap(landmarks[32], landmarks[34]);

		// Eyes
		std::swap(landmarks[36], landmarks[45]);
		std::swap(landmarks[37], landmarks[44]);
		std::swap(landmarks[38], landmarks[43]);
		std::swap(landmarks[39], landmarks[42]);
		std::swap(landmarks[40], landmarks[47]);
		std::swap(landmarks[41], landmarks[46]);

		// Mouth Outer
		std::swap(landmarks[48], landmarks[54]);
		std::swap(landmarks[49], landmarks[53]);
		std::swap(landmarks[50], landmarks[52]);
		std::swap(landmarks[59], landmarks[55]);
		std::swap(landmarks[58], landmarks[56]);

		// Mouth inner
		std::swap(landmarks[60], landmarks[64]);
		std::swap(landmarks[61], landmarks[63]);
		std::swap(landmarks[67], landmarks[65]);
	}

	void generateTexture(const Mesh& mesh, const cv::Mat& img,
		const cv::Mat& seg, const cv::Mat& vecR, const cv::Mat& vecT,
		const cv::Mat& K, cv::Mat& tex, cv::Mat& uv)
	{
		// Resize images to power of 2 size
		cv::Size tex_size(nextPow2(img.cols), nextPow2(img.rows));
		cv::Mat img_scaled, seg_scaled;
		cv::resize(img, img_scaled, tex_size, 0.0, 0.0, cv::INTER_CUBIC);
		if (!seg.empty())
			cv::resize(seg, seg_scaled, tex_size, 0.0, 0.0, cv::INTER_NEAREST);

		// Combine image and segmentation into one 4 channel texture
		if (!seg.empty())
		{
			std::vector<cv::Mat> channels;
			cv::split(img, channels);
			channels.push_back(seg);
			cv::merge(channels, tex);
		}
		else tex = img_scaled;

		uv = generateTextureCoordinates(mesh, img.size(), vecR, vecT, K);
	}

	cv::Mat generateTextureCoordinates(
		const Mesh& mesh, const cv::Size& img_size,
		const cv::Mat & vecR, const cv::Mat & vecT, const cv::Mat & K)
	{
		cv::Mat P = createPerspectiveProj3x4(vecR, vecT, K);
		cv::Mat pts_3d;
		cv::vconcat(mesh.vertices.t(), cv::Mat::ones(1, mesh.vertices.rows, CV_32F), pts_3d);
		cv::Mat proj = P * pts_3d;

		// Normalize projected points
		cv::Mat uv(mesh.vertices.rows, 2, CV_32F);
		float* uv_data = (float*)uv.data;
		float z;
		for (int i = 0; i < uv.rows; ++i)
		{
			z = proj.at<float>(2, i);
			*uv_data++ = proj.at<float>(0, i) / (z * img_size.width);
			*uv_data++ = proj.at<float>(1, i) / (z * img_size.height);
		}

		return uv;
	}

	cv::Mat blend(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask)
	{
		// Find center point
		int minc = std::numeric_limits<int>::max(), minr = std::numeric_limits<int>::max();
		int maxc = 0, maxr = 0;
		unsigned char* mask_data = mask.data;
		for (int r = 0; r < src.rows; ++r)
			for (int c = 0; c < src.cols; ++c)
			{
				if (*mask_data++ < 255) continue;
				minc = std::min(c, minc);
				minr = std::min(r, minr);
				maxc = std::max(c, maxc);
				maxr = std::max(r, maxr);
			}
		if (minc >= maxc || minr >= maxr) return cv::Mat();
		cv::Point center((minc + maxc) / 2, (minr + maxr) / 2);

		/// Debug ///
		//cv::Mat out = src.clone();
		//cv::rectangle(out, cv::Point(minc, minr), cv::Point(maxc, maxr), cv::Scalar(255, 0, 0));
		//cv::imshow("target", out);
		//cv::imshow("mask", mask);
		//cv::waitKey(0);
		/////////////

		// Do blending
		cv::Mat blend;
		cv::seamlessClone(src, dst, mask, center, blend, cv::NORMAL_CLONE);

		return blend;
	}

	bool readFaceData(const std::string& img_path, FaceData& face_data)
	{
		// Read image
		face_data.img = cv::imread(img_path);

#ifdef WITH_PROTOBUF
		// Check if a cache file exists
		path cache_path = path(img_path).replace_extension("fs");
		if (!is_regular_file(cache_path)) return false;

		// Read from file
		io::FaceData io_face_data;
		std::ifstream input(cache_path.string(), std::ifstream::binary);
		io_face_data.ParseFromIstream(&input);

		// cropped_seg
		if (io_face_data.has_cropped_seg())
		{
			const std::string& tmp = io_face_data.cropped_seg();
			std::vector<uchar> buf((const uchar*)tmp.data(), (const uchar*)tmp.data() + tmp.size());
			face_data.cropped_seg = cv::imdecode(buf, -1);
		}

		// scaled_landmarks
		if (io_face_data.scaled_landmarks_size() > 0)
		{
			const cv::Point* data = (const cv::Point*)io_face_data.scaled_landmarks().data();
			face_data.scaled_landmarks.assign(data, data + 68);
		}

		// bbox
		if (io_face_data.bbox_size() > 0)
			memcpy(&face_data.bbox.x, io_face_data.bbox().data(), io_face_data.bbox_size() * sizeof(int));

		// shape_coefficients
		if (io_face_data.shape_coefficients_size() > 0)
		{
			face_data.shape_coefficients.create(io_face_data.shape_coefficients_size(), 1, CV_32F);
			memcpy(face_data.shape_coefficients.data, io_face_data.shape_coefficients().data(),
				io_face_data.shape_coefficients_size() * sizeof(float));
		}

		// tex_coefficients
		if (io_face_data.tex_coefficients_size() > 0)
		{
			face_data.tex_coefficients.create(io_face_data.tex_coefficients_size(), 1, CV_32F);
			memcpy(face_data.tex_coefficients.data, io_face_data.tex_coefficients().data(),
				io_face_data.tex_coefficients_size() * sizeof(float));
		}

		// expr_coefficients
		if (io_face_data.expr_coefficients_size() > 0)
		{
			face_data.expr_coefficients.create(io_face_data.expr_coefficients_size(), 1, CV_32F);
			memcpy(face_data.expr_coefficients.data, io_face_data.expr_coefficients().data(),
				io_face_data.expr_coefficients_size() * sizeof(float));
		}

		// vecR
		if (io_face_data.vecr_size() > 0)
		{
			face_data.vecR.create(io_face_data.vecr_size(), 1, CV_32F);
			memcpy(face_data.vecR.data, io_face_data.vecr().data(),
				io_face_data.vecr_size() * sizeof(float));
		}

		// vecT
		if (io_face_data.vect_size() > 0)
		{
			face_data.vecT.create(io_face_data.vect_size(), 1, CV_32F);
			memcpy(face_data.vecT.data, io_face_data.vect().data(),
				io_face_data.vect_size() * sizeof(float));
		}

		// K
		if (io_face_data.k_size() > 0)
		{
			face_data.K.create(3, 3, CV_32F);
			memcpy(face_data.K.data, io_face_data.k().data(),
				io_face_data.k_size() * sizeof(float));
		}

		// shape_coefficients_flipped
		if (io_face_data.shape_coefficients_flipped_size() > 0)
		{
			face_data.shape_coefficients_flipped.create(io_face_data.shape_coefficients_flipped_size(), 1, CV_32F);
			memcpy(face_data.shape_coefficients_flipped.data, io_face_data.shape_coefficients_flipped().data(),
				io_face_data.shape_coefficients_flipped_size() * sizeof(float));
		}

		// tex_coefficients_flipped
		if (io_face_data.tex_coefficients_flipped_size() > 0)
		{
			face_data.tex_coefficients_flipped.create(io_face_data.tex_coefficients_flipped_size(), 1, CV_32F);
			memcpy(face_data.tex_coefficients_flipped.data, io_face_data.tex_coefficients_flipped().data(),
				io_face_data.tex_coefficients_flipped_size() * sizeof(float));
		}

		// expr_coefficients_flipped
		if (io_face_data.expr_coefficients_flipped_size() > 0)
		{
			face_data.expr_coefficients_flipped.create(io_face_data.expr_coefficients_flipped_size(), 1, CV_32F);
			memcpy(face_data.expr_coefficients_flipped.data, io_face_data.expr_coefficients_flipped().data(),
				io_face_data.expr_coefficients_flipped_size() * sizeof(float));
		}

		// vecR_flipped
		if (io_face_data.vecr_flipped_size() > 0)
		{
			face_data.vecR_flipped.create(io_face_data.vecr_flipped_size(), 1, CV_32F);
			memcpy(face_data.vecR_flipped.data, io_face_data.vecr_flipped().data(),
				io_face_data.vecr_flipped_size() * sizeof(float));
		}

		// vecT_flipped
		if (io_face_data.vect_flipped_size() > 0)
		{
			face_data.vecT_flipped.create(io_face_data.vect_flipped_size(), 1, CV_32F);
			memcpy(face_data.vecT_flipped.data, io_face_data.vect_flipped().data(),
				io_face_data.vect_flipped_size() * sizeof(float));
		}

		// enable_seg
		if (io_face_data.has_enable_seg())
			face_data.enable_seg = io_face_data.enable_seg();

		// max_bbox_res
		if (io_face_data.has_max_bbox_res())
			face_data.max_bbox_res = io_face_data.max_bbox_res();

		/////
		// Reconstruct the rest of the fields
		/////
		// Inforce maximum bounding box resolution
		if (face_data.max_bbox_res > 0 && face_data.max_bbox_res < face_data.bbox.width)
		{
			float scale = (float)face_data.max_bbox_res / (float)face_data.bbox.width;

			// Scale bounding box
			face_data.scaled_bbox.x = (int)std::round((float)face_data.bbox.x * scale);
			face_data.scaled_bbox.y = (int)std::round((float)face_data.bbox.y * scale);
			face_data.scaled_bbox.width = (int)std::round((float)face_data.bbox.width * scale);
			face_data.scaled_bbox.height = (int)std::round((float)face_data.bbox.height * scale);
			face_data.scaled_bbox.width = face_data.scaled_bbox.width / 4 * 4;    // Make sure cropped image is dividable by 4
			face_data.scaled_bbox.height = face_data.scaled_bbox.height / 4 * 4;

			// Scale images
			cv::resize(face_data.img, face_data.scaled_img, cv::Size(), scale, scale, cv::INTER_CUBIC);
		}
		else
		{
			face_data.scaled_bbox = face_data.bbox;
			face_data.scaled_img = face_data.img;
		}

		// Crop landmarks
		face_data.cropped_landmarks = face_data.scaled_landmarks;
		for (cv::Point& p : face_data.cropped_landmarks)
		{
			p.x -= face_data.scaled_bbox.x;
			p.y -= face_data.scaled_bbox.y;
		}

		// cropped_img
		face_data.cropped_img = face_data.scaled_img(face_data.scaled_bbox);

		// scaled_seg
		face_data.scaled_seg = cv::Mat::zeros(face_data.scaled_img.size(), CV_8U);
		face_data.cropped_seg.copyTo(face_data.scaled_seg(face_data.scaled_bbox));

		return true;

#endif // WITH_PROTOBUF	
		return false;
	}

	bool writeFaceData(const std::string& img_path,
		const FaceData& face_data, bool overwrite)
	{
#ifdef WITH_PROTOBUF
		io::FaceData io_face_data;

		// Check if a cache file exists
		path cache_path = path(img_path).replace_extension("fs");
		if (!overwrite && is_regular_file(cache_path)) return false;

		// cropped_seg
		if (!face_data.cropped_seg.empty())
		{
			std::vector<uchar> buf;
			cv::imencode(".png", face_data.cropped_seg, buf,
			{ cv::IMWRITE_PNG_COMPRESSION, 9, cv::IMWRITE_PNG_BILEVEL, 1 });
			*io_face_data.mutable_cropped_seg() = { buf.begin(), buf.end() };
		}

		// scaled_landmarks
		if (!face_data.scaled_landmarks.empty())
		{
			*io_face_data.mutable_scaled_landmarks() = { (const int*)face_data.scaled_landmarks.data(),
				(const int*)face_data.scaled_landmarks.data() + face_data.scaled_landmarks.size() * 2 };
		}

		// bbox
		if (face_data.bbox.width > 0 && face_data.bbox.height > 0)
		{
			*io_face_data.mutable_bbox() = { &(face_data.bbox.x), &(face_data.bbox.x) + 4 };
		}

		// shape_coefficients
		if (!face_data.shape_coefficients.empty())
		{
			*io_face_data.mutable_shape_coefficients() = { (float*)face_data.shape_coefficients.data,
				(float*)face_data.shape_coefficients.data + face_data.shape_coefficients.total() };
		}

		// tex_coefficients
		if (!face_data.tex_coefficients.empty())
		{
			*io_face_data.mutable_tex_coefficients() = { (float*)face_data.tex_coefficients.data,
				(float*)face_data.tex_coefficients.data + face_data.tex_coefficients.total() };
		}

		// expr_coefficients
		if (!face_data.expr_coefficients.empty())
		{
			*io_face_data.mutable_expr_coefficients() = { (float*)face_data.expr_coefficients.data,
				(float*)face_data.expr_coefficients.data + face_data.expr_coefficients.total() };
		}

		// vecR
		if (!face_data.vecR.empty())
		{
			*io_face_data.mutable_vecr() = { (float*)face_data.vecR.data,
				(float*)face_data.vecR.data + face_data.vecR.total() };
		}

		// vecT
		if (!face_data.vecT.empty())
		{
			*io_face_data.mutable_vect() = { (float*)face_data.vecT.data,
				(float*)face_data.vecT.data + face_data.vecT.total() };
		}

		// K
		if (!face_data.K.empty())
		{
			*io_face_data.mutable_k() = { (float*)face_data.K.data,
				(float*)face_data.K.data + face_data.K.total() };
		}

		// shape_coefficients_flipped
		if (!face_data.shape_coefficients_flipped.empty())
		{
			*io_face_data.mutable_shape_coefficients_flipped() = { (float*)face_data.shape_coefficients_flipped.data,
				(float*)face_data.shape_coefficients_flipped.data + face_data.shape_coefficients_flipped.total() };
		}

		// tex_coefficients_flipped
		if (!face_data.tex_coefficients_flipped.empty())
		{
			*io_face_data.mutable_tex_coefficients_flipped() = { (float*)face_data.tex_coefficients_flipped.data,
				(float*)face_data.tex_coefficients_flipped.data + face_data.tex_coefficients_flipped.total() };
		}

		// expr_coefficients_flipped
		if (!face_data.expr_coefficients_flipped.empty())
		{
			*io_face_data.mutable_expr_coefficients_flipped() = { (float*)face_data.expr_coefficients_flipped.data,
				(float*)face_data.expr_coefficients_flipped.data + face_data.expr_coefficients_flipped.total() };
		}

		// vecR_flipped
		if (!face_data.vecR_flipped.empty())
		{
			*io_face_data.mutable_vecr_flipped() = { (float*)face_data.vecR_flipped.data,
				(float*)face_data.vecR_flipped.data + face_data.vecR_flipped.total() };
		}

		// vecT_flipped
		if (!face_data.vecT_flipped.empty())
		{
			*io_face_data.mutable_vect_flipped() = { (float*)face_data.vecT_flipped.data,
				(float*)face_data.vecT_flipped.data + face_data.vecT_flipped.total() };
		}

		// enable_seg
		io_face_data.set_enable_seg(face_data.enable_seg);

		// max_bbox_res
		io_face_data.set_max_bbox_res(face_data.max_bbox_res);

		// Write to file
		std::ofstream output(cache_path.string(), std::fstream::trunc | std::fstream::binary);
		io_face_data.SerializeToOstream(&output);

		//path img_dir_path = path(img_path).parent_path() / path(img_path).stem();
		//path face_data_path = img_dir_path / "face_data.fs";
		//create_directory(img_dir_path);
		//std::ofstream output(face_data_path.string(), std::fstream::trunc | std::fstream::binary);
		//io_face_data.SerializeToOstream(&output);

		return true;

#else
		return false;
#endif // WITH_PROTOBUF		
	}
	
}   // namespace face_swap


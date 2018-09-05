#include "face_swap/render_utilities.h"
#include "face_swap/utilities.h"
#include <iostream>	// Debug

// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>	// Debug

namespace face_swap
{
    void renderWireframe(cv::Mat& img, const Mesh& mesh, const cv::Mat& P, 
        float scale, const cv::Scalar& color)
    {
        cv::Mat vertices = mesh.vertices.t();
        vertices.push_back(cv::Mat::ones(1, vertices.cols, CV_32F));
        cv::Mat proj = P * vertices;
        proj = proj / cv::repeat(proj.row(2), 3, 1);

        if (scale != 1.0f)
            cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_CUBIC);

        // For each face
//        float* faces_data = (float*)mesh.faces.data;
        unsigned short* faces_data = (unsigned short*)mesh.faces.data;
        for (int f = 0; f < mesh.faces.rows; ++f)
        {
            int i1 = (int)*faces_data++;
            int i2 = (int)*faces_data++;
            int i3 = (int)*faces_data++;
            cv::Point p1(std::round(proj.at<float>(0, i1))*scale, std::round(proj.at<float>(1, i1))*scale);
            cv::Point p2(std::round(proj.at<float>(0, i2))*scale, std::round(proj.at<float>(1, i2))*scale);
            cv::Point p3(std::round(proj.at<float>(0, i3))*scale, std::round(proj.at<float>(1, i3))*scale);
            if (is_ccw(p1, p2, p3))
            {
                // Draw face
                cv::line(img, p1, p2, color);
                cv::line(img, p2, p3, color);
                cv::line(img, p3, p1, color);
            }
        }
    }

    void renderWireframe(cv::Mat& img, const Mesh& mesh,
        const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K,
        float scale, const cv::Scalar& color)
    {
        // Project points to image plane
        cv::Mat distCoef = cv::Mat::zeros(1, 4, CV_32F);
        std::vector<cv::Point2f> proj;
        cv::projectPoints(mesh.vertices, rvec, tvec, K, distCoef, proj);

        if (scale != 1.0f)
            cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_CUBIC);

        // For each face
        unsigned short* faces_data = (unsigned short*)mesh.faces.data;
        for (int f = 0; f < mesh.faces.rows; ++f)
        {
            int i1 = (int)*faces_data++;
            int i2 = (int)*faces_data++;
            int i3 = (int)*faces_data++;
            cv::Point p1(std::round(proj[i1].x*scale), std::round(proj[i1].y*scale));
            cv::Point p2(std::round(proj[i2].x*scale), std::round(proj[i2].y*scale));
            cv::Point p3(std::round(proj[i3].x*scale), std::round(proj[i3].y*scale));
            if (is_ccw(p1, p2, p3))
            {
                // Draw face
                cv::line(img, p1, p2, color);
                cv::line(img, p2, p3, color);
                cv::line(img, p3, p1, color);
            }  
        }
    }

    void renderWireframeUV(cv::Mat& img, const Mesh& mesh, const cv::Mat& uv,
        const cv::Scalar& color)
    {

        // For each face
        //        float* faces_data = (float*)mesh.faces.data;
        unsigned short* faces_data = (unsigned short*)mesh.faces.data;
        for (int f = 0; f < mesh.faces.rows; ++f)
        {
            int i1 = (int)*faces_data++;
            int i2 = (int)*faces_data++;
            int i3 = (int)*faces_data++;
            //cv::Point p1(std::round(proj.at<float>(0, i1)), std::round(proj.at<float>(1, i1)));
            //cv::Point p2(std::round(proj.at<float>(0, i2)), std::round(proj.at<float>(1, i2)));
            //cv::Point p3(std::round(proj.at<float>(0, i3)), std::round(proj.at<float>(1, i3)));
            cv::Point p1(std::round(uv.at<float>(i1, 0)*img.cols), std::round(uv.at<float>(i1, 1)*img.rows));
            cv::Point p2(std::round(uv.at<float>(i2, 0)*img.cols), std::round(uv.at<float>(i2, 1)*img.rows));
            cv::Point p3(std::round(uv.at<float>(i3, 0)*img.cols), std::round(uv.at<float>(i3, 1)*img.rows));
            if (is_ccw(p1, p2, p3))
            {
                // Draw face
                cv::line(img, p1, p2, color);
                cv::line(img, p2, p3, color);
                cv::line(img, p3, p1, color);
            }
        }
    }

    void renderBoundary(cv::Mat& img, const Mesh& mesh, 
        const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K,
        const cv::Scalar& color)
    {
    }

	inline float barycentric_weight(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3)
	{
		return (p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x);
	};

	void renderMesh(cv::Mat& img, const Mesh& mesh,
		const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K,
		cv::Mat& depthbuf, int ss)
	{
		if (ss > 1)
			cv::resize(img, img, cv::Size(), (double)ss, (double)ss, cv::INTER_CUBIC);

		// Project points to image plane
		cv::Mat P = createPerspectiveProj3x4(rvec, tvec, K);
		cv::Mat vertices_h;		// 4 X N
		cv::vconcat(mesh.vertices.t(), cv::Mat::ones(1, mesh.vertices.rows, CV_32F), vertices_h);
		cv::Mat proj = P * vertices_h;	// 3 X N

		//cv::Mat M = P(cv::Rect(0, 0, 3, 3));
		//double dM = cv::determinant(M);
		//float m3 = cv::norm(M.rowRange(2, 3));	

		// Calculate image coordinates and depth
		cv::Mat vertices_proj(mesh.vertices.rows, 2, CV_32F);
		cv::Mat vertices_depth(mesh.vertices.rows, 1, CV_32F);
		float* vertices_proj_data = (float*)vertices_proj.data;
		float* vertices_depth_data = (float*)vertices_depth.data;
		for (int i = 0; i < mesh.vertices.rows; ++i)
		{
			float x = proj.at<float>(0, i);
			float y = proj.at<float>(1, i);
			float w = proj.at<float>(2, i);
			*vertices_proj_data++ = x / w;
			*vertices_proj_data++ = y / w;
			*vertices_depth_data++ = -w;
		}

		if (ss > 1)
			vertices_proj *= (float)ss;

		// Initialize buffers
		depthbuf = cv::Mat(img.size(), CV_32F, std::numeric_limits<float>::max());
		cv::Mat uvbuf(img.size(), CV_32FC2, -1.0f);

		// For each triangle
		unsigned short* faces_data = (unsigned short*)mesh.faces.data;
		cv::Point2f* proj_points_data = (cv::Point2f*)vertices_proj.data;
		for (int f = 0; f < mesh.faces.rows; ++f)
		{
			// Get triangle points
			int i1 = (int)*faces_data++;
			int i2 = (int)*faces_data++;
			int i3 = (int)*faces_data++;

			cv::Point2f& p1 = proj_points_data[i1];
			cv::Point2f& p2 = proj_points_data[i2];
			cv::Point2f& p3 = proj_points_data[i3];
			if (!is_ccw(p1, p2, p3)) continue;

			// Calculate triangle's bounding box
			const int min_x = std::max(std::min(std::floor(p1.x), std::min(std::floor(p2.x), std::floor(p3.x))), 0.0f);
			const int max_x = std::min(std::max(std::ceil(p1.x), std::max(std::ceil(p2.x), std::ceil(p3.x))), (float)(img.cols - 1));
			const int min_y = std::max(std::min(std::floor(p1.y), std::min(std::floor(p2.y), std::floor(p3.y))), 0.0f);
			const int max_y = std::min(std::max(std::ceil(p1.y), std::max(std::ceil(p2.y), std::ceil(p3.y))), (float)(img.rows - 1));

			// For each pixel in the triangle's bounding box
			for (int yi = min_y; yi <= max_y; ++yi)
			{
				float* depthbuf_data = depthbuf.ptr<float>(yi, min_x);
				cv::Point2f* uvbuf_data = uvbuf.ptr<cv::Point2f>(yi, min_x);
				for (int xi = min_x; xi <= max_x; ++xi)
				{
					// We want centers of pixels to be used in computations. Todo: Do we?
					const float x = static_cast<float>(xi) + 0.5f;
					const float y = static_cast<float>(yi) + 0.5f;
					const cv::Point2f p(x, y);

					// Calculate affine barycentric weights
					float g1 = barycentric_weight(p2, p3, p);
					float g2 = barycentric_weight(p3, p1, p);
					float g3 = barycentric_weight(p1, p2, p);
					if (g1 >= 0.0f && g2 >= 0.0f && g3 >= 0.0f)
					{
						float inv_area = 1.0f / barycentric_weight(p1, p2, p3);
						g1 *= inv_area;
						g2 *= inv_area;
						g3 *= inv_area;

						// Calculate depth
						float depth = g1 * vertices_depth.at<float>(i1) +
							g2 * vertices_depth.at<float>(i2) +
							g3 * vertices_depth.at<float>(i3);

						if (depth < *depthbuf_data)
						{
							*depthbuf_data = depth;

							// Calculate texture coordinates
							float u = g1 * mesh.uv.at<float>(i1, 0) +
								g2 * mesh.uv.at<float>(i2, 0) +
								g3 * mesh.uv.at<float>(i3, 0);
							float v = g1 * mesh.uv.at<float>(i1, 1) +
								g2 * mesh.uv.at<float>(i2, 1) +
								g3 * mesh.uv.at<float>(i3, 1);
							*uvbuf_data = cv::Point2f(u*mesh.tex.cols, v*mesh.tex.rows);
						}
					}

					++depthbuf_data;
					++uvbuf_data;
				}
			}
		}

		// Interpolate source texture
		cv::Mat colorbuf;
		cv::remap(mesh.tex, colorbuf, uvbuf, cv::Mat(), cv::INTER_CUBIC, cv::BORDER_CONSTANT);
		
		// Write colorbuf to output image
		float* depthbuf_data = (float*)depthbuf.data;
		if (colorbuf.channels() == 3)
		{
			cv::Vec3b* colorbuf_data = (cv::Vec3b*)colorbuf.data;
			for (int r = 0; r < img.rows; ++r)
			{
				cv::Vec3b* img_data = img.ptr<cv::Vec3b>(r);
				for (int c = 0; c < img.cols; ++c)
				{
					if (*depthbuf_data++ < std::numeric_limits<float>::max())
						*img_data = *colorbuf_data;
					++img_data;
					++colorbuf_data;
				}
			}
		}
		else if (colorbuf.channels() == 4)
		{
			float alpha = 0.0f;
			cv::Vec4b* colorbuf_data = (cv::Vec4b*)colorbuf.data;
			for (int r = 0; r < img.rows; ++r)
			{
				cv::Vec3b* img_data = img.ptr<cv::Vec3b>(r);
				for (int c = 0; c < img.cols; ++c)
				{
					float& depth = *depthbuf_data;
					if (depth < std::numeric_limits<float>::max())
					{
						alpha = (*colorbuf_data)[3] / 255.0f;
						cv::Vec3b& out_color = *((cv::Vec3b*)colorbuf_data);
						cv::Vec3b& img_color = *img_data;
						img_color[0] = (uchar)std::round(alpha*out_color[0] + (1 - alpha)*img_color[0]);
						img_color[1] = (uchar)std::round(alpha*out_color[1] + (1 - alpha)*img_color[1]);
						img_color[2] = (uchar)std::round(alpha*out_color[2] + (1 - alpha)*img_color[2]);

						// Set background depth for small alpha values
						if (alpha < 0.1f) depth = std::numeric_limits<float>::max();
					}	
					++img_data;
					++depthbuf_data;
					++colorbuf_data;
				}
			}
		}
	}

	cv::Mat renderDepthMap(const cv::Mat& depth_map)
	{
		// Create inverse depth map
		float curr_idepth = 0.0f, min_depth = 0.0f, max_idepth = std::numeric_limits<float>::max();
		cv::Mat idepth(depth_map.size(), CV_32F);
		float* depth_map_data = (float*)depth_map.data;
		float* idepth_data = (float*)idepth.data;
		for (int i = 0; i < depth_map.total(); ++i)
		{
			if ((*depth_map_data - 1e-6f) < std::numeric_limits<float>::max())
			{
				curr_idepth = 1.0f / *depth_map_data;
				min_depth = std::max(curr_idepth, min_depth);
				max_idepth = std::min(curr_idepth, max_idepth);
				*idepth_data++ = curr_idepth;
			}
			else *idepth_data++ = 0.0f;
			++depth_map_data;
		}	

		// Normalize
		idepth_data = (float*)idepth.data;
		for (int i = 0; i < idepth.total(); ++i)
		{
			if (*idepth_data > 0.0f)
			{
				*idepth_data = ((*idepth_data - min_depth) / (max_idepth - min_depth))*254.0f + 1.0f;

			}

			++idepth_data;
		}
		cv::Mat idepth_norm;
		idepth.convertTo(idepth_norm, CV_8U);

		// Apply color map
		cv::Mat depth_color_map;
		applyColorMap(idepth_norm, depth_color_map, cv::COLORMAP_HOT);

		return depth_color_map;
	}

	cv::Mat renderImagePipe(const std::vector<cv::Mat>& images, int padding,
		const cv::Scalar& border_color)
	{
		if (images.empty()) return cv::Mat();

		// Find pipe height
		int height = 0;
		for (const cv::Mat& img : images)
			height = std::max(height, img.rows);

		// Concatenate images
		cv::Mat frame;
		for (const cv::Mat& img : images)
		{
			int top = 0, bottom = 0;
			if (img.rows < height)
			{
				top = (height - img.rows) / 2;
				bottom = height - img.rows - top;
			}
			cv::Mat padded_img;
			cv::copyMakeBorder(img, padded_img, top + padding, bottom + padding,
				padding, padding, cv::BORDER_CONSTANT, border_color);
			if (frame.empty())
				frame = padded_img;
			else cv::hconcat(frame, padded_img, frame);
		}

		return frame;
	}

	void overlayImage(cv::Mat& img, const cv::Mat& overlay,
		const cv::Point& loc, const cv::Mat& mask)
	{
		// Calculate overlay bounding box in the image space
		int min_c = std::max(loc.x - overlay.cols / 2, 0);
		int min_r = std::max(loc.y - overlay.rows / 2, 0);
		int max_c = std::min(min_c + overlay.cols - 1, img.cols - 1);
		int max_r = std::min(min_r + overlay.rows - 1, img.rows - 1);

		// Write colorbuf to output image
		if (mask.empty())	// Without alpha blending
		{
			for (int r = min_r; r <= max_r; ++r)
			{
				cv::Vec3b* img_data = img.ptr<cv::Vec3b>(r, min_c);
				const cv::Vec3b* overlay_data = overlay.ptr<cv::Vec3b>(r - min_r);
				for (int c = min_c; c <= max_c; ++c)
				{
					*img_data++ = *overlay_data++;
				}
			}
		}
		else	// With alpha blending
		{
			float alpha = 0.0f;
			for (int r = min_r; r <= max_r; ++r)
			{
				cv::Vec3b* img_data = img.ptr<cv::Vec3b>(r, min_c);
				const cv::Vec3b* overlay_data = overlay.ptr<cv::Vec3b>(r - min_r);
				const unsigned char* mask_data = mask.ptr<unsigned char>(r - min_r);
				for (int c = min_c; c <= max_c; ++c)
				{
					cv::Vec3b& img_color = *img_data++;
					const cv::Vec3b& overlay_color = *overlay_data++;
					float alpha = *mask_data++ / 255.0f;
					img_color[0] = (uchar)std::round(alpha*overlay_color[0] + (1 - alpha)*img_color[0]);
					img_color[1] = (uchar)std::round(alpha*overlay_color[1] + (1 - alpha)*img_color[1]);
					img_color[2] = (uchar)std::round(alpha*overlay_color[2] + (1 - alpha)*img_color[2]);
				}
			}
		}
	}

	cv::Mat calcCircleMask(const cv::Mat& img)
	{
		cv::Mat mask(img.size(), CV_8U);
		float radius = (float)std::min(img.rows, img.cols) / 2.0f;
		cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);

		// For each pixel in the image
		for (int r = 0; r < img.rows; ++r)
		{
			unsigned char* mask_data = mask.ptr<unsigned char>(r);
			for (int c = 0; c < img.cols; ++c)
			{
				cv::Point2f p((float)r, (float)c);
				float d = cv::norm(center - p);
				if (d > radius)
					*mask_data++ = 0;
				else *mask_data++ = 255;
			}
		}

		return mask;
	}

	void renderImageOverlay(cv::Mat& img, const cv::Rect& bbox,
		const cv::Mat& src, const cv::Mat& tgt, const cv::Scalar border_color)
	{
		// Scale images
		float src_scale = ((bbox.width + bbox.height) / 4) / (float)src.cols;
		float tgt_scale = ((bbox.width + bbox.height) / 4) / (float)tgt.cols;
		
		// Calculate masks
		cv::Mat src_mask = calcCircleMask(src);
		cv::Mat tgt_mask = calcCircleMask(tgt);

		// Resize images
		cv::Mat src_scaled, tgt_scaled;
		cv::Mat src_mask_scaled, tgt_mask_scaled;
		cv::resize(src, src_scaled, cv::Size(), src_scale, src_scale, cv::INTER_CUBIC);
		cv::resize(tgt, tgt_scaled, cv::Size(), tgt_scale, tgt_scale, cv::INTER_CUBIC);
		cv::resize(src_mask, src_mask_scaled, cv::Size(), src_scale, src_scale, cv::INTER_CUBIC);
		cv::resize(tgt_mask, tgt_mask_scaled, cv::Size(), tgt_scale, tgt_scale, cv::INTER_CUBIC);

		// Calculate overlay center points
		//cv::Point src_loc(src_scaled.cols / 2, src_scaled.rows / 2);
		//cv::Point tgt_loc(img.cols - 1 - tgt_scaled.cols / 2, tgt_scaled.rows / 2);
		cv::Point src_loc(bbox.x, bbox.y);
		cv::Point tgt_loc(bbox.x + bbox.width - 1, bbox.y);

		// Borders
		int top = 0, bottom = 0, left = 0, right = 0;
		int min_y = std::min(src_loc.y - src_scaled.rows / 2, tgt_loc.y - tgt_scaled.rows / 2);
		if (min_y < 0) top = -min_y;
		int src_img_min_x = src_loc.x - src_scaled.cols / 2;
		if (src_img_min_x < 0) left = -src_img_min_x;
		int tgt_img_max_x = tgt_loc.x + tgt_scaled.cols / 2;
		if ((tgt_img_max_x - img.cols + 1) > 0)
			right = tgt_img_max_x - img.cols + 1;

		cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, border_color);
		src_loc.x += left;
		src_loc.y += top;
		tgt_loc.x += left;
		tgt_loc.y += top;	
		
		overlayImage(img, src_scaled, src_loc, src_mask_scaled);
		overlayImage(img, tgt_scaled, tgt_loc, tgt_mask_scaled);
	}

	void renderSegmentation(cv::Mat& img, const cv::Mat& seg, float alpha, const cv::Scalar& color)
	{
		cv::Point3_<uchar> bgr((uchar)color[0], (uchar)color[1], (uchar)color[2]);

		int r, c;
		for (r = 0; r < img.rows; ++r)
		{
			cv::Point3_<uchar>* img_data = img.ptr<cv::Point3_<uchar>>(r);
			const unsigned char* seg_data = seg.ptr<uchar>(r);
			for (c = 0; c < img.cols; ++c)
			{
				if (*seg_data++ == 255)
				{
					img_data->x = (unsigned char)std::round(bgr.x * alpha + img_data->x*(1 - alpha));
					img_data->y = (unsigned char)std::round(bgr.y * alpha + img_data->y*(1 - alpha));
					img_data->z = (unsigned char)std::round(bgr.z * alpha + img_data->z*(1 - alpha));
				}
				++img_data;
			}
		}
	}

    bool is_ccw(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3)
    {
        return ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)) < 0;
    }

    cv::Mat computeFaceNormals(const Mesh& mesh)
    {
        cv::Mat v1, v2, v3, v21, v31, N;
        cv::Mat face_normals = cv::Mat::zeros(mesh.faces.size(), CV_32F);
        int i1, i2, i3;

        // For each face
        unsigned short* faces_data = (unsigned short*)mesh.faces.data;
        for (int i = 0; i < mesh.faces.rows; ++i)
        {
            i1 = (int)*faces_data++;
            i2 = (int)*faces_data++;
            i3 = (int)*faces_data++;
            v1 = mesh.vertices.row(i1);
            v2 = mesh.vertices.row(i2);
            v3 = mesh.vertices.row(i3);
            v21 = v2 - v1;
            v31 = v3 - v1;
            N = v31.cross(v21);
            face_normals.row(i) = N / cv::norm(N);
        }

        return face_normals;
    }

    cv::Mat computeVertexNormals(const Mesh& mesh)
    {
        cv::Mat face_normals = computeFaceNormals(mesh);
        cv::Mat vert_normals = cv::Mat::zeros(mesh.vertices.size(), CV_32F);
        cv::Mat N;
        int i1, i2, i3;

        // For each face
        unsigned short* faces_data = (unsigned short*)mesh.faces.data;
        for (int i = 0; i < mesh.faces.rows; ++i)
        {
            int i1 = (int)*faces_data++;
            int i2 = (int)*faces_data++;
            int i3 = (int)*faces_data++;

            N = face_normals.row(i);
            vert_normals.row(i1) += N;
            vert_normals.row(i2) += N;
            vert_normals.row(i3) += N;
        }

        // For each vertex normal
        for (int i = 0; i < vert_normals.rows; ++i)
        {
            N = vert_normals.row(i);
            N /= cv::norm(N);
        }

        return vert_normals;
    }
	
}   // namespace face_swap


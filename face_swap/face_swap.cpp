#include "face_swap/face_swap.h"
#include "face_swap/utilities.h"

// std
#include <limits>

// OpenCV
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  // Debug

namespace face_swap
{
    FaceSwap::FaceSwap(const std::string& landmarks_path, const std::string& model_3dmm_h5_path,
        const std::string& model_3dmm_dat_path, const std::string& reg_model_path,
        const std::string& reg_deploy_path, const std::string& reg_mean_path,
        bool generic, bool with_expr, bool with_gpu, int gpu_device_id) :
		m_with_gpu(with_gpu),
		m_gpu_device_id(gpu_device_id)
    {
        // Initialize Sequence Face Landmarks
        m_sfl = sfl::SequenceFaceLandmarks::create(landmarks_path);

        // Initialize CNN 3DMM with exression
        m_cnn_3dmm_expr = std::make_unique<CNN3DMMExpr>(
			reg_deploy_path, reg_model_path, reg_mean_path, model_3dmm_dat_path,
			generic, with_expr, with_gpu, gpu_device_id);

        // Load Basel 3DMM
        m_basel_3dmm = std::make_unique<Basel3DMM>();
        *m_basel_3dmm = Basel3DMM::load(model_3dmm_h5_path);

        // Create renderer
        m_face_renderer = std::make_unique<FaceRenderer>();
    }

	void FaceSwap::setSegmentationModel(const std::string& seg_model_path,
		const std::string& seg_deploy_path)
	{
		m_face_seg = std::make_unique<face_seg::FaceSeg>(seg_deploy_path,
			seg_model_path, m_with_gpu, m_gpu_device_id);
	}

	void FaceSwap::clearSegmentationModel()
	{
		m_face_seg = nullptr;
	}

	bool FaceSwap::isSegmentationModelInit()
	{
		return m_face_seg != nullptr;
	}

	bool FaceSwap::setSource(const cv::Mat& img, const cv::Mat& seg)
    {
        m_source_img = img;

        // Preprocess image
        std::vector<cv::Point> cropped_landmarks;
        cv::Mat cropped_img, cropped_seg;
        if (!preprocessImages(img, seg, m_src_landmarks, cropped_landmarks,
			cropped_img, cropped_seg))
            return false;

		// If segmentation was not specified and we have a segmentation model then
		// calculate the segmentation
		if (cropped_seg.empty() && m_face_seg != nullptr)
			cropped_seg = m_face_seg->process(cropped_img);

        // Calculate coefficients and pose
        cv::Mat shape_coefficients, tex_coefficients, expr_coefficients;
        cv::Mat vecR, vecT, K;
        m_cnn_3dmm_expr->process(cropped_img, cropped_landmarks, shape_coefficients,
            tex_coefficients, expr_coefficients, vecR, vecT, K);

        // Create mesh
        m_src_mesh = m_basel_3dmm->sample(shape_coefficients, tex_coefficients, 
            expr_coefficients);

        // Texture mesh
        generateTexture(m_src_mesh, cropped_img, cropped_seg, vecR, vecT, K, 
            m_tex, m_uv);

        /// Debug ///
        m_src_cropped_img = cropped_img;
        m_src_cropped_seg = cropped_seg;
        m_src_cropped_landmarks = cropped_landmarks;
        m_src_vecR = vecR;
        m_src_vecT = vecT;
        m_src_K = K;
        /////////////

        return true;
    }

    bool FaceSwap::setTarget(const cv::Mat& img, const cv::Mat& seg)
    {
        m_target_img = img;
        m_target_seg = seg;

        // Preprocess image
        std::vector<cv::Point> cropped_landmarks;
        cv::Mat cropped_img, cropped_seg;
        if (!preprocessImages(img, seg, m_tgt_landmarks, cropped_landmarks,
            cropped_img, cropped_seg, m_target_bbox))
            return false;

		// If segmentation was not specified and we have a segmentation model then
		// calculate the segmentation
		if (cropped_seg.empty() && m_face_seg != nullptr)
		{
			cropped_seg = m_face_seg->process(cropped_img);
			m_target_seg = cv::Mat::zeros(img.size(), CV_8U);
			cropped_seg.copyTo(m_target_seg(m_target_bbox));
		}
			
        m_tgt_cropped_img = cropped_img;
        m_tgt_cropped_seg = cropped_seg;
        
        /// Debug ///
        m_tgt_cropped_landmarks = cropped_landmarks;
        /////////////

        // Calculate coefficients and pose
        cv::Mat shape_coefficients, tex_coefficients, expr_coefficients;
        m_cnn_3dmm_expr->process(cropped_img, cropped_landmarks, shape_coefficients,
            tex_coefficients, expr_coefficients, m_vecR, m_vecT, m_K);

        // Create mesh
        m_dst_mesh = m_basel_3dmm->sample(shape_coefficients, tex_coefficients,
            expr_coefficients);
        m_dst_mesh.tex = m_tex;
        m_dst_mesh.uv = m_uv;

        // Initialize renderer
        m_face_renderer->init(cropped_img.cols, cropped_img.rows);
        m_face_renderer->setProjection(m_K.at<float>(4));
        m_face_renderer->setMesh(m_dst_mesh);

        return true;
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

	bool FaceSwap::setImages(const cv::Mat& src, const cv::Mat& tgt,
		const cv::Mat& src_seg, const cv::Mat& tgt_seg)
	{
		m_source_img = src;
		m_target_img = tgt;
		m_target_seg = tgt_seg;

		// Preprocess source image
		std::vector<cv::Point> cropped_src_landmarks;
		cv::Mat cropped_src, cropped_src_seg;
		if (!preprocessImages(src, src_seg, m_src_landmarks, cropped_src_landmarks,
			cropped_src, cropped_src_seg))
			return false;

		// Preprocess target image
		std::vector<cv::Point> cropped_tgt_landmarks;
		cv::Mat cropped_tgt, cropped_tgt_seg;
		if (!preprocessImages(tgt, tgt_seg, m_tgt_landmarks, cropped_tgt_landmarks,
			cropped_tgt, cropped_tgt_seg, m_target_bbox))
			return false;

		// Check if horizontal flip is required
		float src_angle = sfl::getFaceApproxHorAngle(cropped_src_landmarks);
		float tgt_angle = sfl::getFaceApproxHorAngle(cropped_tgt_landmarks);
		if ((src_angle * tgt_angle) < 0 && std::abs(src_angle - tgt_angle) > (CV_PI / 18.0f))
		{
			// Horizontal flip the source image
			cv::flip(cropped_src, cropped_src, 1);
			if(!cropped_src_seg.empty())
				cv::flip(cropped_src_seg, cropped_src_seg, 1);

			// Horizontal flip the source landmarks
			horFlipLandmarks(cropped_src_landmarks, cropped_src.cols);
		}

		// If source segmentation was not specified and we have a segmentation model then
		// calculate the segmentation
		if (cropped_src_seg.empty() && m_face_seg != nullptr)
			cropped_src_seg = m_face_seg->process(cropped_src);

		// If target segmentation was not specified and we have a segmentation model then
		// calculate the segmentation
		if (cropped_tgt_seg.empty() && m_face_seg != nullptr)
		{
			cropped_tgt_seg = m_face_seg->process(cropped_tgt);
			m_target_seg = cv::Mat::zeros(tgt.size(), CV_8U);
			cropped_tgt_seg.copyTo(m_target_seg(m_target_bbox));
		}

		m_tgt_cropped_img = cropped_tgt;
		m_tgt_cropped_seg = cropped_tgt_seg;

		/// Debug ///
		m_tgt_cropped_landmarks = cropped_tgt_landmarks;
		/////////////

		// Calculate source coefficients and pose
		{
			cv::Mat shape_coefficients, tex_coefficients, expr_coefficients;
			cv::Mat vecR, vecT, K;
			m_cnn_3dmm_expr->process(cropped_src, cropped_src_landmarks, shape_coefficients,
				tex_coefficients, expr_coefficients, vecR, vecT, K);

			// Create source mesh
			m_src_mesh = m_basel_3dmm->sample(shape_coefficients, tex_coefficients,
				expr_coefficients);

			// Texture source mesh
			generateTexture(m_src_mesh, cropped_src, cropped_src_seg, vecR, vecT, K,
				m_tex, m_uv);

			/// Debug ///
			m_src_cropped_img = cropped_src;
			m_src_cropped_seg = cropped_src_seg;
			m_src_cropped_landmarks = cropped_src_landmarks;
			m_src_vecR = vecR;
			m_src_vecT = vecT;
			m_src_K = K;
			/////////////
		}
		
		// Calculate target coefficients and pose
		{
			cv::Mat shape_coefficients, tex_coefficients, expr_coefficients;
			m_cnn_3dmm_expr->process(cropped_tgt, cropped_tgt_landmarks, shape_coefficients,
				tex_coefficients, expr_coefficients, m_vecR, m_vecT, m_K);

			// Create target mesh
			m_dst_mesh = m_basel_3dmm->sample(shape_coefficients, tex_coefficients,
				expr_coefficients);
			m_dst_mesh.tex = m_tex;
			m_dst_mesh.uv = m_uv;
		}

		// Initialize renderer
		m_face_renderer->init(cropped_tgt.cols, cropped_tgt.rows);
		m_face_renderer->setProjection(m_K.at<float>(4));
		m_face_renderer->setMesh(m_dst_mesh);

		return true;
	}

    cv::Mat FaceSwap::swap()
    {
        // Render
        cv::Mat rendered_img;
        m_face_renderer->render(m_vecR, m_vecT);
        m_face_renderer->getFrameBuffer(rendered_img);

        // Blend images
        cv::Mat tgt_rendered_img = cv::Mat::zeros(m_target_img.size(), CV_8UC3);
        rendered_img.copyTo(tgt_rendered_img(m_target_bbox));

        m_tgt_rendered_img = tgt_rendered_img;  // For debug

        return blend(tgt_rendered_img, m_target_img, m_target_seg);
    }

    const Mesh & FaceSwap::getSourceMesh() const
    {
        return m_src_mesh;
    }

    const Mesh & FaceSwap::getTargetMesh() const
    {
        return m_dst_mesh;
    }

    bool FaceSwap::preprocessImages(const cv::Mat& img, const cv::Mat& seg,
        std::vector<cv::Point>& landmarks, std::vector<cv::Point>& cropped_landmarks,
        cv::Mat& cropped_img, cv::Mat& cropped_seg, cv::Rect& bbox)
    {
        // Calculate landmarks
        m_sfl->clear();
        const sfl::Frame& lmsFrame = m_sfl->addFrame(img);
        if (lmsFrame.faces.empty()) return false;
        //std::cout << "faces found = " << lmsFrame.faces.size() << std::endl;    // Debug
        const sfl::Face* face = lmsFrame.getFace(sfl::getMainFaceID(m_sfl->getSequence()));
        landmarks = face->landmarks; // Debug
        cropped_landmarks = landmarks; 

        // Calculate crop bounding box
        bbox = sfl::getFaceBBoxFromLandmarks(landmarks, img.size(), true);
        bbox.width = bbox.width / 4 * 4;    // Make sure cropped image is dividable by 4
        bbox.height = bbox.height / 4 * 4;

        // Crop landmarks
        for (cv::Point& p : cropped_landmarks)
        {
            p.x -= bbox.x;
            p.y -= bbox.y;
        }

        // Crop images
        cropped_img = img(bbox);
        if(!seg.empty()) cropped_seg = seg(bbox);

        return true;
    }

    bool FaceSwap::preprocessImages(const cv::Mat& img, const cv::Mat& seg,
        std::vector<cv::Point>& landmarks, std::vector<cv::Point>& cropped_landmarks,
        cv::Mat& cropped_img, cv::Mat& cropped_seg)
    {
        cv::Rect bbox;
        return preprocessImages(img, seg, landmarks, cropped_landmarks,
            cropped_img, cropped_seg, bbox);
    }

    void FaceSwap::generateTexture(const Mesh& mesh, const cv::Mat& img, 
        const cv::Mat& seg, const cv::Mat& vecR, const cv::Mat& vecT,
        const cv::Mat& K, cv::Mat& tex, cv::Mat& uv)
    {
        // Resize images to power of 2 size
        cv::Size tex_size(nextPow2(img.cols), nextPow2(img.rows));
        cv::Mat img_scaled, seg_scaled;
        cv::resize(img, img_scaled, tex_size, 0.0, 0.0, cv::INTER_CUBIC);
        if(!seg.empty())
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

        uv = generateTextureCoordinates(m_src_mesh, img.size(), vecR, vecT, K);
    }

    cv::Mat FaceSwap::generateTextureCoordinates(
        const Mesh& mesh,const cv::Size& img_size,
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

    cv::Mat FaceSwap::blend(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& dst_seg)
    {
        // Calculate mask
        cv::Mat mask(src.size(), CV_8U);
        unsigned char* src_data = src.data;
        unsigned char* dst_seg_data = dst_seg.data;
        unsigned char* mask_data = mask.data;
        for (int i = 0; i < src.total(); ++i)
        {
            unsigned char cb = *src_data++;
            unsigned char cg = *src_data++;
            unsigned char cr = *src_data++;
            if (!(cb == 0 && cg == 0 && cr == 0))  *mask_data++ = 255;
            else *mask_data++ = 0;
        }

        // Combine the segmentation with the mask
        if (!dst_seg.empty())
            cv::bitwise_and(mask, dst_seg, mask);

        // Find center point
        int minc = std::numeric_limits<int>::max(), minr = std::numeric_limits<int>::max();
        int maxc = 0, maxr = 0;
        mask_data = mask.data;
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

    cv::Mat FaceSwap::debugSourceMeshWireframe()
    {
        const float scale = 3.0f;
        cv::Mat out = m_src_cropped_img.clone();
        cv::Mat P = createPerspectiveProj3x4(m_src_vecR, m_src_vecT, m_src_K);
        std::vector<cv::Point> scaled_landmarks(m_src_cropped_landmarks);
        for (cv::Point& p : scaled_landmarks) p *= (int)scale;

        // Render
        renderWireframe(out, m_src_mesh, P, scale);
        //sfl::render(out, scaled_landmarks, false, cv::Scalar(0, 0, 255));
        return out;
    }

    cv::Mat FaceSwap::debugTargetMeshWireframe()
    {
        const float scale = 3.0f;
        cv::Mat out = m_tgt_cropped_img.clone();
        cv::Mat P = createPerspectiveProj3x4(m_vecR, m_vecT, m_K);
        std::vector<cv::Point> scaled_landmarks(m_tgt_cropped_landmarks);
        for (cv::Point& p : scaled_landmarks) p *= (int)scale;

        // Render
        renderWireframe(out, m_dst_mesh, P, scale);
        //sfl::render(out, scaled_landmarks, false, cv::Scalar(0, 0, 255));
        return out;
    }

    cv::Mat FaceSwap::debug()
    {
        cv::Mat src_d = debugSourceMeshWireframe();
        cv::Mat tgt_d = debugTargetMeshWireframe();
        cv::Size max_size(std::max(src_d.cols, tgt_d.cols), 
            std::max(src_d.rows, tgt_d.rows));

        cv::Mat src_d_out = cv::Mat::zeros(max_size, CV_8UC3);
        cv::Mat tgt_d_out = cv::Mat::zeros(max_size, CV_8UC3);
        src_d.copyTo(src_d_out(cv::Rect(0, 0, src_d.cols, src_d.rows)));
        tgt_d.copyTo(tgt_d_out(cv::Rect(0, 0, tgt_d.cols, tgt_d.rows)));
        cv::Mat out;
        cv::hconcat(src_d_out, tgt_d_out, out);
        return out;
    }

    cv::Mat FaceSwap::debugSourceMesh()
    {
        return debugMesh(m_src_cropped_img, m_src_cropped_seg, m_uv, 
            m_src_mesh, m_src_vecR, m_src_vecT, m_src_K);
    }

    cv::Mat FaceSwap::debugTargetMesh()
    {
        cv::Mat uv = generateTextureCoordinates(m_dst_mesh, m_tgt_cropped_seg.size(),
            m_vecR, m_vecT, m_K);
        return debugMesh(m_tgt_cropped_img, m_tgt_cropped_seg, uv,
            m_dst_mesh, m_vecR, m_vecT, m_K);
    }

    cv::Mat FaceSwap::debugMesh(const cv::Mat& img, const cv::Mat& seg,
        const cv::Mat& uv, const Mesh& mesh,
        const cv::Mat& vecR, const cv::Mat& vecT, const cv::Mat& K)
    {
        cv::Mat out;

        // Create texture
        cv::Mat tex(img.size(), CV_8UC3);
        unsigned char* tex_data = tex.data;
        int total_pixels = tex.total() * tex.channels();
        for (int i = 0; i < total_pixels; ++i) *tex_data++ = 192;

        // Add segmentation colors
        if (!seg.empty())
        {
            cv::Vec3b* tex_data = (cv::Vec3b*)tex.data;
            unsigned char* seg_data = seg.data;
            for (int i = 0; i < tex.total(); ++i)
            {
                //if (*seg_data++ > 0)
                if (seg.at<unsigned char>(i) > 0)
                {
                    (*tex_data)[0] = 0;
                    (*tex_data)[1] = 0;
                    (*tex_data)[2] = 240;
                }
                ++tex_data;
            }
        }

        cv::Size tex_size(nextPow2(img.cols), nextPow2(img.rows));
        cv::resize(tex, tex, tex_size, 0.0, 0.0, cv::INTER_CUBIC);

        // Initialize mesh
        Mesh tmp_mesh = mesh;
        tmp_mesh.tex = tex;
        tmp_mesh.uv = uv;
        tmp_mesh.normals = computeVertexNormals(tmp_mesh);

        // Render
        m_face_renderer->init(img.cols, img.rows);
        m_face_renderer->setProjection(K.at<float>(4));
        m_face_renderer->setMesh(tmp_mesh);

        cv::Mat pos_dir = (cv::Mat_<float>(4, 1) << -0.25f, -0.5f, -1, 0);
        cv::Mat ambient = (cv::Mat_<float>(4, 1) << 0.3f, 0.3f, 0.3f, 1);
        cv::Mat diffuse = (cv::Mat_<float>(4, 1) << 1.0f, 1.0f, 1.0f, 1);
        m_face_renderer->setLight(pos_dir, ambient, diffuse);
        m_face_renderer->render(vecR, vecT);
        m_face_renderer->clearLight();

        m_face_renderer->getFrameBuffer(out);

        // Overwrite black pixels with original pixels
        cv::Vec3b* out_data = (cv::Vec3b*)out.data;
        for (int i = 0; i < out.total(); ++i)
        {
            unsigned char b = (*out_data)[0];
            unsigned char g = (*out_data)[1];
            unsigned char r = (*out_data)[2];
            if (b == 0 && g == 0 && r == 0)
                *out_data = img.at<cv::Vec3b>(i);
            ++out_data;
        }

        return out;
    }

    cv::Mat FaceSwap::debugSourceLandmarks()
    {
        cv::Mat out = m_source_img.clone();
        sfl::render(out, m_src_landmarks);
        return out;
    }

    cv::Mat FaceSwap::debugTargetLandmarks()
    {
        cv::Mat out = m_target_img.clone();
        sfl::render(out, m_tgt_landmarks);
        return out;
    }

    cv::Mat FaceSwap::debugRender()
    {
        cv::Mat out = m_tgt_rendered_img.clone();

        // Overwrite black pixels with original pixels
        cv::Vec3b* out_data = (cv::Vec3b*)out.data;
        for (int i = 0; i < out.total(); ++i)
        {
            unsigned char b = (*out_data)[0];
            unsigned char g = (*out_data)[1];
            unsigned char r = (*out_data)[2];
            if (b == 0 && g == 0 && r == 0)
                *out_data = m_target_img.at<cv::Vec3b>(i);
            ++out_data;
        }

        return out;
    }

}   // namespace face_swap
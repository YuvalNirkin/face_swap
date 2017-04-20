#include "face_swap/cnn_3dmm.h"
#include "face_swap/cnn_3dmm_expr.h"

// iris_sfs
#include <FaceServices2.h>

// std
#include <exception>
#include <fstream>

// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  // debug

namespace face_swap
{
    CNN3DMMExpr::CNN3DMMExpr(const std::string& deploy_file,
		const std::string& caffe_model_file, const std::string& mean_file,
		const std::string& model_file, bool generic, bool with_expr,
		bool with_gpu, int gpu_device_id) :
        CNN3DMM(deploy_file, caffe_model_file, mean_file, 
			!generic, with_gpu, gpu_device_id),
        m_generic(generic), m_with_expr(with_expr)
    {
        // Load Basel 3DMM
        BaselFace::load_BaselFace_data(model_file.c_str());

        // Initialize face service
        fservice = std::make_unique<FaceServices2>();
    }

    CNN3DMMExpr::~CNN3DMMExpr()
    {
    }

    void CNN3DMMExpr::process(const cv::Mat& img, 
        const std::vector<cv::Point>& landmarks, cv::Mat& shape_coefficients,
        cv::Mat& tex_coefficients, cv::Mat& expr_coefficients,
        cv::Mat& vecR, cv::Mat& vecT, cv::Mat& K)
    {
        // Calculate shape and texture coefficients
        if (m_generic)
        {
            shape_coefficients = cv::Mat::zeros(99, 1, CV_32F);
            tex_coefficients = cv::Mat::zeros(99, 1, CV_32F);
        }
        else CNN3DMM::process(img, shape_coefficients, tex_coefficients);

        // Set up face service
        //fservice->setUp(img.cols, img.rows, 1000.0f);
        fservice->init(img.cols, img.rows, 1000.0f);

        // Convert landmarks format
        //cv::Mat_<double> LMs(68 * 2, 1);
        //for (int i = 0; i < 68; ++i) LMs.at<double>(i) = landmarks[i].x;
        //for (int i = 0; i < 68; ++i) LMs.at<double>(i + 68) = landmarks[i].y;
        cv::Mat LMs(68, 2, CV_32F);
        float* lms_data = (float*)LMs.data;
        for (int i = 0; i < 68; ++i)
        {
            *lms_data++ = (float)landmarks[i].x;
            *lms_data++ = (float)landmarks[i].y;
        }

        // Calculate pose and expression
        fservice->estimatePoseExpr(img, LMs, shape_coefficients, vecR, vecT, K, 
            expr_coefficients, "", m_with_expr);
    }
}   // namespace face_swap

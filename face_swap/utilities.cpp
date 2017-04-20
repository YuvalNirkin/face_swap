#include "face_swap/utilities.h"

// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

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


#ifndef FACE_SWAP_UTILITIES_H
#define FACE_SWAP_UTILITIES_H

#include "face_swap/basel_3dmm.h"
#include <opencv2/core.hpp>

namespace face_swap
{
    /** @brief Create a rotation matrix from Euler angles.
    The rotation values are given in radians and estimated using the RPY convention.
    Yaw is applied first to the model, then pitch, then roll (R * P * Y * vertex).
    @param x Pitch. rotation around the X axis [Radians].
    @param y Yaw. rotation around the Y axis [Radians].
    @param z Roll. rotation around the Z axis [Radians].
    */
    cv::Mat euler2RotMat(float x, float y, float z);

    cv::Mat euler2RotMat(const cv::Mat& euler);

    cv::Mat createModelView(const cv::Mat& euler, const cv::Mat& translation);

    cv::Mat createOrthoProj4x4(const cv::Mat& euler, const cv::Mat& translation, 
        int width, int height);

    cv::Mat createOrthoProj3x4(const cv::Mat& euler, const cv::Mat& translation, 
        int width, int height);

    cv::Mat createPerspectiveProj3x4(const cv::Mat& euler,
        const cv::Mat& translation, const cv::Mat& K);

    void renderWireframe(cv::Mat& img, const Mesh& mesh, const cv::Mat& P,
        float scale = 1, const cv::Scalar& color = cv::Scalar(0, 255, 0));

    void renderWireframe(cv::Mat& img, const Mesh& mesh, 
        const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K,
        float scale = 1, const cv::Scalar& color = cv::Scalar(0, 255, 0));

    void renderWireframeUV(cv::Mat& img, const Mesh& mesh, const cv::Mat& uv,
        const cv::Scalar& color = cv::Scalar(0, 255, 0));

    void renderBoundary(cv::Mat& img, const Mesh& mesh,
        const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K,
        const cv::Scalar& color = cv::Scalar(255, 0, 0));

    bool is_ccw(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3);

    inline unsigned int nextPow2(unsigned int x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    cv::Mat computeFaceNormals(const Mesh& mesh);

    cv::Mat computeVertexNormals(const Mesh& mesh);

}   // namespace face_swap

#endif // FACE_SWAP_UTILITIES_H
    
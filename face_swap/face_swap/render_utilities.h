#ifndef FACE_SWAP_RENDER_UTILITIES_H
#define FACE_SWAP_RENDER_UTILITIES_H

#include "face_swap/face_swap_engine.h"
#include <opencv2/core.hpp>

namespace face_swap
{
	FACE_SWAP_EXPORT void renderWireframe(cv::Mat& img, const Mesh& mesh, const cv::Mat& P,
        float scale = 1, const cv::Scalar& color = cv::Scalar(0, 255, 0));

	FACE_SWAP_EXPORT void renderWireframe(cv::Mat& img, const Mesh& mesh,
        const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K,
        float scale = 1, const cv::Scalar& color = cv::Scalar(0, 255, 0));

	FACE_SWAP_EXPORT void renderWireframeUV(cv::Mat& img, const Mesh& mesh, const cv::Mat& uv,
        const cv::Scalar& color = cv::Scalar(0, 255, 0));

	FACE_SWAP_EXPORT void renderBoundary(cv::Mat& img, const Mesh& mesh,
        const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K,
        const cv::Scalar& color = cv::Scalar(255, 0, 0));

	FACE_SWAP_EXPORT void renderSegmentation(cv::Mat& img, const cv::Mat& seg, float alpha = 0.5f,
		const cv::Scalar& color = cv::Scalar(255, 0, 0));

	/** @brief Render a textured mesh.
	All the polygons must be triangles. The texture can be either in BGR or BGRA format.
	@param img The image to render the mesh to.
	@param mesh The mesh to render.
	@param rvec Euler angles (Pitch, Yaw, Roll) [3x1].
	@param tvec translation vector [3x1].
	@param K Intrinsic camera matrix [3x3].
	@param depth Output depth map (same size as img, in floating point format).
	*/
	FACE_SWAP_EXPORT void renderMesh(cv::Mat& img, const Mesh& mesh,
		const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K,
		cv::Mat& depthbuf, int ss = 1);

	/** @brief Render depth map.
	The depth values are inversed and rendered as a heat map, hotter values correspond to closer pixels.
	Pixels of infinite values (MAX_FLOAT) are considered background.
	@param depth_map The depth map to render.
	@return Rendered depth map.
	*/
	FACE_SWAP_EXPORT cv::Mat renderDepthMap(const cv::Mat& depth_map);

	FACE_SWAP_EXPORT cv::Mat renderImagePipe(const std::vector<cv::Mat>& images, int padding = 4,
		const cv::Scalar& border_color = cv::Scalar(255, 255, 255));

	FACE_SWAP_EXPORT void overlayImage(cv::Mat& img, const cv::Mat& overlay,
		const cv::Point& loc, const cv::Mat& mask = cv::Mat());

	FACE_SWAP_EXPORT void renderImageOverlay(cv::Mat& img, const cv::Rect& bbox,
		const cv::Mat& src, const cv::Mat& tgt,
		const cv::Scalar border_color = cv::Scalar(255.0, 255.0, 255.0));

	FACE_SWAP_EXPORT bool is_ccw(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3);

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

	FACE_SWAP_EXPORT cv::Mat computeFaceNormals(const Mesh& mesh);

	FACE_SWAP_EXPORT cv::Mat computeVertexNormals(const Mesh& mesh);

}   // namespace face_swap

#endif // FACE_SWAP_RENDER_UTILITIES_H
    
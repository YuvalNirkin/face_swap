#ifndef FACE_SWAP_UTILITIES_H
#define FACE_SWAP_UTILITIES_H

#include "face_swap/render_utilities.h"
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
	FACE_SWAP_EXPORT cv::Mat euler2RotMat(float x, float y, float z);

	FACE_SWAP_EXPORT cv::Mat euler2RotMat(const cv::Mat& euler);

	FACE_SWAP_EXPORT cv::Mat createModelView(const cv::Mat& euler, const cv::Mat& translation);

	FACE_SWAP_EXPORT cv::Mat createOrthoProj4x4(const cv::Mat& euler, const cv::Mat& translation,
        int width, int height);

	FACE_SWAP_EXPORT cv::Mat createOrthoProj3x4(const cv::Mat& euler, const cv::Mat& translation,
        int width, int height);

	FACE_SWAP_EXPORT cv::Mat createPerspectiveProj3x4(const cv::Mat& euler,
        const cv::Mat& translation, const cv::Mat& K);

	FACE_SWAP_EXPORT cv::Mat refineMask(const cv::Mat& img, const cv::Mat& mask);

	FACE_SWAP_EXPORT void horFlipLandmarks(std::vector<cv::Point>& landmarks, int width);

	/**	Generate texture for the mesh based on the image size, intrinsic and
	extrinsic transformations.
	@param[in] mesh The mesh to generate the texture for.
	@param[in] img The image for the texture
	@param[in] seg The segmentation for the texture (will be used as the
	texture's alpha channel).
	@param[in] vecR Mesh's rotation vector [Euler angles].
	@param[in] vecT Mesh's translation vector.
	@param[in] K Camera intrinsic parameters.
	@param[out] tex Generated texture image.
	@param[out] uv Generated texture coordinates.
	*/
	FACE_SWAP_EXPORT void generateTexture(const Mesh& mesh, const cv::Mat& img, const cv::Mat& seg,
		const cv::Mat& vecR, const cv::Mat& vecT, const cv::Mat& K,
		cv::Mat& tex, cv::Mat& uv);

	/**	Generate texture coordinates for the mesh based on the image size,
	intrinsic and extrinsic transformations.
	@param[in] mesh The mesh to generate the texture coordinates for.
	@param[in] img_size The image size that the texture coordinates will be
	relative to.
	@param[in] vecR Mesh's rotation vector [Euler angles].
	@param[in] vecT Mesh's translation vector.
	@param[in] K Camera intrinsic parameters.
	@return n X 2 matrix where n is the number of vertices. Each row contain the
	xy texture coordinate of the corresponding vertex.
	*/
	FACE_SWAP_EXPORT cv::Mat generateTextureCoordinates(const Mesh& mesh, const cv::Size& img_size,
		const cv::Mat& vecR, const cv::Mat& vecT, const cv::Mat& K);

	/**	Blend the source and destination images based on a segmentation mask.
	@param[in] src The source image.
	@param[in] dst The destination image.
	@param[in] mask Object segmentation mask.
	@return The src and dst blended image.
	*/
	FACE_SWAP_EXPORT cv::Mat blend(const cv::Mat& src, const cv::Mat& dst,
		const cv::Mat& mask = cv::Mat());

	/**	Read saved face data.
	@param[in] path Path to an image or a directory. If the path is an image,
	a directory will be created with the name of the image without the extension.
	@param[in] face_data Includes all the images and intermediate data for the specific face.
	@return true if cache was loaded, false otherwise.
	*/
	FACE_SWAP_EXPORT bool readFaceData(const std::string& path, FaceData& face_data);

	/**	Write face data to file.
	@param[in] path Path to an image or a directory. If the path is an image,
	a directory will be created with the name of the image without the extension.
	@param[in] face_data Includes all the images and intermediate data for the specific face.
	@param[in] overwrite Toggle whether to overwrite existing cache or leave as it is.
	@return true if the face data was written to file, else false.
	*/
	FACE_SWAP_EXPORT bool writeFaceData(const std::string& path, const FaceData& face_data,
		bool overwrite = false);


}   // namespace face_swap

#endif // FACE_SWAP_UTILITIES_H
    
/** @file
@brief Face landmarks utility functions.
*/

#ifndef FACE_SWAP_LANDMARKS_UTILITIES_H
#define FACE_SWAP_LANDMARKS_UTILITIES_H

#include "face_detection_landmarks.h"
#include "face_swap/face_swap_export.h"

// OpenCV
#include <opencv2/core.hpp>

namespace face_swap
{
	/** @brief Render landmarks.
	@param img The image that the landmarks will be rendered on.
	@param landmarks The landmark points to render.
	@param drawLabels if true, for each landmark, it's 0 based index will be
	rendererd as a label.
	@param color Line/point and label color.
	@param thickness Line/point thickness.
	*/
	FACE_SWAP_EXPORT void render(cv::Mat& img, const std::vector<cv::Point>& landmarks,
		bool drawLabels = false, const cv::Scalar& color = cv::Scalar(0, 255, 0),
		int thickness = 1);

	/** @brief Render bounding box.
	@param img The image that the bounding box will be rendered on.
	@param bbox The bounding box rectangle to render.
	@param color Line color.
	@param thickness Line thickness.
	*/
	FACE_SWAP_EXPORT void render(cv::Mat& img, const cv::Rect& bbox,
		const cv::Scalar& color = cv::Scalar(0, 255, 0), int thickness = 1);

	/** @brief Render face's bounding box and landmarks.
	@param img The image that the face will be rendered on.
	@param face The face to render.
	@param drawLabels if true, for each landmark, it's 0 based index will be
	rendererd as a label.
	@param bbox_color Bounding box line color.
	@param landmarks_color Landmarks line/point and label color.
	@param thickness Line/point thickness.
    @param fontScale The size of the font for the labels.
	*/
	FACE_SWAP_EXPORT void render(cv::Mat& img, const Face& face, bool drawLabels = false,
		const cv::Scalar& bbox_color = cv::Scalar(0, 0, 255),
		const cv::Scalar& landmarks_color = cv::Scalar(0, 255, 0), int thickness = 1,
		double fontScale = 1.0);

	/** @brief Render all frame faces including bounding boxs and landmarks.
	@param img The image that the faces will be rendered on.
	@param frame The frame to render.
    @param drawIDs if true, the 0 based id will be rendererd as a label.
	@param drawLabels if true, for each landmark, it's 0 based index will be
	rendererd as a label.
	@param bbox_color Bounding box line color.
	@param landmarks_color Landmarks line/point and label color.
	@param thickness Line/point thickness.
    @param fontScale The size of the font for the labels.
	*/
	FACE_SWAP_EXPORT void render(cv::Mat& img, const std::vector<Face>& faces, bool drawLabels = false,
		const cv::Scalar& bbox_color = cv::Scalar(0, 0, 255),
		const cv::Scalar& landmarks_color = cv::Scalar(0, 255, 0), int thickness = 1,
		double fontScale = 1.0);

	/** @brief Get the main face index in a frame.
	*/
	FACE_SWAP_EXPORT int getMainFaceID(const std::vector<Face>& faces, const cv::Size& frame_size);

    /** @brief Get the face's left eye center position (right eye in the image).
    @param landmarks 68 face points.
    */
	FACE_SWAP_EXPORT cv::Point2f getFaceLeftEye(const std::vector<cv::Point>& landmarks);

    /** @brief Get the face's right eye center position (left eye in the image).
    @param landmarks 68 face points.
    */
	FACE_SWAP_EXPORT cv::Point2f getFaceRightEye(const std::vector<cv::Point>& landmarks);

    /** @brief Get the face's vertical angle [radians].
    The angles are in the range [-75/180*pi, 75/180*pi].
    When the face is looking up the angle will be positive and when it is
    looking down it will be negative.
    @param landmarks 68 face points.
    */
	FACE_SWAP_EXPORT float getFaceApproxVertAngle(const std::vector<cv::Point>& landmarks);

    /** @brief Get the face's horizontal angle [radians].
    The angles are in the range [-75/180*pi, 75/180*pi].
    When the face is looking right (left in the image) the angle will be positive and
    when it is looking left (right in the image) it will be negative.
    @param landmarks 68 face points.
    */
	FACE_SWAP_EXPORT float getFaceApproxHorAngle(const std::vector<cv::Point>& landmarks);

    /** @brief Get the face's tilt angle [radians].
    The angles are in the range [-75/180*pi, 75/180*pi].
    When the face is tilting left (right in the image) the angle will be positive and
    when it is tilting right (left in the image) it will be negative.
    @param landmarks 68 face points.
    */
	FACE_SWAP_EXPORT float getFaceApproxTiltAngle(const std::vector<cv::Point>& landmarks);

    /** @brief Get the face's euler angles [radians].
    The angles are in the range [-75/180*pi, 75/180*pi].
    @param landmarks 68 face points.
    @return Return a vector with the 3 euler angles.
    The x axis represents vertical rotation angle, up is positive.
    The y axis represents horizontal rotation angle, right is positive.
    The z axis represents tilt rotation angle, left is positive.
    */
	FACE_SWAP_EXPORT cv::Point3f getFaceApproxEulerAngles(const std::vector<cv::Point>& landmarks);

    /** @brief Get face bounding box from landmarks.
    @param landmarks Face points.
    @param frameSize The size of the image.
    @param square Make the bounding box square (limited to frame boundaries).
    */
	FACE_SWAP_EXPORT cv::Rect getFaceBBoxFromLandmarks(const std::vector<cv::Point>& landmarks,
        const cv::Size& frameSize, bool square);

}   // namespace face_swap

#endif	// FACE_SWAP_LANDMARKS_UTILITIES_H

#ifndef FACE_SWAP_BASEL_3DMM_H
#define FACE_SWAP_BASEL_3DMM_H

#include "face_swap/face_swap_export.h"

// Includes
#include <opencv2/core.hpp>

#include <string>

namespace face_swap
{
	/** Represents a 3D renderable shape.
	*/
    struct FACE_SWAP_EXPORT Mesh
    {
		/**	Save mesh to a ply file.
		@param mesh The mesh to write to file.
		@param ply_file The name of the file (.ply)
		*/
        static void save_ply(const Mesh& mesh, const std::string& ply_file);

        cv::Mat vertices;
        cv::Mat colors;
        cv::Mat faces;
        cv::Mat uv;
        cv::Mat tex;
        cv::Mat normals;
    };

	/**	Represents Basel's 3D Morphable Model.
	This is a PCA model of 3D faces that includes shape, texture and expressions.
	Based on the paper:
	A 3D Face Model for Pose and Illumination Invariant Face Recognition, 
	P. Paysan and R. Knothe and B. Amberg and S. Romdhani and T. Vetter.
	*/
    struct Basel3DMM
    {
		/**	Sample a mesh from the PCA model.
		@param[in] shape_coefficients PCA shape coefficients.
		@param[in] tex_coefficients PCA texture coefficients.
		*/
        Mesh sample(const cv::Mat& shape_coefficients,
            const cv::Mat& tex_coefficients);

		/**	Sample a mesh from the PCA model.
		@param[in] shape_coefficients PCA shape coefficients.
		@param[in] tex_coefficients PCA texture coefficients.
		@param[in] expr_coefficients PCA expression coefficients.
		*/
        Mesh sample(const cv::Mat& shape_coefficients,
            const cv::Mat& tex_coefficients, const cv::Mat& expr_coefficients);

		/**	Load a Basel's 3DMM from file.
		@param model_file Path to 3DMM file (.h5).
		*/
        static Basel3DMM load(const std::string& model_file);

        cv::Mat faces;
        cv::Mat shapeMU, shapePC, shapeEV;
        cv::Mat texMU, texPC, texEV;
        cv::Mat exprMU, exprPC, exprEV;
    };

}   // namespace face_swap

#endif // FACE_SWAP_BASEL_3DMM_H

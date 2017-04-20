#ifndef FACE_SWAP_FACE_RENDERER_H
#define FACE_SWAP_FACE_RENDERER_H

#include "face_swap/basel_3dmm.h"
#include <opencv2/core.hpp>

namespace face_swap
{
	/**	Simple renderer for rendering textured face meshes.
	*/
    class FaceRenderer
    {
    public:

		/** Create an instance of FaceRenderer.
		*/
        FaceRenderer();

		/**	Destrcutor.
		*/
        ~FaceRenderer();

		/**	Initialize buffers and viewport.
		@param width The width of the viewport [pixels].
		@param height The height of the viewport [pixels].
		@param force Force initialization even if the resolution is the same as the last one.
		*/
        void init(int width, int height, bool force = false);

		/** Clear all allocated data.
		*/
        void clear();

		/** Clear the frame buffer.
		*/
        void clearFrameBuffer();

		/** Clear the mesh.
		*/
        void clearMesh();

		/** Set viewport resolution.
		@param width The width of the viewport [pixels].
		@param height The height of the viewport [pixels].
		*/
        void setViewport(int width, int height);

		/** Set projection.
		@param f Focal length.
		@param z_near The near plane distance on Z axis.
		@param z_far The far plane distance on Z axis.
		*/
        void setProjection(float f, float z_near = 10, float z_far= 10000);

		/**	Set mesh to render.
		*/
        void setMesh(const Mesh& mesh);

		/**	Render with the specified pose.
		@param vecR Rotation vector [Euler angles].
		@param vecT Translation vector.
		*/
        void render(const cv::Mat& vecR, const cv::Mat& vecT);

		/**	Get frame buffer (should be called after render).
		*/
        void getFrameBuffer(cv::Mat& img);

		/**	Get depth buffer (should be called after render).
		*/
        void getDepthBuffer(cv::Mat& depth);

		/**	Set light position\direction and color.
		@param pos_dir Light's position or direction in OpenGL's format.
		@param ambient Light's ambient color in OpenGL's format.
		@param diffuse Light's diffuse color in OpenGL's format.
		*/
        void setLight(const cv::Mat& pos_dir, const cv::Mat& ambient, const cv::Mat& diffuse);

		/**	Clear previously set light.
		*/
        void clearLight();

    private:
        void drawMesh();

    private:
        int m_width = 0, m_height = 0;
        int m_tex_width = 0, m_tex_height = 0;
        unsigned int m_dynamic_tex_id = 0;
        unsigned int m_fbo = 0;
        unsigned int m_depth_rb = 0;

        // Mesh
        unsigned int m_mesh_vert_id = 0;
        unsigned int m_mesh_uv_id = 0;
        unsigned int m_mesh_faces_id = 0;
        unsigned int m_mesh_total_faces = 0;
        unsigned int m_mesh_total_vertices = 0;
        unsigned int m_mesh_tex_id = 0;
        unsigned int m_mesh_tex_width = 0;
        unsigned int m_mesh_tex_height = 0;
        unsigned int m_mesh_normals_id = 0;
    };

}   // namespace face_swap

#endif //FACE_SWAP_FACE_RENDERER_H
    
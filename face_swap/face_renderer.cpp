#include "face_swap/face_renderer.h"

#include <GL/glew.h>

namespace face_swap
{
    FaceRenderer::FaceRenderer()
    {
    }

    FaceRenderer::~FaceRenderer()
    {
        clear();
    }

    void FaceRenderer::init(int width, int height, bool force)
    {
        if (!force && m_width == width && m_height == height) return;
        //clearFrameBuffer();
        m_width = width;
        m_height = height;

        // Create a texture
        if (m_dynamic_tex_id == 0)
        {
            glGenTextures(1, &m_dynamic_tex_id);
            glBindTexture(GL_TEXTURE_2D, m_dynamic_tex_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        }
        
        // Allocate texture storage
        glBindTexture(GL_TEXTURE_2D, m_dynamic_tex_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

        // Create framebuffer
        if(m_fbo == 0)
            glGenFramebuffersEXT(1, &m_fbo);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);

        // Attach 2D texture to this FBO
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_dynamic_tex_id, 0);

        // Create renderbuffer
        if(m_depth_rb == 0)
            glGenRenderbuffersEXT(1, &m_depth_rb);
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, m_depth_rb);
        glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, width, height);

        // Attach depth buffer to FBO
        glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, m_depth_rb);

        // Clean up
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

        // Setup viewport
        glViewport(0, 0, width, height);

        // Settings
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_TEXTURE_2D);

        // Enable blending
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    void FaceRenderer::clear()
    {
        clearMesh();
        clearFrameBuffer();
    }

    void FaceRenderer::clearFrameBuffer()
    {
        if (m_dynamic_tex_id > 0)
            glDeleteTextures(1, &m_dynamic_tex_id);
        if (m_depth_rb > 0)
            glDeleteRenderbuffersEXT(1, &m_depth_rb);

        // Bind 0, which means render to back buffer, as a result, fb is unbound
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        if (m_fbo > 0)
            glDeleteFramebuffersEXT(1, &m_fbo);

        m_dynamic_tex_id = m_depth_rb = m_fbo = 0;
    }

    void FaceRenderer::clearMesh()
    {
        if (m_mesh_vert_id > 0)
            glDeleteBuffers(1, &m_mesh_vert_id);
        if (m_mesh_uv_id > 0)
            glDeleteBuffers(1, &m_mesh_uv_id);
        if (m_mesh_faces_id > 0)
            glDeleteBuffers(1, &m_mesh_faces_id);
        if (m_mesh_normals_id > 0)
            glDeleteBuffers(1, &m_mesh_normals_id);
        if(m_mesh_tex_id > 0)
            glDeleteTextures(1, &m_mesh_tex_id);

        m_mesh_tex_id = m_mesh_total_faces = m_mesh_vert_id = 
            m_mesh_uv_id = m_mesh_normals_id = m_mesh_faces_id = 0;
    }

    void FaceRenderer::setViewport(int width, int height)
    {
        glViewport(0, 0, width, height);
        m_width = width;
        m_height = height;
    }

    void FaceRenderer::setProjection(float f, float z_near, float z_far)
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        //flip y, because image y increases downward, whereas camera y increases upward
        glScaled(1, -1, 1);
        double fovy = 2.0 * atan((double)m_height / (2.0 * (double)f)) * 180.0 / CV_PI;
        double aspect = (double)m_width / (double)m_height;
        gluPerspective(fovy, aspect, (double)z_near, (double)z_far);
    }

    void FaceRenderer::setMesh(const Mesh& mesh)
    {
        if (mesh.vertices.empty() || mesh.uv.empty() || mesh.faces.empty()) return;

        // Initialize vertices
        if(m_mesh_vert_id == 0)
            glGenBuffers(1, &m_mesh_vert_id);
        m_mesh_total_vertices = mesh.vertices.rows; 
        glBindBuffer(GL_ARRAY_BUFFER, m_mesh_vert_id);
        glBufferData(GL_ARRAY_BUFFER, mesh.vertices.total() * sizeof(float), 
            mesh.vertices.data, GL_STATIC_DRAW);

        // Initialize texture coordinates
        if(m_mesh_uv_id == 0)
            glGenBuffers(1, &m_mesh_uv_id);
        glBindBuffer(GL_ARRAY_BUFFER, m_mesh_uv_id);
        glBufferData(GL_ARRAY_BUFFER, mesh.uv.total() * sizeof(float),
            mesh.uv.data, GL_STATIC_DRAW);

        // Initialize faces (triangles only)
        if (m_mesh_faces_id == 0)
            glGenBuffers(1, &m_mesh_faces_id);
        m_mesh_total_faces = mesh.faces.rows;
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_mesh_faces_id);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.faces.total() * sizeof(unsigned short),
            mesh.faces.data, GL_STATIC_DRAW);

        // Initialize normals
        if (mesh.normals.empty())
        {
            if (m_mesh_normals_id > 0)
            {
                glDeleteBuffers(1, &m_mesh_normals_id);
                m_mesh_normals_id = 0;
            }
        }
        else
        {
            if (m_mesh_normals_id == 0)
                glGenBuffers(1, &m_mesh_normals_id);
            glBindBuffer(GL_ARRAY_BUFFER, m_mesh_normals_id);
            glBufferData(GL_ARRAY_BUFFER, mesh.vertices.total() * sizeof(float),
                mesh.normals.data, GL_STATIC_DRAW);
        }

        // Initialize texture
        if (m_mesh_tex_id == 0)
        {
            glGenTextures(1, &m_mesh_tex_id);
            glBindTexture(GL_TEXTURE_2D, m_mesh_tex_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        }

        // Allocate texture storage or upload pixel data
        glBindTexture(GL_TEXTURE_2D, m_mesh_tex_id);
        if (m_mesh_tex_width != mesh.tex.cols || m_mesh_tex_height != mesh.tex.rows)
        {
            m_mesh_tex_width = mesh.tex.cols;
            m_mesh_tex_height = mesh.tex.rows;
            if (mesh.tex.channels() == 3)   // BGR
            {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mesh.tex.cols, mesh.tex.rows,
                    0, GL_BGR, GL_UNSIGNED_BYTE, mesh.tex.data);
            }
            else    // BGRA
            {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mesh.tex.cols, mesh.tex.rows,
                    0, GL_BGRA, GL_UNSIGNED_BYTE, mesh.tex.data);
            } 
        }
        else
        {
            if (mesh.tex.channels() == 3)   // BGR
            {
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mesh.tex.cols, mesh.tex.rows,
                    GL_BGR, GL_UNSIGNED_BYTE, mesh.tex.data);
            }
            else    // BGRA
            {
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mesh.tex.cols, mesh.tex.rows,
                    GL_BGRA, GL_UNSIGNED_BYTE, mesh.tex.data);
            }
        }
    }

    void FaceRenderer::render(const cv::Mat& vecR, const cv::Mat& vecT)
    {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();

        //Translations
        glTranslatef(vecT.at<float>(0), vecT.at<float>(1), vecT.at<float>(2));

        //Axis angle
        //rot = { tRx, tRy, tRz } , angle = ||rot||
        float rx = vecR.at<float>(0);
        float ry = vecR.at<float>(1);
        float rz = vecR.at<float>(2);
        float angle = (float)sqrt(rx * rx + ry * ry + rz * rz);
        if(std::abs(angle) > 1e-6f)
            glRotatef(angle / CV_PI*180.0f, rx / angle, ry / angle, rz / angle);

        drawMesh();

        glPopMatrix();
        glFlush();
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);//
    }

    void FaceRenderer::getFrameBuffer(cv::Mat& img)
    {
        img.create(m_height, m_width, CV_8UC3);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
        glReadPixels(0, 0, m_width, m_height, GL_BGR, GL_UNSIGNED_BYTE,
            (GLvoid*)img.data);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    }

    void FaceRenderer::getDepthBuffer(cv::Mat& depth)
    {
        depth.create(m_height, m_width, CV_32F);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
        glReadPixels(0, 0, m_width, m_height, GL_DEPTH_COMPONENT, GL_FLOAT,
            (GLvoid*)depth.data);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    }

    void FaceRenderer::setLight(const cv::Mat& pos_dir, const cv::Mat& ambient,
        const cv::Mat& diffuse)
    {
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glShadeModel(GL_SMOOTH);

        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (float*)ambient.data);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (float*)diffuse.data);
        glLightfv(GL_LIGHT0, GL_POSITION, (float*)pos_dir.data);
        float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, white);
    }

    void FaceRenderer::clearLight()
    {
        glDisable(GL_LIGHTING);
        glDisable(GL_LIGHT0);
    }

    void FaceRenderer::drawMesh()
    {
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        if (m_mesh_normals_id > 0)
            glEnableClientState(GL_NORMAL_ARRAY);

        glBindTexture(GL_TEXTURE_2D, m_mesh_tex_id);

        glBindBuffer(GL_ARRAY_BUFFER, m_mesh_vert_id);
        glVertexPointer(3, GL_FLOAT, 0, NULL);

        glBindBuffer(GL_ARRAY_BUFFER, m_mesh_uv_id);
        glTexCoordPointer(2, GL_FLOAT, 0, NULL);

        if (m_mesh_normals_id > 0)
        {
            glBindBuffer(GL_ARRAY_BUFFER, m_mesh_normals_id);
            glNormalPointer(GL_FLOAT, 0, NULL);
        }

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_mesh_faces_id);
        
        glDrawElements(GL_TRIANGLES, m_mesh_total_faces * 3, GL_UNSIGNED_SHORT, NULL);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);//
        glBindBuffer(GL_ARRAY_BUFFER, 0);//

        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        if (m_mesh_normals_id > 0)
            glDisableClientState(GL_NORMAL_ARRAY);
    }

}   // namespace face_swap

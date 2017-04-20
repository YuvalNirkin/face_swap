/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#pragma once
#include "cv.h"
#include "highgui.h"

#define RENDER_PARAMS_COUNT 21

#define RENDER_PARAMS_R 0
#define RENDER_PARAMS_T 3
#define RENDER_PARAMS_AMBIENT	6
#define RENDER_PARAMS_DIFFUSE	9
#define RENDER_PARAMS_LDIR		12
#define RENDER_PARAMS_CONTRAST	14
#define RENDER_PARAMS_GAIN		15
#define RENDER_PARAMS_OFFSET	18
#define RENDER_PARAMS_SPECULAR	21
#define RENDER_PARAMS_SHINENESS	24

#define RENDER_PARAMS_AMBIENT_DEFAULT	0.5f
#define RENDER_PARAMS_DIFFUSE_DEFAULT	0.5f
#define RENDER_PARAMS_CONTRAST_DEFAULT	1.0f
#define RENDER_PARAMS_GAIN_DEFAULT		1.0f
#define RENDER_PARAMS_OFFSET_DEFAULT	0.0f
#define RENDER_PARAMS_SPECULAR_DEFAULT	80.0f
#define RENDER_PARAMS_SHINENESS_DEFAULT	16.0f

class RenderServices
{
	bool triangleNormalFromVertex(cv::Mat shape, cv::Mat faces, int face_id, int vertex_id, float &nx, float &ny, float &nz);
	float triangleNormal(cv::Mat shape, cv::Mat faces, int face_id, float &nx, float &ny, float &nz);
public:
	bool estimateFaceNormals(cv::Mat shape, cv::Mat faces, cv::Mat &normals);
	bool estimateFaceNormals(cv::Mat shape, cv::Mat faces, cv::Mat &normals, cv::Mat &centers, cv::Mat &areas);
	bool estimateVertexNormals(cv::Mat shape, cv::Mat faces, cv::Mat &normals);
	
	bool estimatePointColor(cv::Mat &points, cv::Mat &texs, cv::Mat &normals, std::vector<int> &inds, cv::Mat &visible, cv::Mat &noShadow, float* render_model, cv::Mat &colors);
	bool estimatePointColor(cv::Mat &points, cv::Mat &texs, cv::Mat &normals, std::vector<int> &inds, cv::Mat &noShadow, float* render_model, cv::Mat &colors);
	bool estimateColor(cv::Mat shape, cv::Mat tex, cv::Mat faces, bool* visible, bool* noShadow, float* render_model, cv::Mat &colors);
	bool estimateColor(cv::Mat shape, cv::Mat tex, cv::Mat faces, bool* visible, bool* noShadow, float* render_model, cv::Mat &colors, cv::Mat &normals);
};

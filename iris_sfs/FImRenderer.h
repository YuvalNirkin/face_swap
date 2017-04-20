/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#pragma once
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>

//Glew needs to be first
//#include <GL/glew.h>
//#include <GL/gl.h>
//#include <GL/glu.h>
//#include <GL/glext.h>
//#include <GL/osmesa.h>
#include "FTModel.h"
#include "FBRender.h"

class FImRenderer
{	
public:
	FBRender *render_;
	Face* face_;
	cv::Mat img_;
	float zNear, zFar;
	FImRenderer(cv::Mat img);
    void init(const cv::Mat& img);  // Yuval
	void loadPLYFile(char* filename, bool useTexture = false);
	void loadMesh(cv::Mat shape, cv::Mat tex, cv::Mat faces);
	void computeTexture(float *rot, float *t, float f);
	void computeTexture(cv::Mat rVec, cv::Mat tVec, float f);
	void computeNormals();
	void loadModel(char* meshFile = 0);
	void setIm(cv::Mat inIm);
	void mapRendering(float* r, float* t, float f, cv::Mat *im, cv::Mat *mask);
	void render( float *r, float *t, float f, cv::Mat &color, cv::Mat &depth );
	void unmapRendering();
	void resetRenderer(Face* face2, bool resetTexture = false, float *rot = 0, float *t = 0, float f = 1000);
	
	void copyFaces(cv::Mat faces);
	void copyShape(cv::Mat shape);
	void copyColors(cv::Mat colors);
	void copyNormals(cv::Mat normals);
	~FImRenderer(void);
};

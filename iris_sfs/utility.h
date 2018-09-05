/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#pragma once
//#include "cv.h"
#include "highgui.h"
#include <string>
#include <fstream>
#include <Eigen/Dense>

int splittext(char* str, char** pos);

Eigen::Matrix3Xd* toMatrix3Xd(cv::Mat mat);
cv::Mat toMat(Eigen::Matrix3Xd emat);
void qr(cv::Mat input, cv::Mat &q, cv::Mat &r);

cv::Vec3b avSubMatValue8UC3( const CvPoint2D64f* pt, const cv::Mat* mat );
cv::Vec3d avSubMatValue8UC3_2( const CvPoint2D64f* pt, const cv::Mat* mat );

double avSubPixelValue64F( const CvPoint2D64f* pt, const IplImage* img );

double avSubPixelValue32F( const CvPoint2D64f* pt, const IplImage* img );

double avSubPixelValue8U( const CvPoint2D64f* pt, const IplImage* img );

double avSubMatValue64F( const CvPoint2D64f* pt, const cv::Mat* mat );

double avSubMatValue32F( const CvPoint2D64f* pt, const cv::Mat* mat );

double avSubMatValue8U( const CvPoint2D64f* pt, const cv::Mat* mat );

void write_ply4(char* outname, cv::Mat mat_Depth,std::vector<cv::Vec3b> colors,std::vector<cv::Vec3i> faces);

void write_ply(char* outname, std::vector<cv::Point3f> points);
void write_ply(char* outname, cv::Mat mat_Depth, std::vector<cv::Vec3b> colors);
void write_ply(char* outname, cv::Mat mat_Depth);
void write_ply(char* outname, cv::Mat mat_Depth, cv::Mat mat_Faces);
void write_ply(char* outname, cv::Mat mat_Depth, cv::Mat mat_Color, cv::Mat mat_Faces);
void write_ply(char* outname, cv::Mat mat_Depth,std::vector<cv::Vec3b> colors,cv::Mat faces);

//void write_ply(char* outname, bool* visible, cv::Mat mat_Depth,std::vector<cv::Vec3b> colors,cv::Mat faces);
void write_ply(char* outname, int count, bool* visible, float* points_);
void write_ply(char* outname, int count, float* points_);
void write_plyF(char* outname, cv::Mat mat_Depth,std::vector<cv::Vec3b> colors,int nFaces, unsigned* faces);
void write_plyFloat(char* outname, cv::Mat mat_Depth);
void write_plyFloat(char* outname, cv::Mat mat_Depth, cv::Mat mat_tex, cv::Mat faces);

void write_ply(char* outname, Eigen::Matrix3Xd*);
//void write_ply(char* outname, int count, float* points_, float* colors_);
cv::Mat skew(cv::Mat v1);
void groundScale(cv::Mat input, cv::Mat &output, float bgThresh, float gapPc);
cv::Mat findRotation(cv::Mat v1, cv::Mat v2);

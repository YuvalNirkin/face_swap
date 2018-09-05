/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#pragma once
//#include "cv.h"
#include "highgui.h"
//#include "FTModel.h"
#include "BaselFace.h"

class BaselFaceEstimator
{
	cv::Mat coef2object(cv::Mat weight, cv::Mat MU, cv::Mat PCs, cv::Mat EV);
	cv::Mat coef2objectParts(cv::Mat &weight, cv::Mat &MU, cv::Mat &PCs, cv::Mat &EV);

public:
	BaselFaceEstimator(/*std::string baselFile = ""*/);
	cv::Mat getFaces();
	cv::Mat getFaces_fill();
	cv::Mat getShape(cv::Mat weight, cv::Mat exprWeight = cv::Mat());
	cv::Mat getShape2(cv::Mat weight, cv::Mat exprWeight = cv::Mat());
	cv::Mat getTexture(cv::Mat weight);
	cv::Mat getTexture2(cv::Mat weight);
	cv::Mat getShapeParts(cv::Mat weight, cv::Mat exprWeight = cv::Mat());
	cv::Mat getTextureParts(cv::Mat weight);
	cv::Mat getLM(cv::Mat shape, float yaw);
	cv::Mat getLMByAlpha(cv::Mat alpha, float yaw, std::vector<int> inds, cv::Mat exprWeight = cv::Mat());
	cv::Mat getLMByAlphaParts(cv::Mat alpha, float yaw, std::vector<int> inds, cv::Mat exprWeight = cv::Mat());
	cv::Mat getTriByAlpha(cv::Mat alpha, std::vector<int> inds, cv::Mat exprWeight = cv::Mat());
	cv::Mat getTriByAlphaParts(cv::Mat alpha, std::vector<int> inds, cv::Mat exprWeight = cv::Mat());
	cv::Mat getTriByBeta(cv::Mat beta, std::vector<int> inds);
	cv::Mat getTriByBetaParts(cv::Mat beta, std::vector<int> inds);
	void estimatePose3D0(cv::Mat landModel, cv::Mat landImage, cv::Mat k_m, cv::Mat &r, cv::Mat &t);
	void estimatePose3D(cv::Mat landModel, cv::Mat landImage, cv::Mat k_m, cv::Mat &r, cv::Mat &t);
	int* getLMIndices(int &count);

	//void estimatePose(cv::Mat landModel, cv::Mat lm, cv::Mat &A, cv::Mat &S, cv::Mat &R, cv::Mat &t);
	//void estimateT3D(cv::Mat landModel, cv::Mat landImage, int w, int h, cv::Mat R, float f, cv::Mat &t3D);
	//void fixSign(cv::Mat &R, cv::Mat &t);
	//cv::Mat estimateShape(cv::Mat landModel, cv::Mat landImage, cv::Mat A, cv::Mat t, int M);
	cv::Mat estimateShape3D(cv::Mat landModel, cv::Mat lm, cv::Mat k_m, cv::Mat r, cv::Mat t);

	~BaselFaceEstimator(void);
};


/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#pragma once
#include "cv.h"
#include "highgui.h"
#include "BaselFaceEstimator.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>

#define NUM_EXTRA_FEATURES 1
#define FEATURES_LANDMARK	  0
#define REG_FROM_PREV	  100
#define REG_FROM_CURR	  0

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> SpT;
typedef std::vector<std::pair<int, double> > IndWeight;

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

typedef struct BFMParams {
	float sI;
	float sF[NUM_EXTRA_FEATURES];
	float sR[RENDER_PARAMS_COUNT];
	float initR[RENDER_PARAMS_COUNT];
	bool  doOptimize[RENDER_PARAMS_COUNT];
	bool  optimizeAB[2];
	bool  optimizeExpr;
	bool  computeEI;
	float sExpr;
	cv::Mat hessDiag;  
	cv::Mat gradVec;  
	
	void init(){
		memset(doOptimize,false,sizeof(bool)*RENDER_PARAMS_COUNT);
		optimizeAB[0] = false;
		optimizeAB[1] = false;
		optimizeExpr = true;
		computeEI = false;
		sExpr = 1;
		for (int i=0;i<6;i++) doOptimize[i] = true;

		if (RENDER_PARAMS_AMBIENT < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_AMBIENT+i] = RENDER_PARAMS_AMBIENT_DEFAULT;
		}
		if (RENDER_PARAMS_DIFFUSE < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_DIFFUSE+i] = 0.0f;
		}
		if (RENDER_PARAMS_LDIR < RENDER_PARAMS_COUNT){
			for (int i=0;i<2;i++) initR[RENDER_PARAMS_LDIR+i] = 0.0f;
		}
		if (RENDER_PARAMS_CONTRAST < RENDER_PARAMS_COUNT){
			initR[RENDER_PARAMS_CONTRAST] = RENDER_PARAMS_CONTRAST_DEFAULT;
		}
		if (RENDER_PARAMS_GAIN < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_GAIN+i] = RENDER_PARAMS_GAIN_DEFAULT;
		}
		if (RENDER_PARAMS_OFFSET < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_OFFSET+i] = RENDER_PARAMS_OFFSET_DEFAULT;
		}
		if (RENDER_PARAMS_SPECULAR < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_SPECULAR+i] = RENDER_PARAMS_SPECULAR_DEFAULT;
		}
		if (RENDER_PARAMS_SHINENESS < RENDER_PARAMS_COUNT){
			initR[RENDER_PARAMS_SHINENESS] = RENDER_PARAMS_SHINENESS_DEFAULT;
		}
		
		// sR
		for (int i=0;i<3;i++) sR[RENDER_PARAMS_R+i] = (M_PI/6)*(M_PI/6);
		for (int i=0;i<3;i++) sR[RENDER_PARAMS_T+i] = 900.0f;
		if (RENDER_PARAMS_AMBIENT < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_AMBIENT+i] = 1;
		}
		if (RENDER_PARAMS_DIFFUSE < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_DIFFUSE+i] = 1;
		}
		if (RENDER_PARAMS_LDIR < RENDER_PARAMS_COUNT){
			for (int i=0;i<2;i++) sR[RENDER_PARAMS_LDIR+i] = M_PI*M_PI;
		}
		if (RENDER_PARAMS_CONTRAST < RENDER_PARAMS_COUNT){
			sR[RENDER_PARAMS_CONTRAST] = 1;
		}
		if (RENDER_PARAMS_GAIN < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_GAIN+i] = 4.0f;
		}
		if (RENDER_PARAMS_OFFSET < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_OFFSET+i] = 10000.0f;
		}
		if (RENDER_PARAMS_SPECULAR < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) sR[RENDER_PARAMS_SPECULAR+i] = 10000.0f;
		}
		if (RENDER_PARAMS_SHINENESS < RENDER_PARAMS_COUNT){
			sR[RENDER_PARAMS_SHINENESS] = 1000000.0f;
		}
	}
} BFMParams;


class FaceServices2
{
	float _k[9];
	cv::Mat faces, shape, tex;
	BaselFaceEstimator festimator;
	
	float prevEF;
	float cEF;
	float mstep;
	int countFail;
	float maxVal;
	float mlambda;
	float PREV_USE_THRESH;

public:
	FaceServices2(void);
	void setUp(int w, int h, float f);
    void init(int w, int h, float f = 1000.0f); // Yuval
	float updateHessianMatrix(bool part, cv::Mat alpha, float* renderParams, cv::Mat faces, cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &prevR, cv::Mat &prevT, cv::Mat exprW = cv::Mat() );
	cv::Mat computeGradient(bool part, cv::Mat alpha, float* renderParams, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params,std::vector<int> &inds, cv::Mat exprW, cv::Mat &prevR, cv::Mat &prevT);
	void sno_step2(bool part, cv::Mat &alpha, float* renderParams, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW, cv::Mat &prevR, cv::Mat &prevT);
	float line_search(bool part, cv::Mat &alpha, float* renderParams, cv::Mat &dirMove,std::vector<int> inds, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW, cv::Mat &prevR, cv::Mat &prevT, int maxIters = 4);
	float computeCost(float vEF, cv::Mat &alpha, float* renderParams, BFMParams &params, cv::Mat &exprW, cv::Mat &prevR, cv::Mat &prevT );
	
	float eF(bool part, cv::Mat alpha, std::vector<int> inds, cv::Mat landIm, float* renderParams, cv::Mat exprW);
	bool loadReference(std::string refDir, std::string model_file, cv::Mat &alpha, cv::Mat &beta, float* renderParams, int &M, cv::Mat &exprW, int &EM);
	bool loadReference2(std::string refDir, std::string model_file, cv::Mat &alpha, cv::Mat &beta, int &M);
	
	void initRenderer(cv::Mat &colorIm);
	void mergeIm(cv::Mat* output,cv::Mat bg,cv::Mat depth);
	~FaceServices2(void);

	bool estimatePoseExpr(cv::Mat colorIm, cv::Mat lms, cv::Mat alpha, cv::Mat &vecR, cv::Mat &vecT, cv::Mat& K, cv::Mat &exprWeightse, const char* outputDir, bool with_expr = true);
	bool updatePoseExpr(cv::Mat colorIm, cv::Mat lms, cv::Mat alpha, cv::Mat &vecR, cv::Mat &vecT, cv::Mat &exprWeightse, char* outputDir, cv::Mat &prevR, cv::Mat &prevT);

	void nextMotion(int &currFrame, cv::Mat &vecR, cv::Mat &vecT, cv::Mat &exprWeights);
};


/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "FaceServices2.h"
#include <fstream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/SparseLU>
//#include <Eigen/SPQRSupport>
#include <omp.h>

using namespace std;
using namespace cv;

FaceServices2::FaceServices2(void)
{
	//omp_set_num_threads(8);
	prevEF = 1000;
	mstep = 0.0001;
	countFail = 0;
	maxVal = 4;
	mlambda = 0.005;
	PREV_USE_THRESH = 3.141592/9;
}

void FaceServices2::setUp(int w, int h, float f){
	memset(_k,0,9*sizeof(float));
	_k[8] = 1;
	_k[0] = -f;
	_k[4] = f;
	_k[2] = w/2.0f;
	_k[5] = h/2.0f;
	faces = festimator.getFaces() - 1;
	cv::Mat shape = festimator.getShape(cv::Mat(99,1,CV_32F));
	tex = shape*0 + 128;
}

// Yuval
void FaceServices2::init(int w, int h, float f)
{
    // Initialize camera matrix
    memset(_k, 0, 9 * sizeof(float));
    _k[8] = 1;
    _k[0] = -f;
//	_k[0] = f;
    _k[4] = f;
    _k[2] = w / 2.0f;
    _k[5] = h / 2.0f;

    // Initialize shape and texture
    if (faces.empty())
        faces = festimator.getFaces() - 1;
    cv::Mat shape = festimator.getShape(cv::Mat(99, 1, CV_32F));
    tex = shape * 0 + 128;
}

float FaceServices2::updateHessianMatrix(bool part, cv::Mat alpha, float* renderParams, cv::Mat faces, cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &prevR, cv::Mat &prevT, cv::Mat exprW ){
	int M = alpha.rows;
	int EM = exprW.rows;
	float step;
	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	params.hessDiag.release();
	params.hessDiag = cv::Mat::zeros(2*M+EM+RENDER_PARAMS_COUNT,1,CV_32F);
	//printf("sno_step --------\n"); 
	//std::cout << "alpha " << alpha.t() <<std::endl;
	//std::cout << "beta " << beta.t() <<std::endl;
	//printf("renderParams ");
	//for (int i=0;i<RENDER_PARAMS_COUNT;i++)
	//	printf("(%d) %f, ",i,renderParams[i]);
	//printf("\n");
	cv::Mat alpha2, expr2;
	float renderParams2[RENDER_PARAMS_COUNT];

	cv::Mat rVec(3,1,CV_32F,renderParams+RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F,renderParams+RENDER_PARAMS_T);
	cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	float currEF = eF(part, alpha, lmInds, landIm, renderParams, exprW);
	//printf("currF: %f\n",currEF);
	cEF = currEF;

	// expr
	step = mstep*5;
	if (params.optimizeExpr) {
		for (int i=0;i<EM; i++){
			expr2.release(); expr2 = exprW.clone();
			expr2.at<float>(i,0) += step;
			float tmpEF1 = eF(part, alpha, lmInds, landIm, renderParams,expr2);
			expr2.at<float>(i,0) -= 2*step;
			float tmpEF2 = eF(part, alpha, lmInds, landIm, renderParams,expr2);
			params.hessDiag.at<float>(2*M+i,0) = params.sF[FEATURES_LANDMARK] * (tmpEF1 - 2*currEF + tmpEF2)/(step*step) 
				+ params.sExpr * 2/(0.25f*29) ;
		}
	}
	// r
	//step = 0.05;
	step = mstep*2;
	//step = 0.02;
	//step = 0.01;
	if (params.doOptimize[RENDER_PARAMS_R]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_R+i] += step;
			float tmpEF1 = eF(part, alpha, lmInds, landIm, renderParams2,exprW);

			renderParams2[RENDER_PARAMS_R+i] -= 2*step;
			float tmpEF2 = eF(part, alpha, lmInds, landIm, renderParams2,exprW);
			params.hessDiag.at<float>(2*M+EM+i,0) = params.sF[FEATURES_LANDMARK] * (tmpEF1 - 2*currEF + tmpEF2)/(step*step) + 2.0f/params.sR[RENDER_PARAMS_R+i];

		}
	}
	// t
	step = mstep*10;
	//step = 0.05;
	//step = 0.1;
	if (params.doOptimize[RENDER_PARAMS_T]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_T+i] += step;
			float tmpEF1 = eF(part, alpha, lmInds, landIm, renderParams2,exprW);
			renderParams2[RENDER_PARAMS_T+i] -= 2*step;
			float tmpEF2 = eF(part, alpha, lmInds, landIm, renderParams2,exprW);
			params.hessDiag.at<float>(2*M+EM+RENDER_PARAMS_T+i,0) = params.sF[FEATURES_LANDMARK] * (tmpEF1 - 2*currEF + tmpEF2)/(step*step) 
				+ 2.0f/params.sR[RENDER_PARAMS_T+i];
		}
	}
	return 0;
}

FaceServices2::~FaceServices2(void)
{
}

cv::Mat FaceServices2::computeGradient(bool part, cv::Mat alpha, float* renderParams, cv::Mat faces,cv::Mat colorIm, std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, std::vector<int> &inds, cv::Mat exprW, cv::Mat &prevR, cv::Mat &prevT){
	int M = alpha.rows;
	int EM = exprW.rows;
	int nTri = 40;
	float step;
	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	cv::Mat out(2*M+EM+RENDER_PARAMS_COUNT,1,CV_32F);

	cv::Mat alpha2, expr2;
	float renderParams2[RENDER_PARAMS_COUNT];
	cv::Mat rVec(3,1,CV_32F,renderParams+RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F,renderParams+RENDER_PARAMS_T);
	cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	float currEF = eF(part, alpha, lmInds, landIm, renderParams,exprW);
	cEF = currEF;
	
	#pragma omp parallel for
	for (int target=0;target<EM+6; target++){
	  if (target < EM) {
		// expr
		float step = mstep*5;
		if (params.optimizeExpr) {
				int i = target;
				std::vector<cv::Point2f> pPoints;
				cv::Mat expr2 = exprW.clone();
				expr2.at<float>(i,0) += step;
				float tmpEF = eF(part, alpha, lmInds, landIm, renderParams,expr2);
				out.at<float>(2*M+i,0) = params.sF[FEATURES_LANDMARK] * (tmpEF - currEF)/step
					+ params.sExpr * 2*exprW.at<float>(i,0)/(0.25f*29);
				//if (cEF > prevEF) printf("%f, ",tmpEF);
		}
	   }
	   else if (target < EM+3) {
		// r
		//step = 0.05;
		//step = 0.01;
		float step = mstep*2;
		//step = 0.01;
		if (params.doOptimize[RENDER_PARAMS_R]) {
			int i = target-EM;
				float renderParams2[RENDER_PARAMS_COUNT];
				memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
				renderParams2[RENDER_PARAMS_R+i] += step;
				float tmpEF = eF(part, alpha, lmInds, landIm, renderParams2,exprW);
				out.at<float>(2*M+EM+i,0) = params.sF[FEATURES_LANDMARK] * (tmpEF - currEF)/step;
				if (prevR.rows == 0)
					out.at<float>(2*M+EM+i,0) += 2*(renderParams[RENDER_PARAMS_R+i] - params.initR[RENDER_PARAMS_R+i])/params.sR[RENDER_PARAMS_R+i];
				else {
					float val = renderParams[RENDER_PARAMS_R+i] - prevR.at<float>(i,0);
					if (val > PREV_USE_THRESH)
						out.at<float>(2*M+EM+i,0) += 2*(renderParams[RENDER_PARAMS_R+i] - params.initR[RENDER_PARAMS_R+i])/params.sR[RENDER_PARAMS_R+i];
					else {
						//printf("usePrev %f (%d %d)\n",prevR.at<float>(i,0),REG_FROM_CURR,REG_FROM_PREV);
						out.at<float>(2*M+EM+i,0) += 2*REG_FROM_CURR*(renderParams[RENDER_PARAMS_R+i] - params.initR[RENDER_PARAMS_R+i])/params.sR[RENDER_PARAMS_R+i] + 2*REG_FROM_PREV*val/params.sR[RENDER_PARAMS_R+i];
					}
				}
				//if (cEF > prevEF) printf("%f, ",tmpEF);
		}
	  }
	  else {
		// t
		float step = mstep*10;
		//step = 1;
		if (params.doOptimize[RENDER_PARAMS_T]) {
			int i = target-EM-3;
				float renderParams2[RENDER_PARAMS_COUNT];
				memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
				renderParams2[RENDER_PARAMS_T+i] += step;
				float tmpEF = eF(part, alpha, lmInds, landIm, renderParams2,exprW);
				out.at<float>(2*M+EM+RENDER_PARAMS_T+i,0) = params.sF[FEATURES_LANDMARK] * (tmpEF - currEF)/step 
					+ 2*(renderParams[RENDER_PARAMS_T+i] - params.initR[RENDER_PARAMS_T+i])/params.sR[RENDER_PARAMS_T+i];
				//if (cEF > prevEF) printf("%f, ",tmpEF);
		}
	  }
	}
	return out;
}

float FaceServices2::eF(bool part, cv::Mat alpha, std::vector<int> inds, cv::Mat landIm, float* renderParams, cv::Mat exprW){
	Mat k_m(3,3,CV_32F,_k);
	//printf("%f\n",renderParams[RENDER_PARAMS_R+1]);
	cv::Mat mLM;
	if (!part)
		mLM = festimator.getLMByAlpha(alpha,-renderParams[RENDER_PARAMS_R+1], inds, exprW);
	else
		mLM = festimator.getLMByAlphaParts(alpha,-renderParams[RENDER_PARAMS_R+1], inds, exprW);
	//write_plyFloat("vismLM.ply",mLM.t());
	//printf("inds\n");
	//for (int i=0;i<inds.size(); i++)
//		printf("%d\n", inds[i]);
	//printf("renderParams %f %f %f   %f %f %f\n", renderParams[0],renderParams[1],renderParams[2],renderParams[3],renderParams[4],renderParams[5]);
	//std::cout << "landIm " << landIm << std::endl;
	//cv::Mat ads = cv::Mat::zeros(720,540,CV_8UC3);
	
	cv::Mat rVec(3,1,CV_32F, renderParams + RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F, renderParams + RENDER_PARAMS_T);
	std::vector<cv::Point2f> allImgPts;
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );

	cv::projectPoints( mLM, rVec, tVec, k_m, distCoef, allImgPts );
	float err = 0;
	for (int i=0;i<mLM.rows;i++){
		float val = landIm.at<float>(i,0) - allImgPts[i].x;
		err += val*val;
		val = landIm.at<float>(i,1) - allImgPts[i].y;
		err += val*val;
		//cv::circle(ads,cv::Point(landIm.at<float>(i,0),landIm.at<float>(i,1)),2,cv::Scalar(255,0,0));
		//cv::circle(ads,cv::Point(allImgPts[i].x,allImgPts[i].y),2,cv::Scalar(0,0,255));
	}
	//imwrite("sas.png",ads);
	//getchar();
	return sqrt(err/mLM.rows);
}

void FaceServices2::sno_step2(bool part, cv::Mat &alpha, float* renderParams, cv::Mat faces,cv::Mat colorIm, std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW, cv::Mat &prevR, cv::Mat &prevT){
	//printf("sno_step --------\n"); 
	//std::cout << "alpha " << alpha.t() <<std::endl;
	//std::cout << "beta " << beta.t() <<std::endl;
	//printf("renderParams ");
	//for (int i=0;i<RENDER_PARAMS_COUNT;i++)
	//	printf("%f, ",renderParams[i]);
	//printf("\n");
	float lambda = 0.05;
	std::vector<int> inds;
	cv::Mat dE = computeGradient(part, alpha, renderParams, faces, colorIm, lmInds, landIm, params,inds, exprW, prevR, prevT);
	params.gradVec.release(); params.gradVec = dE.clone();
	cv::Mat dirMove = dE*0;

	int M = alpha.rows;
	int EM = exprW.rows;
	if (params.optimizeExpr){
		for (int i=0;i<EM;i++)
			if (abs(params.hessDiag.at<float>(2*M+i,0)) > 0.0000001) {
				dirMove.at<float>(2*M+i,0) = - lambda*dE.at<float>(2*M+i,0)/abs(params.hessDiag.at<float>(2*M+i,0));
			}
	}

	for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
		if (params.doOptimize[i]){
			if (abs(params.hessDiag.at<float>(2*M+EM+i,0)) > 0.0000001) {
				dirMove.at<float>(2*M+EM+i,0) = - lambda*dE.at<float>(2*M+EM+i,0)/abs(params.hessDiag.at<float>(2*M+EM+i,0));
			}
		}
	}
	//float pc = 1;
	float pc = line_search(part, alpha, renderParams, dirMove,inds, faces, colorIm, lmInds, landIm, params, exprW, prevR, prevT, 10);
	//printf("pc = %f\n",pc);
	if (pc == 0) countFail++;
	else {
		if (params.optimizeExpr){
			for (int i=0;i<EM;i++) {
				exprW.at<float>(i,0) += pc*dirMove.at<float>(i+2*M,0);
				if (exprW.at<float>(i,0) > 3) exprW.at<float>(i,0) = 3;
				else if (exprW.at<float>(i,0) < -3) exprW.at<float>(i,0) = -3;
			}
		}

		for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
			if (params.doOptimize[i]){
				renderParams[i] += pc*dirMove.at<float>(2*M+EM+i,0);
				if (i == RENDER_PARAMS_CONTRAST || (i>=RENDER_PARAMS_AMBIENT && i<RENDER_PARAMS_DIFFUSE+3) ) {
					if (renderParams[i] > 1.0) renderParams[i] = 1.0;
					if (renderParams[i] < 0) renderParams[i] = 0;

				}
				else if (i >= RENDER_PARAMS_GAIN && i<=RENDER_PARAMS_GAIN+3) {
					if (renderParams[i] > 3.0) renderParams[i]  = 3;
					if (renderParams[i] < 0.3) renderParams[i] = 0.3;
				}
			}
		}
	}
	prevEF = cEF;
}

float FaceServices2::line_search(bool part, cv::Mat &alpha, float* renderParams, cv::Mat &dirMove,std::vector<int> inds, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW, cv::Mat &prevR, cv::Mat &prevT, int maxIters){
	float step = 1.0f;
	float sstep = 2.0f;
	float minStep = 0.0001f;
	//float cCost = computeCost(cEF, cEI,cETE, cECE, alpha, beta, renderParams,params);
	cv::Mat alpha2, exprW2;
	float renderParams2[RENDER_PARAMS_COUNT];
	alpha2 = alpha.clone();
	exprW2 = exprW.clone();
	memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);

	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	std::vector<cv::Point2f> pPoints;
	cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	int M = alpha.rows;
	int EM = exprW.rows;
	float ssize = 0;
	for (int i=0;i<dirMove.rows;i++) ssize += dirMove.at<float>(i,0)*dirMove.at<float>(i,0);
	ssize = sqrt(ssize);
	//printf("ssize: %f\n",ssize);
	if (ssize > (2*M+EM+RENDER_PARAMS_COUNT)/5.0f) {
		step = (2*M+EM+RENDER_PARAMS_COUNT)/(5.0f * ssize);
		ssize = (2*M+EM+RENDER_PARAMS_COUNT)/5.0f;
	}
	if (ssize < minStep){
		return 0;
	}
	int tstep = floor(log(ssize/minStep));
	if (tstep < maxIters) maxIters = tstep;

	float curCost = computeCost(cEF, alpha, renderParams, params, exprW, prevR, prevT );

	bool hasNoBound = false;
	int iter = 0;
	for (; iter<maxIters; iter++){
		if (params.optimizeExpr){
			for (int i=0;i<EM;i++) {
				float tmp = exprW.at<float>(i,0) + step*dirMove.at<float>(2*M+i,0);
				if (tmp >= 3) exprW2.at<float>(i,0) = 3;
				else if (tmp <= -3) exprW2.at<float>(i,0) = -3;
				else {
					exprW2.at<float>(i,0) = tmp;
					hasNoBound = true;
				}
			}
		}

		for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
			if (params.doOptimize[i]){
				float tmp = renderParams[i] + step*dirMove.at<float>(2*M+EM+i,0);
				if (i == RENDER_PARAMS_CONTRAST || (i>=RENDER_PARAMS_AMBIENT && i<RENDER_PARAMS_DIFFUSE+3) ) {
					if (tmp > 1.0) renderParams2[i] = 1.0f;
					else if (tmp < -1.0) renderParams2[i] = -1.0f;
					else {
						renderParams2[i] = tmp;
						hasNoBound = true;
					}
				}
				else if (i >= RENDER_PARAMS_GAIN && i<=RENDER_PARAMS_GAIN+3) {
					if (tmp >= 3.0) renderParams2[i] = 3.0f;
					else if (tmp <= -3.0) renderParams2[i] = -3.0f;
					else {
						renderParams2[i] = tmp;
						hasNoBound = true;
					}
				}
				else renderParams2[i] = tmp;
			}
		}
		if (!hasNoBound) {
			iter = maxIters; break;
		}
		float tmpEF = cEF;
		if (params.sF[FEATURES_LANDMARK] > 0) tmpEF = eF(part,alpha2, lmInds,landIm,renderParams2, exprW2);
		float tmpCost = computeCost(tmpEF, alpha2, renderParams2, params,exprW2, prevR, prevT);
		//printf("tmpCost %f\n",tmpCost);
		if (tmpCost < curCost) {
			break;
		}
		else {
			step = step/sstep;
			//printf("step %f\n",step);
		}
	}
	//getchar();
	if (iter >= maxIters) return 0;
	else return step;
}

float FaceServices2::computeCost(float vEF, cv::Mat &alpha, float* renderParams, BFMParams &params, cv::Mat &exprW, cv::Mat &prevR, cv::Mat &prevT ){
	float val = params.sF[FEATURES_LANDMARK]*vEF;
	int M = alpha.rows;
	if (params.optimizeExpr){
		for (int i=0;i<exprW.rows;i++)
			val += params.sExpr * exprW.at<float>(i,0)*exprW.at<float>(i,0)/(0.5f*29);
	}

	for (int i=0;i<3;i++) {
		if (params.doOptimize[i]){
			if (prevR.rows == 0) val += (renderParams[i] - params.initR[i])*(renderParams[i] - params.initR[i])/params.sR[i];
			else {
				float vval = renderParams[i] - prevR.at<float>(i,0);
				if (vval > PREV_USE_THRESH) 
					val += (renderParams[i] - params.initR[i])*(renderParams[i] - params.initR[i])/params.sR[i];
				else 
					val += REG_FROM_CURR*(renderParams[i] - params.initR[i])*(renderParams[i] - params.initR[i])/params.sR[i] + REG_FROM_PREV*vval*vval/params.sR[i];
			}
		}
	}
	for (int i=3;i<RENDER_PARAMS_COUNT;i++) {
		if (params.doOptimize[i]){
			val += (renderParams[i] - params.initR[i])*(renderParams[i] - params.initR[i])/params.sR[i];
		}
	}
	return val;
}

bool FaceServices2::loadReference(string refDir, string model_file, cv::Mat &alpha, cv::Mat &beta, float* renderParams, int &M, cv::Mat &exprW, int &EM){
	string fname(model_file);
	size_t sep = model_file.find_last_of("\\/");
	if (sep != std::string::npos)
		fname = model_file.substr(sep + 1, model_file.size() - sep - 1);
	fname = refDir + fname;

	alpha = cv::Mat::zeros(M,1,CV_32F);
	beta = cv::Mat::zeros(M,1,CV_32F);
	char fpath[250];
	char text[250];
	sprintf(fpath,"%s.alpha",fname.c_str());
	//printf("fpath %s\n",fpath);
	FILE* f = fopen(fpath,"r");
	if (f == 0) return false;

	text[0] = '\0';
	for (int i=0;i<M;i++){
		fgets(text,250,f);
		int l = strlen(text);
		if (text[l-1] <'0' || text[l-1] > '9') text[l-1] = '\0';
		alpha.at<float>(i,0) = atof(text);
	}
	fclose(f);

	sprintf(fpath,"%s.beta",fname.c_str());
	f = fopen(fpath,"r");
	text[0] = '\0';
	for (int i=0;i<M;i++){
		fgets(text,250,f);
		int l = strlen(text);
		if (text[l-1] <'0' || text[l-1] > '9') text[l-1] = '\0';
		beta.at<float>(i,0) = atof(text);
	}
	fclose(f);

	sprintf(fpath,"%s.rend",fname.c_str());
	f = fopen(fpath,"r");
	text[0] = '\0';
	for (int i=0;i<RENDER_PARAMS_COUNT;i++){
		fgets(text,250,f);
		int l = strlen(text);
		if (text[l-1] <'0' || text[l-1] > '9') text[l-1] = '\0';
		renderParams[i] = atof(text);
	}
	fclose(f);
	
	sprintf(fpath,"%s.expr",fname.c_str());
	//printf("fpath %s\n",fpath);
	f = fopen(fpath,"r");
	if (f == 0) return false;

	text[0] = '\0';
	for (int i=0;i<EM;i++){
		fgets(text,250,f);
		int l = strlen(text);
		if (text[l-1] <'0' || text[l-1] > '9') text[l-1] = '\0';
		exprW.at<float>(i,0) = atof(text);
	}
	fclose(f);
	return true;
}

bool FaceServices2::loadReference2(string refDir, string model_file, cv::Mat &alpha, cv::Mat &beta, int &M){
	string fname(model_file);
	size_t sep = model_file.find_last_of("\\/");
	if (sep != std::string::npos)
		fname = model_file.substr(sep + 1, model_file.size() - sep - 1);
	fname = refDir + fname;

	alpha = cv::Mat::zeros(M,1,CV_32F);
	beta = cv::Mat::zeros(M,1,CV_32F);
	char fpath[250];
	char text[250];
	sprintf(fpath,"%s.alpha",fname.c_str());
	//printf("fpath %s\n",fpath);
	FILE* f = fopen(fpath,"r");
	if (f == 0) return false;

	text[0] = '\0';
	for (int i=0;i<M;i++){
		fgets(text,250,f);
		int l = strlen(text);
		if (text[l-1] <'0' || text[l-1] > '9') text[l-1] = '\0';
		alpha.at<float>(i,0) = atof(text);
	}
	fclose(f);

	sprintf(fpath,"%s.beta",fname.c_str());
	f = fopen(fpath,"r");
	text[0] = '\0';
	for (int i=0;i<M;i++){
		fgets(text,250,f);
		int l = strlen(text);
		if (text[l-1] <'0' || text[l-1] > '9') text[l-1] = '\0';
		beta.at<float>(i,0) = atof(text);
	}
	fclose(f);
	return true;
}
void FaceServices2::initRenderer(cv::Mat &colorIm){
    cv::Mat faces = festimator.getFaces() - 1;
    cv::Mat faces_fill = festimator.getFaces_fill() - 1;
    cv::Mat colors;
    cv::Mat shape = festimator.getShape(cv::Mat::zeros(1,1,CV_32F));
    cv::Mat tex = festimator.getTexture(cv::Mat::zeros(1,1,CV_32F));
}

void FaceServices2::mergeIm(cv::Mat* output,cv::Mat bg,cv::Mat depth){
	for (int i=0;i<bg.rows;i++){
		for (int j=0;j<bg.cols;j++){
			if (depth.at<float>(i,j) >= 0.9999)
				output->at<Vec3b>(i, j) = bg.at<Vec3b>(i,j);
		}
	}
}



bool FaceServices2::estimatePoseExpr(cv::Mat colorIm, cv::Mat lms, cv::Mat alpha, cv::Mat &vecR, cv::Mat &vecT, cv::Mat& K, cv::Mat &exprW, const char* outputDir, bool with_expr){
	char text[200];
	float renderParams[RENDER_PARAMS_COUNT];
	float renderParams2[RENDER_PARAMS_COUNT];
	Mat k_m(3,3,CV_32F,_k);
    K = k_m.clone();    // Yuval
	BFMParams params;
	params.init();
	exprW = cv::Mat::zeros(29,1,CV_32F);
	cv::Mat prevR;
	cv::Mat prevT;

	int M = 99;
	cv::Mat shape = festimator.getShape(alpha);

	Mat landModel0 = festimator.getLM(shape,0);
	int nLM = landModel0.rows;
	Mat landIm = cv::Mat( 60,2,CV_32F);
	Mat landModel = cv::Mat( 60,3,CV_32F);
	for (int i=0;i<60;i++){
		landModel.at<float>(i,0) = landModel0.at<float>(i,0);
		landModel.at<float>(i,1) = landModel0.at<float>(i,1);
		landModel.at<float>(i,2) = landModel0.at<float>(i,2);
		landIm.at<float>(i,0) = lms.at<float>(i,0);
		landIm.at<float>(i,1) = lms.at<float>(i,1);
	}
	festimator.estimatePose3D0(landModel,landIm,k_m,vecR,vecT);
	float yaw = -vecR.at<float>(1,0);
	landModel0 = festimator.getLM(shape,yaw);
	std::vector<int> lmVisInd;
	for (int i=0;i<60;i++){
		if ((yaw > M_PI/10 && i > 7) || (yaw < -M_PI/10 && i < 9) || i > 16 || abs(yaw) <= M_PI/10)
			lmVisInd.push_back(i);
	}
	cv::Mat tmpIm = colorIm/5;
	landModel = cv::Mat( lmVisInd.size(),3,CV_32F);
	landIm = cv::Mat::zeros( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landModel.at<float>(i,0) = landModel0.at<float>(ind,0);
		landModel.at<float>(i,1) = landModel0.at<float>(ind,1);
		landModel.at<float>(i,2) = landModel0.at<float>(ind,2);
		landIm.at<float>(i,0) = lms.at<float>(ind,0);
		landIm.at<float>(i,1) = lms.at<float>(ind,1);
		//cv::circle(tmpIm,Point(landIm.at<float>(i,0),landIm.at<float>(i,1)),2,Scalar(255,0,0),2);
	}
	//sprintf(text,"%s/withLM.png",outputDir);
	//imwrite(text,tmpIm);
	//sprintf(text,"%s/%s_withLM.ply",outputDir,filename.c_str());
	//write_plyFloat(text,landModel.t());
	//getchar();

	festimator.estimatePose3D0(landModel,landIm,k_m,vecR,vecT);

    // Yuval
    if (!with_expr) return true;
	
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_R+i] = vecR.at<float>(i,0);
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_T+i] = vecT.at<float>(i,0);
	memcpy(renderParams,params.initR,sizeof(float)*RENDER_PARAMS_COUNT);

	for (int i=60;i<68;i++) lmVisInd.push_back(i);
	landIm = cv::Mat::zeros( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landIm.at<float>(i,0) = lms.at<float>(ind,0);
		landIm.at<float>(i,1) = lms.at<float>(ind,1);
	}

	float bCost, cCost, fCost;
	int bestIter = 0;
	bCost = 10000.0f;

	memset(params.sF,0,sizeof(float)*NUM_EXTRA_FEATURES);

	params.sI = 0.0;
	params.sF[FEATURES_LANDMARK] = 8.0f;
	Mat alpha0;
	int iter=0;
	int badCount = 0;
	memset(params.doOptimize,true,sizeof(bool)*6);

	int EM = 29;
	float renderParams_tmp[RENDER_PARAMS_COUNT];

	params.optimizeAB[0] = params.optimizeAB[1] = false;
	for (;iter<60;iter++) {
			if (iter%20 == 0) {
				cCost = updateHessianMatrix(false, alpha,renderParams,faces,colorIm,lmVisInd,landIm,params, prevR, prevT, exprW);
				if (countFail > 10) {
					countFail = 0;
					break;
				}
				prevEF = cEF;
			}
			sno_step2(false, alpha, renderParams, faces,colorIm,lmVisInd,landIm,params,exprW,prevR, prevT);
		}
	iter = 60;
	memset(params.doOptimize,false,sizeof(bool)*6);countFail = 0;
    //int total_iter = 200;
    int total_iter = 400;   // Yuval
	for (;iter<total_iter;iter++) {
			if (iter%60 == 0) {
				cCost = updateHessianMatrix(false, alpha,renderParams,faces,colorIm,lmVisInd,landIm,params, prevR, prevT, exprW);
				if (countFail > 10) {
					countFail = 0;
					break;
				}
				prevEF = cEF;
			}
			sno_step2(false, alpha, renderParams, faces,colorIm,lmVisInd,landIm,params,exprW,prevR, prevT);
		}

	///for (int i=0;i<3; i++) vecR.at<float>(i,0) = renderParams[i];
	///for (int i=0;i<3; i++) vecT.at<float>(i,0) = renderParams[i+3];
	//std::cout << "vecR " << vecR << std::endl;
	//std::cout << "vecT " << vecT << std::endl;
	//std::cout << "exprW " << exprW << std::endl;

    return true;
}

bool FaceServices2::updatePoseExpr(cv::Mat colorIm, cv::Mat lms, cv::Mat alpha, cv::Mat &vecR, cv::Mat &vecT, cv::Mat &exprW, char* outputDir, cv::Mat &prevR, cv::Mat &prevT )
{
	char text[200];
	float renderParams[RENDER_PARAMS_COUNT];
	float renderParams2[RENDER_PARAMS_COUNT];
	Mat k_m(3,3,CV_32F,_k);
	BFMParams params;
	params.init();

	int M = 99;
	float yaw = -vecR.at<float>(1,0);
	Mat landModel0 = festimator.getLM(shape,yaw);
	std::vector<int> lmVisInd;
	for (int i=0;i<60;i++){
		if (i > 16 || std::abs(yaw) <= M_PI/10 || (yaw > M_PI/10 && i > 7) || (yaw < -M_PI/10 && i < 9))
			lmVisInd.push_back(i);
	}
	cv::Mat landModel = cv::Mat( lmVisInd.size(),3,CV_32F);
	cv::Mat landIm = cv::Mat::zeros( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landModel.at<float>(i,0) = landModel0.at<float>(ind,0);
		landModel.at<float>(i,1) = landModel0.at<float>(ind,1);
		landModel.at<float>(i,2) = landModel0.at<float>(ind,2);
		landIm.at<float>(i,0) = lms.at<float>(ind,0);
		landIm.at<float>(i,1) = lms.at<float>(ind,1);
		//cv::circle(tmpIm,Point(landIm.at<float>(i,0),landIm.at<float>(i,1)),2,Scalar(255,0,0),2);
	}
	//sprintf(text,"%s/withLM.png",outputDir);
	//imwrite(text,tmpIm);
	//sprintf(text,"%s/%s_withLM.ply",outputDir,filename.c_str());
	//write_plyFloat(text,landModel.t());
	//getchar();

	festimator.estimatePose3D0(landModel,landIm,k_m,vecR,vecT);
	for (int i=60;i<68;i++) lmVisInd.push_back(i);

	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_R+i] = vecR.at<float>(i,0);
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_T+i] = vecT.at<float>(i,0);
	memcpy(renderParams,params.initR,sizeof(float)*RENDER_PARAMS_COUNT);

	landIm = cv::Mat::zeros( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landIm.at<float>(i,0) = lms.at<float>(ind,0);
		landIm.at<float>(i,1) = lms.at<float>(ind,1);
	}

	float bCost, cCost, fCost;
	int bestIter = 0;
	bCost = 10000.0f;

	memset(params.sF,0,sizeof(float)*NUM_EXTRA_FEATURES);

	params.sI = 0.0;
	params.sF[FEATURES_LANDMARK] = 8.0f;
	int iter=0;
	int badCount = 0;
	memset(params.doOptimize,true,sizeof(bool)*6);

	int EM = 29;
	float renderParams_tmp[RENDER_PARAMS_COUNT];

	params.optimizeAB[0] = params.optimizeAB[1] = false;

    for (; iter < 30; iter++)
    {
        if (iter % 10 == 0) {
            cCost = updateHessianMatrix(false, alpha, renderParams, faces, colorIm, lmVisInd, landIm, params, prevR, prevT, exprW);
            if (countFail > 10) {
                countFail = 0;

                break;
            }
            prevEF = cEF;
        }
        sno_step2(false, alpha, renderParams, faces, colorIm, lmVisInd, landIm, params, exprW, prevR, prevT);

    }

    return true;
}

void FaceServices2::nextMotion(int &currFrame, cv::Mat &vecR, cv::Mat &vecT, cv::Mat &exprWeights){
    float stepYaw = 1;
    float PPI = 3.141592;
    int maxYaw = 70/stepYaw;
    float stepPitch = 1;
    int maxPitch = 45/stepPitch;

    int totalFrames = 4 * (maxYaw + maxPitch);
    currFrame = (currFrame + 1) % totalFrames;
    vecR = vecR*0 + 0.00001;
    // Rotate left
    if (currFrame < maxYaw) vecR.at<float>(1,0) = -currFrame*stepYaw * PPI/180;
    else if (currFrame < 2*maxYaw) vecR.at<float>(1,0) = -stepYaw * (2*maxYaw - currFrame) * PPI/180;
    // Rotate right
    else if (currFrame < 3*maxYaw) vecR.at<float>(1,0) = (currFrame-2*maxYaw) * stepYaw * PPI/180;
    else if (currFrame < 4*maxYaw) vecR.at<float>(1,0) = stepYaw * (4*maxYaw - currFrame) * PPI/180;
    
    // Rotate up
    else if (currFrame < 4*maxYaw + maxPitch) vecR.at<float>(0,0) = -(currFrame-4*maxYaw)*stepPitch * PPI/180;
    else if (currFrame < 4*maxYaw + 2*maxPitch) vecR.at<float>(0,0) = -stepPitch * (4*maxYaw+2*maxPitch - currFrame) * PPI/180;
    // Rotate right
    else if (currFrame < 4*maxYaw + 3*maxPitch) vecR.at<float>(0,0) = (currFrame-4*maxYaw-2*maxPitch) * stepPitch * PPI/180;
    else if (currFrame < 4*maxYaw + 4*maxPitch) vecR.at<float>(0,0) = stepPitch * (4*maxPitch+4*maxYaw - currFrame) * PPI/180;
    //std::cout << "vecR " << vecR.t() << std::endl;
}

/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "BaselFaceEstimator.h"
#include <string.h>
#include "utility.h"
#include "epnp.h"
#include <vector>
#include <opencv2/calib3d.hpp>

using namespace cv;

BaselFaceEstimator::BaselFaceEstimator(/*string baselFile*/)
{
	//if (baselFile.length() > 0)
	//	BaselFace::load_BaselFace_data(baselFile.c_str());
}

cv::Mat BaselFaceEstimator::coef2object(cv::Mat weight, cv::Mat MU, cv::Mat PCs, cv::Mat EV){
	int M = weight.rows;
	Mat tmpShape;
	if (M == 0) 
		tmpShape = MU.clone();
	else {
		Mat subPC = PCs(Rect(0,0,M,PCs.rows));
		Mat subEV = EV(Rect(0,0,1,M));
		tmpShape = MU + subPC * weight.mul(subEV);
	}
	return tmpShape.reshape(1,tmpShape.rows/3);
}

cv::Mat BaselFaceEstimator::coef2objectParts(cv::Mat &weight, cv::Mat &MU, cv::Mat &PCs, cv::Mat &EV){
	//printf("weight %d %d\n",weight.rows,weight.cols);
	int numparts = BaselFace::BaselFace_wparts_w;
	int M = weight.rows/numparts;
	Mat tmpShape = MU.clone();
	if (weight.rows != 0) {
	Mat wparts(BaselFace::BaselFace_wparts_h,numparts,CV_32F,BaselFace::BaselFace_wparts);

	for (int i=0;i<BaselFace::BaselFace_wparts_h;i++){
		for (int j=0;j<3;j++){
			for (int k=0;k<M;k++){
				for (int p=0;p<numparts;p++){
					tmpShape.at<float>(3*i+j) += weight.at<float>(p*M+k) * PCs.at<float>(3*i+j,k) * EV.at<float>(k,0) * BaselFace::BaselFace_wparts[i*numparts+p];
				}
			}
		}
	}
	}
	return tmpShape.reshape(1,tmpShape.rows/3);
}

BaselFaceEstimator::~BaselFaceEstimator(void)
{
}

//cv::Mat BaselFaceEstimator::getFaces(){
//	int* arr = new int[BaselFace::BaselFace_faces_h*BaselFace::BaselFace_faces_w];
//	memcpy(arr,BaselFace::BaselFace_faces,BaselFace::BaselFace_faces_h*BaselFace::BaselFace_faces_w*sizeof(int));
//	cv::Mat out(BaselFace::BaselFace_faces_h,BaselFace::BaselFace_faces_w,CV_32S,arr);
//	return out;
//}

// Yuval
cv::Mat BaselFaceEstimator::getFaces()
{
	cv::Mat out(BaselFace::BaselFace_faces_h, BaselFace::BaselFace_faces_w, CV_32S, BaselFace::BaselFace_faces);
	return out.clone();
}

cv::Mat BaselFaceEstimator::getFaces_fill(){
	int h1 = BaselFace::BaselFace_faces_h;
	int h2 = BaselFace::BaselFace_faces_extra_h;
	int* arr = new int[(h1+h2)*BaselFace::BaselFace_faces_w];
	memcpy(arr,BaselFace::BaselFace_faces,h1*BaselFace::BaselFace_faces_w*sizeof(int));
	memcpy(arr + h1*BaselFace::BaselFace_faces_w,BaselFace::BaselFace_faces_extra,h2*BaselFace::BaselFace_faces_w*sizeof(int));
	cv::Mat out(h1+h2,BaselFace::BaselFace_faces_w,CV_32S,arr);
	return out;
}

int* BaselFaceEstimator::getLMIndices(int &count){
	count = BaselFace::BaselFace_lmInd_h*BaselFace::BaselFace_lmInd_w;
	int* out = new int[BaselFace::BaselFace_lmInd_h];
	for (int i=0;i<BaselFace::BaselFace_lmInd_h;i++)
		out[i] = BaselFace::BaselFace_lmInd[i] - 1;
	return out;
}

cv::Mat BaselFaceEstimator::getShape(cv::Mat weight, cv::Mat exprWeight){
	Mat shapeMU(BaselFace::BaselFace_shapeMU_h,1,CV_32F,BaselFace::BaselFace_shapeMU);
	Mat shapePC(BaselFace::BaselFace_shapePC_h,BaselFace::BaselFace_shapePC_w,CV_32F,BaselFace::BaselFace_shapePC);
	Mat shapeEV(BaselFace::BaselFace_shapeEV_h,1,CV_32F,BaselFace::BaselFace_shapeEV);
	Mat exprMU(BaselFace::BaselFace_expMU_h,1,CV_32F,BaselFace::BaselFace_expMU);
	Mat exprPC(BaselFace::BaselFace_expPC_h,BaselFace::BaselFace_expPC_w,CV_32F,BaselFace::BaselFace_expPC);
	Mat exprEV(BaselFace::BaselFace_expEV_h,1,CV_32F,BaselFace::BaselFace_expEV);
	return coef2object(weight,shapeMU,shapePC,shapeEV) + coef2object(exprWeight,exprMU,exprPC,exprEV);
}

cv::Mat BaselFaceEstimator::getShapeParts(cv::Mat weight, cv::Mat exprWeight){
	Mat shapeMU(BaselFace::BaselFace_shapeMU_h,1,CV_32F,BaselFace::BaselFace_shapeMU);
	Mat shapePC(BaselFace::BaselFace_shapePC_h,BaselFace::BaselFace_shapePC_w,CV_32F,BaselFace::BaselFace_shapePC);
	Mat shapeEV(BaselFace::BaselFace_shapeEV_h,1,CV_32F,BaselFace::BaselFace_shapeEV);
	Mat exprMU(BaselFace::BaselFace_expMU_h,1,CV_32F,BaselFace::BaselFace_expMU);
	Mat exprPC(BaselFace::BaselFace_expPC_h,BaselFace::BaselFace_expPC_w,CV_32F,BaselFace::BaselFace_expPC);
	Mat exprEV(BaselFace::BaselFace_expEV_h,1,CV_32F,BaselFace::BaselFace_expEV);
	return coef2objectParts(weight,shapeMU,shapePC,shapeEV) + coef2object(exprWeight,exprMU,exprPC,exprEV);
}

cv::Mat BaselFaceEstimator::getTextureParts(cv::Mat weight){
	Mat texMU(BaselFace::BaselFace_texMU_h,1,CV_32F,BaselFace::BaselFace_texMU);
	Mat texPC(BaselFace::BaselFace_texPC_h,BaselFace::BaselFace_texPC_w,CV_32F,BaselFace::BaselFace_texPC);
	Mat texEV(BaselFace::BaselFace_texEV_h,1,CV_32F,BaselFace::BaselFace_texEV);
	return coef2objectParts(weight,texMU,texPC,texEV);
}

cv::Mat BaselFaceEstimator::getTexture(cv::Mat weight){
	Mat texMU(BaselFace::BaselFace_texMU_h,1,CV_32F,BaselFace::BaselFace_texMU);
	Mat texPC(BaselFace::BaselFace_texPC_h,BaselFace::BaselFace_texPC_w,CV_32F,BaselFace::BaselFace_texPC);
	Mat texEV(BaselFace::BaselFace_texEV_h,1,CV_32F,BaselFace::BaselFace_texEV);
	return coef2object(weight,texMU,texPC,texEV);
}

cv::Mat BaselFaceEstimator::getLM(cv::Mat shape, float yaw){
	cv::Mat lm(BaselFace::BaselFace_lmInd_h,3,CV_32F);
	for (int i=0;i<BaselFace::BaselFace_lmInd_h;i++){
		int ind;
		if (yaw > 0 && yaw < M_PI/9 && i < 8) ind = BaselFace::BaselFace_lmInd[i]-1;
		else if (yaw < 0 && yaw > -M_PI/9 && i > 8 && i < 17) ind = BaselFace::BaselFace_lmInd[i]-1;
		else
		ind = BaselFace::BaselFace_lmInd2[i]-1;
/*
		if (abs(yaw) < M_PI/20 || i > 14) ind = BaselFace::BaselFace_lmInd[i]-1;
		else {
			if (i < 7) {
				if (yaw > -M_PI/10) ind = BaselFace::BaselFace_lmInd[i]-1;
				else  ind = BaselFace::BaselFace_lmInd2[i]-1;
			}
			else {
				if (yaw < M_PI/10) ind = BaselFace::BaselFace_lmInd[i]-1;
				else  ind = BaselFace::BaselFace_lmInd2[i]-1;
			}
		}
*/
		for (int j=0;j<3;j++){
			lm.at<float>(i,j) = shape.at<float>(ind,j);
		}
	}
	return lm;
}

void BaselFaceEstimator::estimatePose3D0(cv::Mat landModel, cv::Mat landImage, cv::Mat k_m, cv::Mat &r, cv::Mat &t){
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );

	double R_est[3][3], t_est[3];
	cv::Mat rVec( 3, 1, CV_64F );
	cv::Mat tVec( 3, 1, CV_64F );	
	//std::cout << "k_m " << k_m << std::endl;

	epnp PnP;
	PnP.set_internal_parameters(k_m.at<float>(0,2),k_m.at<float>(1,2),k_m.at<float>(0,0),k_m.at<float>(1,1));
	PnP.set_maximum_number_of_correspondences(landImage.rows);

	std::vector<cv::Point3f> allObjPts;
	std::vector<cv::Point2f> allObj2DPts;
	std::vector<cv::Point2f> allImgPts;
	//printf("prepare allObjPts allObj2DPts\n");
	for ( int i=0; i<landModel.rows; ++i )
	{
		allObjPts.push_back(Point3f(landModel.at<float>(i,0),landModel.at<float>(i,1),landModel.at<float>(i,2)));
		allObj2DPts.push_back(Point2f(landImage.at<float>(i,0),landImage.at<float>(i,1)));
	}
	PnP.reset_correspondences();
	for(int ind = 0; ind < landImage.rows; ind++){
		PnP.add_correspondence(allObjPts[ind].x,allObjPts[ind].y,allObjPts[ind].z, allObj2DPts[ind].x,allObj2DPts[ind].y);
	}
	double err2 = PnP.compute_pose(R_est, t_est);
	cv::Mat rMatP( 3, 3, CV_64F, R_est);
	cv::Rodrigues(rMatP, rVec);
	rVec.convertTo(rVec,CV_32F);
	cv::Mat tVecP( 3, 1, CV_64F, t_est);
	tVec = tVecP.clone();
	tVec.convertTo(tVec,CV_32F);

	r = rVec.clone();
	t = tVec.clone();
}

void BaselFaceEstimator::estimatePose3D(cv::Mat landModel, cv::Mat landImage, cv::Mat k_m, cv::Mat &r, cv::Mat &t){
	int bestCons = -1;
	int sample = 20;
	cv::RNG rng(cv::getTickCount());
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );

	double R_est[3][3], t_est[3];
	cv::Mat rVec( 3, 1, CV_64F );
	cv::Mat tVec( 3, 1, CV_64F );	
	//std::cout << "k_m " << k_m << std::endl;

	epnp PnP;
	PnP.set_internal_parameters(k_m.at<float>(0,2),k_m.at<float>(1,2),k_m.at<float>(0,0),k_m.at<float>(1,1));
	PnP.set_maximum_number_of_correspondences(sample);

	std::vector<cv::Point3f> allObjPts;
	std::vector<cv::Point2f> allObj2DPts;
	std::vector<cv::Point2f> allImgPts;
	//printf("prepare allObjPts allObj2DPts\n");
	for ( int i=0; i<landModel.rows; ++i )
	{
		allObjPts.push_back(Point3f(landModel.at<float>(i,0),landModel.at<float>(i,1),landModel.at<float>(i,2)));
		allObj2DPts.push_back(Point2f(landImage.at<float>(i,0),landImage.at<float>(i,1)));
	}
	//printf("allImgPts %d\n",landModel.rows);
	allImgPts.reserve(landModel.rows);

	int* sinds = new int[landModel.rows];
	for (int iter=0;iter<100;iter++){
		for (int i=0;i<landModel.rows;i++)
			sinds[i] = i;
		for (int i=0;i<sample;i++) {
			int ind = rng.next() % landModel.rows;
			if (ind != i)
			{
				int k = sinds[i];
				sinds[i] = sinds[ind];
				sinds[ind] = k;
			}
		}
		// PnP solver
		PnP.reset_correspondences();
		for(int ip = 0; ip < sample; ip++){
			int ind = sinds[ip];
			PnP.add_correspondence(allObjPts[ind].x,allObjPts[ind].y,allObjPts[ind].z, allObj2DPts[ind].x,allObj2DPts[ind].y);
		}
		double err2 = PnP.compute_pose(R_est, t_est);
		cv::Mat rMatP( 3, 3, CV_64F, R_est);
		cv::Rodrigues(rMatP, rVec);
		rVec.convertTo(rVec,CV_32F);
		cv::Mat tVecP( 3, 1, CV_64F, t_est);
		tVec = tVecP.clone();
		tVec.convertTo(tVec,CV_32F);
		
		//printf("Estimate 3d pose %d, (%d, %d, %d), (%d %d %d), (%d %d %d), (%d %d %d), %d\n",allObjPts.size(), 
		//	rVec.cols,rVec.rows, rVec.type(),tVec.cols,tVec.rows, tVec.type(),
		//	k_m.cols,k_m.rows, k_m.type(),distCoef.cols,distCoef.rows, distCoef.type(), allImgPts.size());
		cv::projectPoints( allObjPts, rVec, tVec, k_m, distCoef, allImgPts );

		int cons = 0;
		for ( int j=0; j<landModel.rows; ++j )
		{	
			float dist = pow( allObj2DPts[j].x - allImgPts[j].x, 2 ) \
				+ pow( allObj2DPts[j].y - allImgPts[j].y, 2 );
			if ( dist < 33 ) cons = cons + 1;
		}

		if (cons > bestCons) {
			r = rVec.clone();
			t = tVec.clone();
			bestCons = cons;
		}

	}
	//std::cout << "bestCons " << bestCons << std::endl;
	//std::cout << "r3D " << r << std::endl;
	//std::cout << "t3D " << t << std::endl;
	delete sinds;
}

cv::Mat BaselFaceEstimator::getLMByAlpha(cv::Mat alpha, float yaw, std::vector<int> inds, cv::Mat exprWeight){
	//int N = inds.size();
	//Mat lmMU(N*3,1,CV_32F);
	//Mat lmPC(N*3,alpha.rows,CV_32F);
	//Mat lmEV(alpha.rows,1,CV_32F);
	//for (int i=0;i<inds.size();i++){
	//	
	//	int ind;
	//	if (yaw > 0 && yaw < M_PI/9 && i < 8) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
	//	else if (yaw < 0 && yaw > -M_PI/9 && i > 8 && i < 17) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
	//	else
	//	ind = BaselFace::BaselFace_lmInd2[inds[i]]-1;
	//	//if (abs(yaw) < M_PI/20 || i > 14) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
	//	//else {
	//	//	if (i < 7) {
	//	//		if (yaw > -M_PI/10) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
	//	//		else  ind = BaselFace::BaselFace_lmInd2[inds[i]]-1;
	//	//	}
	//	//	else {
	//	//		if (yaw < M_PI/10) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
	//	//		else  ind = BaselFace::BaselFace_lmInd2[inds[i]]-1;
	//	//	}
	//	//}
	//	for (int j=0;j<3;j++) {
	//		lmMU.at<float>(3*i+j,0) = BaselFace::BaselFace_shapeMU[3*ind+j];
	//		for (int k=0;k<alpha.rows;k++) {
	//			lmPC.at<float>(3*i+j,k) = BaselFace::BaselFace_shapePC[(3*ind+j)*BaselFace::BaselFace_shapePC_w + k];
	//		}
	//	}
	//}
	//for (int k=0;k<alpha.rows;k++) {
	//	lmEV.at<float>(k,0) = BaselFace::BaselFace_shapeEV[k];
	//}
	//cv::Mat tmpShape = lmMU + lmPC * alpha.mul(lmEV);
	//return tmpShape.reshape(1,tmpShape.rows/3);
	
	cv::Mat alpha2 = alpha.clone();
	cv::Mat exp2 = exprWeight.clone();
	for (int i=0;i<alpha.rows;i++) alpha2.at<float>(i,0) *= BaselFace::BaselFace_shapeEV[i];
	for (int i=0;i<exp2.rows;i++) exp2.at<float>(i,0) *= BaselFace::BaselFace_expEV[i];
	int N = inds.size();
	int BPC = BaselFace::BaselFace_shapePC_w;
	int EPC = BaselFace::BaselFace_expPC_w;
	Mat tmpShape(N,3,CV_32F);
	float val;
	for (int i=0;i<inds.size();i++){
		
		int ind;
		if (yaw > 0 && yaw < M_PI/9 && i < 8) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
		else if (yaw < 0 && yaw > -M_PI/9 && i > 8 && i < 17) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
		else
		ind = BaselFace::BaselFace_lmInd2[inds[i]]-1;
		//float* p = BaselFace::BaselFace_shapePC + 3*ind*BPC;
		for (int j=0;j<3;j++) {
			val = BaselFace::BaselFace_shapeMU[3*ind+j] + BaselFace::BaselFace_expMU[3*ind+j];
			//float* pp = p + j*BPC;
			int k=0;
			for (;k<=alpha2.rows-5;k+=5) {
				val += alpha2.at<float>(k,0) * BaselFace::BaselFace_shapePC[(3*ind+j)*BPC + k] + alpha2.at<float>(k+1,0) * BaselFace::BaselFace_shapePC[(3*ind+j)*BPC + k+1]
					+ alpha2.at<float>(k+2,0) * BaselFace::BaselFace_shapePC[(3*ind+j)*BPC + k+2]
					+ alpha2.at<float>(k+3,0) * BaselFace::BaselFace_shapePC[(3*ind+j)*BPC + k+3]
					+ alpha2.at<float>(k+4,0) * BaselFace::BaselFace_shapePC[(3*ind+j)*BPC + k+4];
			}
			for (;k<alpha2.rows;k++) {
				val += alpha2.at<float>(k,0) * BaselFace::BaselFace_shapePC[(3*ind+j)*BPC + k];
			}
			for (k = 0;k<=exp2.rows-5;k+=5) {
				val += exp2.at<float>(k,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k] + exp2.at<float>(k+1,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+1]
					+ exp2.at<float>(k+2,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+2]
					+ exp2.at<float>(k+3,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+3]
					+ exp2.at<float>(k+4,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+4];
			}
			for (;k<exp2.rows;k++) {
				val += exp2.at<float>(k,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k];
			}
			tmpShape.at<float>(i,j) = val;
		}
	}
	alpha2.release();
	exp2.release();
	return tmpShape;
}

cv::Mat BaselFaceEstimator::getLMByAlphaParts(cv::Mat alpha, float yaw, std::vector<int> inds, cv::Mat exprWeight){
	//int N = inds.size();
	//int M = alpha.rows/4;
	//int numparts = 4;
	//cv::Mat tmpShape(N*3,1,CV_32F);
	//for (int i=0;i<inds.size();i++){
	//	int ind;
	//	if (yaw > 0 && yaw < M_PI/9 && i < 8) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
	//	else if (yaw < 0 && yaw > -M_PI/9 && i > 8 && i < 17) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
	//	else
	//		ind = BaselFace::BaselFace_lmInd2[inds[i]]-1;
	//	//if (abs(yaw) < M_PI/20 || i > 14) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
	//	//else {
	//	//	if (i < 7) {
	//	//		if (yaw > -M_PI/10) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
	//	//		else  ind = BaselFace::BaselFace_lmInd2[inds[i]]-1;
	//	//	}
	//	//	else {
	//	//		if (yaw < M_PI/10) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
	//	//		else  ind = BaselFace::BaselFace_lmInd2[inds[i]]-1;
	//	//	}
	//	//}
	//	for (int j=0;j<3;j++) {
	//		float val = BaselFace::BaselFace_shapeMU[3*ind+j];
	//		for (int k=0;k<M;k++) {
	//			float tm = BaselFace::BaselFace_shapePC[(3*ind+j)*BaselFace::BaselFace_shapePC_w + k]*BaselFace::BaselFace_shapeEV[k];
	//			for (int p=0;p<numparts;p++) {
	//				val += alpha.at<float>(p*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+p] * tm;
	//			}
	//		}
	//		tmpShape.at<float>(3*i+j,0) = val;
	//	}
	//}
	//return tmpShape.reshape(1,tmpShape.rows/3);

	
	int N = inds.size();
	int numparts = 4;
	int M = alpha.rows/numparts;
	cv::Mat alpha2 = alpha.clone();
	cv::Mat exp2 = exprWeight.clone();
	for (int i=0;i<numparts;i++) 
		for (int j=0;j<M;j++) alpha2.at<float>(i*M+j,0) *= BaselFace::BaselFace_shapeEV[j];
	for (int i=0;i<exp2.rows;i++) exp2.at<float>(i,0) *= BaselFace::BaselFace_expEV[i];
	
	int EPC = BaselFace::BaselFace_expPC_w;
	Mat tmpShape(N,3,CV_32F);
	for (int i=0;i<inds.size();i++){
		int ind;
		if (yaw > 0 && yaw < M_PI/9 && i < 8) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
		else if (yaw < 0 && yaw > -M_PI/9 && i > 8 && i < 17) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
		else
			ind = BaselFace::BaselFace_lmInd2[inds[i]]-1;
		//if (abs(yaw) < M_PI/20 || i > 14) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
		//else {
		//	if (i < 7) {
		//		if (yaw > -M_PI/10) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
		//		else  ind = BaselFace::BaselFace_lmInd2[inds[i]]-1;
		//	}
		//	else {
		//		if (yaw < M_PI/10) ind = BaselFace::BaselFace_lmInd[inds[i]]-1;
		//		else  ind = BaselFace::BaselFace_lmInd2[inds[i]]-1;
		//	}
		//}
		for (int j=0;j<3;j++) {
			float val = BaselFace::BaselFace_shapeMU[3*ind+j] + BaselFace::BaselFace_expMU[3*ind+j];
			int k=0;
			for (;k<=M-2;k+=2) {
				float tm1 = BaselFace::BaselFace_shapePC[(3*ind+j)*BaselFace::BaselFace_shapePC_w + k];
				float tm2 = BaselFace::BaselFace_shapePC[(3*ind+j)*BaselFace::BaselFace_shapePC_w + k+1];
				//for (int p=0;p<numparts;p++){
					val += (alpha2.at<float>(k,0) * BaselFace::BaselFace_wparts[ind*numparts]
						+ alpha2.at<float>(M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+1]
						+ alpha2.at<float>(2*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+2]
						+ alpha2.at<float>(3*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+3]) * tm1
						+ (alpha2.at<float>(k+1,0) * BaselFace::BaselFace_wparts[ind*numparts]
						+ alpha2.at<float>(M+k+1,0) * BaselFace::BaselFace_wparts[ind*numparts+1]
						+ alpha2.at<float>(2*M+k+1,0) * BaselFace::BaselFace_wparts[ind*numparts+2]
						+ alpha2.at<float>(3*M+k+1,0) * BaselFace::BaselFace_wparts[ind*numparts+3]) * tm2;
			}
			for (;k<M;k++) {
				float tm = BaselFace::BaselFace_shapePC[(3*ind+j)*BaselFace::BaselFace_shapePC_w + k];
					val += (alpha2.at<float>(k,0) * BaselFace::BaselFace_wparts[ind*numparts]
						+ alpha2.at<float>(M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+1]
						+ alpha2.at<float>(2*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+2]
						+ alpha2.at<float>(3*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+3]) * tm;
			}
			for (k = 0;k<=exp2.rows-5;k+=5) {
				val += exp2.at<float>(k,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k] + exp2.at<float>(k+1,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+1]
					+ exp2.at<float>(k+2,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+2]
					+ exp2.at<float>(k+3,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+3]
					+ exp2.at<float>(k+4,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+4];
			}
			for (;k<exp2.rows;k++) {
				val += exp2.at<float>(k,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k];
			}
			tmpShape.at<float>(i,j) = val;
		}
	}
	alpha2.release();
	exp2.release();
	return tmpShape;
}


cv::Mat BaselFaceEstimator::getShape2(cv::Mat alpha, cv::Mat exprWeight){
	cv::Mat alpha2 = alpha.clone();
	cv::Mat exp2 = exprWeight.clone();
	for (int i=0;i<alpha.rows;i++) alpha2.at<float>(i,0) *= BaselFace::BaselFace_shapeEV[i];
	for (int i=0;i<exp2.rows;i++) exp2.at<float>(i,0) *= BaselFace::BaselFace_expEV[i];
	int N = BaselFace::BaselFace_shapePC_h/3;
	int BPC = BaselFace::BaselFace_shapePC_w;
	int EPC = BaselFace::BaselFace_expPC_w;
	Mat tmpShape(N,3,CV_32F);
	#pragma omp parallel for
	for (int i=0;i<N;i++){
		for (int j=0;j<3;j++) {
			float val = BaselFace::BaselFace_shapeMU[3*i+j] + BaselFace::BaselFace_expMU[3*i+j];
			//float* pp = p + j*BPC;
			int k=0;
			for (;k<=alpha2.rows-5;k+=5) {
				val += alpha2.at<float>(k,0) * BaselFace::BaselFace_shapePC[(3*i+j)*BPC + k] + alpha2.at<float>(k+1,0) * BaselFace::BaselFace_shapePC[(3*i+j)*BPC + k+1]
					+ alpha2.at<float>(k+2,0) * BaselFace::BaselFace_shapePC[(3*i+j)*BPC + k+2]
					+ alpha2.at<float>(k+3,0) * BaselFace::BaselFace_shapePC[(3*i+j)*BPC + k+3]
					+ alpha2.at<float>(k+4,0) * BaselFace::BaselFace_shapePC[(3*i+j)*BPC + k+4];
			}
			for (;k<alpha2.rows;k++) {
				val += alpha2.at<float>(k,0) * BaselFace::BaselFace_shapePC[(3*i+j)*BPC + k];
			}
			for (k = 0;k<=exp2.rows-5;k+=5) {
				val += exp2.at<float>(k,0) * BaselFace::BaselFace_expPC[(3*i+j)*EPC + k] + exp2.at<float>(k+1,0) * BaselFace::BaselFace_expPC[(3*i+j)*EPC + k+1]
					+ exp2.at<float>(k+2,0) * BaselFace::BaselFace_expPC[(3*i+j)*EPC + k+2]
					+ exp2.at<float>(k+3,0) * BaselFace::BaselFace_expPC[(3*i+j)*EPC + k+3]
					+ exp2.at<float>(k+4,0) * BaselFace::BaselFace_expPC[(3*i+j)*EPC + k+4];
			}
			for (;k<exp2.rows;k++) {
				val += exp2.at<float>(k,0) * BaselFace::BaselFace_expPC[(3*i+j)*EPC + k];
			}
			tmpShape.at<float>(i,j) = val;
		}
	}
	alpha2.release();
	exp2.release();
	return tmpShape;
}

cv::Mat BaselFaceEstimator::getTriByAlpha(cv::Mat alpha, std::vector<int> inds, cv::Mat exprWeight){
	//int N = inds.size();
	//Mat lmMU(N*3,1,CV_32F);
	//Mat lmPC(N*3,alpha.rows,CV_32F);
	//Mat lmEV(alpha.rows,1,CV_32F);
	//for (int i=0;i<inds.size();i++){
	//	int ind = inds[i];
	//	for (int j=0;j<3;j++) {
	//		lmMU.at<float>(3*i+j,0) = BaselFace::BaselFace_shapeMU[3*ind+j];
	//		for (int k=0;k<alpha.rows;k++) {
	//			lmPC.at<float>(3*i+j,k) = BaselFace::BaselFace_shapePC[(3*ind+j)*BaselFace::BaselFace_shapePC_w + k];
	//		}
	//	}
	//}
	//for (int k=0;k<alpha.rows;k++) {
	//	lmEV.at<float>(k,0) = BaselFace::BaselFace_shapeEV[k];
	//}
	//cv::Mat tmpShape = lmMU + lmPC * alpha.mul(lmEV);
	//return tmpShape.reshape(1,tmpShape.rows/3);
	cv::Mat alpha2 = alpha.clone();
	for (int i=0;i<alpha.rows;i++) alpha2.at<float>(i,0) *= BaselFace::BaselFace_shapeEV[i];
	cv::Mat exp2 = exprWeight.clone();
	for (int i=0;i<exp2.rows;i++) exp2.at<float>(i,0) *= BaselFace::BaselFace_expEV[i];
	
	int EPC = BaselFace::BaselFace_expPC_w;
	int N = inds.size();
	int BPC = BaselFace::BaselFace_shapePC_w;
	Mat tmpShape(N,3,CV_32F);
	#pragma loop(hint_parallel(8))
	for (int i=0;i<N;i++){
		int ind = inds[i];
		//float* p = BaselFace::BaselFace_shapePC + 3*ind*BPC;
		for (int j=0;j<3;j++) {
			float val = BaselFace::BaselFace_shapeMU[3*ind+j] + BaselFace::BaselFace_expMU[3*ind+j];
			//float* pp = p + j*BPC;
			int k=0;
			for (;k<=alpha2.rows-5;k+=5) {
				val += alpha2.at<float>(k,0) * BaselFace::BaselFace_shapePC[(3*ind+j)*BPC + k] + alpha2.at<float>(k+1,0) * BaselFace::BaselFace_shapePC[(3*ind+j)*BPC + k+1]
					+ alpha2.at<float>(k+2,0) * BaselFace::BaselFace_shapePC[(3*ind+j)*BPC + k+2]
					+ alpha2.at<float>(k+3,0) * BaselFace::BaselFace_shapePC[(3*ind+j)*BPC + k+3]
					+ alpha2.at<float>(k+4,0) * BaselFace::BaselFace_shapePC[(3*ind+j)*BPC + k+4];
			}
			for (;k<alpha2.rows;k++) {
				val += alpha2.at<float>(k,0) * BaselFace::BaselFace_shapePC[(3*ind+j)*BPC + k];
			}
			for (k = 0;k<=exp2.rows-5;k+=5) {
				val += exp2.at<float>(k,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k] + exp2.at<float>(k+1,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+1]
					+ exp2.at<float>(k+2,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+2]
					+ exp2.at<float>(k+3,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+3]
					+ exp2.at<float>(k+4,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+4];
			}
			for (;k<exp2.rows;k++) {
				val += exp2.at<float>(k,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k];
			}
			tmpShape.at<float>(i,j) = val;
		}
	}
	alpha2.release();
	return tmpShape;
}
	
cv::Mat BaselFaceEstimator::getTriByBeta(cv::Mat beta, std::vector<int> inds){
	//cv::Mat beta2 = beta.clone();
	//for (int i=0;i<beta.rows;i++) beta2.at<float>(i,0) *= BaselFace::BaselFace_texEV[i];
	//int N = inds.size();
	//Mat lmMU(N*3,1,CV_32F);
	//Mat lmPC(N*3,beta.rows,CV_32F);
	//Mat lmEV(beta.rows,1,CV_32F);
	//for (int i=0;i<inds.size();i++){
	//	int ind = inds[i];
	//	for (int j=0;j<3;j++) {
	//		lmMU.at<float>(3*i+j,0) = BaselFace::BaselFace_texMU[3*ind+j];
	//		for (int k=0;k<beta.rows;k++) {
	//			lmPC.at<float>(3*i+j,k) = BaselFace::BaselFace_texPC[(3*ind+j)*BaselFace::BaselFace_texPC_w + k];
	//		}
	//	}
	//}
	//for (int k=0;k<beta.rows;k++) {
	//	lmEV.at<float>(k,0) = BaselFace::BaselFace_texEV[k];
	//}
	//cv::Mat tmptex = lmMU + lmPC * beta.mul(lmEV);
	//return tmptex.reshape(1,tmptex.rows/3);

	cv::Mat beta2 = beta.clone();
	for (int i=0;i<beta.rows;i++) beta2.at<float>(i,0) *= BaselFace::BaselFace_texEV[i];
	int N = inds.size();
	int BPC = BaselFace::BaselFace_texPC_w;
	Mat tmpShape(N,3,CV_32F);
	float val;
	for (int i=0;i<N;i++){
		int ind = inds[i];
		//float* p = BaselFace::BaselFace_shapePC + 3*ind*BPC;
		for (int j=0;j<3;j++) {
			val = BaselFace::BaselFace_texMU[3*ind+j];
			//float* pp = p + j*BPC;
			int k=0;
			for (;k<=beta2.rows-5;k+=5) {
				val += beta2.at<float>(k,0) * BaselFace::BaselFace_texPC[(3*ind+j)*BPC + k] + beta2.at<float>(k+1,0) * BaselFace::BaselFace_texPC[(3*ind+j)*BPC + k+1]
					+ beta2.at<float>(k+2,0) * BaselFace::BaselFace_texPC[(3*ind+j)*BPC + k+2]
					+ beta2.at<float>(k+3,0) * BaselFace::BaselFace_texPC[(3*ind+j)*BPC + k+3]
					+ beta2.at<float>(k+4,0) * BaselFace::BaselFace_texPC[(3*ind+j)*BPC + k+4];
			}
			for (;k<beta2.rows;k++) {
				val += beta2.at<float>(k,0) * BaselFace::BaselFace_texPC[(3*ind+j)*BPC + k];
			}
			tmpShape.at<float>(i,j) = val;
		}
	}
	beta2.release();
	return tmpShape;
}

cv::Mat BaselFaceEstimator::getTexture2(cv::Mat beta){
	cv::Mat beta2 = beta.clone();
	for (int i=0;i<beta.rows;i++) beta2.at<float>(i,0) *= BaselFace::BaselFace_texEV[i];
	int N = BaselFace::BaselFace_texPC_h/3;
	int BPC = BaselFace::BaselFace_texPC_w;
	Mat tmpShape(N,3,CV_32F);
	for (int i=0;i<N;i++){
		for (int j=0;j<3;j++) {
			float val = BaselFace::BaselFace_texMU[3*i+j];
			//float* pp = p + j*BPC;
			int k=0;
			for (;k<=beta2.rows-5;k+=5) {
				val += beta2.at<float>(k,0) * BaselFace::BaselFace_texPC[(3*i+j)*BPC + k] + beta2.at<float>(k+1,0) * BaselFace::BaselFace_texPC[(3*i+j)*BPC + k+1]
					+ beta2.at<float>(k+2,0) * BaselFace::BaselFace_texPC[(3*i+j)*BPC + k+2]
					+ beta2.at<float>(k+3,0) * BaselFace::BaselFace_texPC[(3*i+j)*BPC + k+3]
					+ beta2.at<float>(k+4,0) * BaselFace::BaselFace_texPC[(3*i+j)*BPC + k+4];
			}
			for (;k<beta2.rows;k++) {
				val += beta2.at<float>(k,0) * BaselFace::BaselFace_texPC[(3*i+j)*BPC + k];
			}
			tmpShape.at<float>(i,j) = val;
		}
	}
	beta2.release();
	return tmpShape;
}

cv::Mat BaselFaceEstimator::getTriByAlphaParts(cv::Mat alpha, std::vector<int> inds, cv::Mat exprWeight){
	//int N = inds.size();
	//int numparts = 4;
	//int M = alpha.rows/numparts;
	////cv::Mat alpha2 = alpha.clone();
	////for (int i=0;i<numparts;i++) 
	////	for (int j=0;j<M;j++) alpha2.at<float>(i*M+j,0) *= BaselFace::BaselFace_shapeEV[j];

	//Mat tmpShape(N,3,CV_32F);
	//for (int i=0;i<inds.size();i++){
	//	int ind = inds[i];
	//	for (int j=0;j<3;j++) {
	//		float val = BaselFace::BaselFace_shapeMU[3*ind+j];
	//		for (int k=0;k<M;k++) {
	//			float tm = BaselFace::BaselFace_shapePC[(3*ind+j)*BaselFace::BaselFace_shapePC_w + k]*BaselFace::BaselFace_shapeEV[k];
	//			for (int p=0;p<numparts;p++){
	//				val += alpha.at<float>(p*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+p] * tm;
	//				//lmPC.at<float>(3*i+j,k) = BaselFace::BaselFace_shapePC[(3*ind+j)*BaselFace::BaselFace_shapePC_w + k];
	//			}
	//		}
	//		tmpShape.at<float>(i,j) = val;
	//	}
	//}
	////alpha2.release();
	//return tmpShape;

	int N = inds.size();
	int numparts = 4;
	int M = alpha.rows/numparts;
	cv::Mat alpha2 = alpha.clone();
	for (int i=0;i<numparts;i++) 
		for (int j=0;j<M;j++) alpha2.at<float>(i*M+j,0) *= BaselFace::BaselFace_shapeEV[j];
	cv::Mat exp2 = exprWeight.clone();
	for (int i=0;i<exp2.rows;i++) exp2.at<float>(i,0) *= BaselFace::BaselFace_expEV[i];
	
	int EPC = BaselFace::BaselFace_expPC_w;

	Mat tmpShape(N,3,CV_32F);
	for (int i=0;i<N;i++){
		int ind = inds[i];
		for (int j=0;j<3;j++) {
			float val = BaselFace::BaselFace_shapeMU[3*ind+j] + BaselFace::BaselFace_expMU[3*ind+j];
			int k=0;
			for (;k<=M-2;k+=2) {
				float tm1 = BaselFace::BaselFace_shapePC[(3*ind+j)*BaselFace::BaselFace_shapePC_w + k];
				float tm2 = BaselFace::BaselFace_shapePC[(3*ind+j)*BaselFace::BaselFace_shapePC_w + k+1];
				//for (int p=0;p<numparts;p++){
					val += (alpha2.at<float>(k,0) * BaselFace::BaselFace_wparts[ind*numparts]
						+ alpha2.at<float>(M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+1]
						+ alpha2.at<float>(2*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+2]
						+ alpha2.at<float>(3*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+3]) * tm1
						+ (alpha2.at<float>(k+1,0) * BaselFace::BaselFace_wparts[ind*numparts]
						+ alpha2.at<float>(M+k+1,0) * BaselFace::BaselFace_wparts[ind*numparts+1]
						+ alpha2.at<float>(2*M+k+1,0) * BaselFace::BaselFace_wparts[ind*numparts+2]
						+ alpha2.at<float>(3*M+k+1,0) * BaselFace::BaselFace_wparts[ind*numparts+3]) * tm2;
			}
			for (;k<M;k++) {
				float tm = BaselFace::BaselFace_shapePC[(3*ind+j)*BaselFace::BaselFace_shapePC_w + k];
					val += (alpha2.at<float>(k,0) * BaselFace::BaselFace_wparts[ind*numparts]
						+ alpha2.at<float>(M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+1]
						+ alpha2.at<float>(2*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+2]
						+ alpha2.at<float>(3*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+3]) * tm;
			}
			for (k = 0;k<=exp2.rows-5;k+=5) {
				val += exp2.at<float>(k,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k] + exp2.at<float>(k+1,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+1]
					+ exp2.at<float>(k+2,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+2]
					+ exp2.at<float>(k+3,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+3]
					+ exp2.at<float>(k+4,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k+4];
			}
			for (;k<exp2.rows;k++) {
				val += exp2.at<float>(k,0) * BaselFace::BaselFace_expPC[(3*ind+j)*EPC + k];
			}
			tmpShape.at<float>(i,j) = val;
		}
	}
	alpha2.release();
	return tmpShape;
}
	
cv::Mat BaselFaceEstimator::getTriByBetaParts(cv::Mat beta, std::vector<int> inds){
	//int N = inds.size();
	//int numparts = 4;
	//int M = beta.rows/numparts;
	//Mat tmpTex(N*3,1,CV_32F);
	//for (int i=0;i<inds.size();i++){
	//	int ind = inds[i];
	//	for (int j=0;j<3;j++) {
	//		float val = BaselFace::BaselFace_texMU[3*ind+j];
	//		for (int k=0;k<M;k++) {
	//			float tm = BaselFace::BaselFace_texPC[(3*ind+j)*BaselFace::BaselFace_texPC_w + k]*BaselFace::BaselFace_texEV[k];
	//			for (int p=0;p<numparts;p++){
	//				val += beta.at<float>(p*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+p] * tm;
	//				//lmPC.at<float>(3*i+j,k) = BaselFace::BaselFace_shapePC[(3*ind+j)*BaselFace::BaselFace_shapePC_w + k];
	//			}
	//		}
	//		tmpTex.at<float>(3*i+j,0) = val;
	//	}
	//}
	//return tmpTex.reshape(1,tmpTex.rows/3);
		
	int N = inds.size();
	int numparts = 4;
	int M = beta.rows/numparts;
	cv::Mat beta2 = beta.clone();
	for (int i=0;i<numparts;i++) 
		for (int j=0;j<M;j++) beta2.at<float>(i*M+j,0) *= BaselFace::BaselFace_texEV[j];

	Mat tmpShape(N,3,CV_32F);
	for (int i=0;i<N;i++){
		int ind = inds[i];
		for (int j=0;j<3;j++) {
			float val = BaselFace::BaselFace_texMU[3*ind+j];
			int k=0;
			for (;k<=M-2;k+=2) {
				float tm1 = BaselFace::BaselFace_texPC[(3*ind+j)*BaselFace::BaselFace_texPC_w + k];
				float tm2 = BaselFace::BaselFace_texPC[(3*ind+j)*BaselFace::BaselFace_texPC_w + k+1];
				//for (int p=0;p<numparts;p++){
					val += (beta2.at<float>(k,0) * BaselFace::BaselFace_wparts[ind*numparts]
						+ beta2.at<float>(M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+1]
						+ beta2.at<float>(2*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+2]
						+ beta2.at<float>(3*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+3]) * tm1
						+ (beta2.at<float>(k+1,0) * BaselFace::BaselFace_wparts[ind*numparts]
						+ beta2.at<float>(M+k+1,0) * BaselFace::BaselFace_wparts[ind*numparts+1]
						+ beta2.at<float>(2*M+k+1,0) * BaselFace::BaselFace_wparts[ind*numparts+2]
						+ beta2.at<float>(3*M+k+1,0) * BaselFace::BaselFace_wparts[ind*numparts+3]) * tm2;
			}
			for (;k<M;k++) {
				float tm = BaselFace::BaselFace_texPC[(3*ind+j)*BaselFace::BaselFace_texPC_w + k];
					val += (beta2.at<float>(k,0) * BaselFace::BaselFace_wparts[ind*numparts]
						+ beta2.at<float>(M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+1]
						+ beta2.at<float>(2*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+2]
						+ beta2.at<float>(3*M+k,0) * BaselFace::BaselFace_wparts[ind*numparts+3]) * tm;
			}
			tmpShape.at<float>(i,j) = val;
		}
	}
	beta2.release();
	return tmpShape;
}

cv::Mat BaselFaceEstimator::estimateShape3D(cv::Mat landModel, cv::Mat landImage, cv::Mat k_m, cv::Mat r, cv::Mat t){
	printf("estimateShape3D\n");
	float THRESH = 3.0;
	int M = 29;
	int N = landImage.rows;
	Mat matR;
	cv::Rodrigues(r,matR);
	Mat P;
	hconcat(matR,t,P);
	P = k_m*P;
	cv::Mat mu(68,3,CV_32F);

	float* pPC = new float[3*M*N]; 
	//Mat PC(M*N,3,CV_32F,pPC);
	for (int i=0;i<M;i++){
		for (int j=0;j<N;j++){
			int ind = BaselFace::BaselFace_lmInd2[j] - 1;
			for (int k=0;k<3;k++){
				mu.at<float>(j,k) = BaselFace::BaselFace_shapeMU[3*ind + k] + BaselFace::BaselFace_expMU[3*ind + k];
				pPC[3*N*i + 3*j + k] = BaselFace::BaselFace_expPC[3*BaselFace::BaselFace_expPC_w*ind + k*BaselFace::BaselFace_expPC_w + i] * BaselFace::BaselFace_expEV[i];
			}
		}
	}
	//write_plyFloat("mu.ply",mu.t());

	float* pB = new float[2*N+M];
	memset(pB,0,(2*N+M)*sizeof(float));

	for (int i=0;i<N;i++){
		pB[2*i] = - (P.at<float>(0,0)*landModel.at<float>(i,0) + P.at<float>(0,1)*landModel.at<float>(i,1) + P.at<float>(0,2)*landModel.at<float>(i,2)) +
			landImage.at<float>(i,0) * (P.at<float>(2,0)*landModel.at<float>(i,0) + P.at<float>(2,1)*landModel.at<float>(i,1) + P.at<float>(2,2)*landModel.at<float>(i,2));
		pB[2*i+1] = - (P.at<float>(1,0)*landModel.at<float>(i,0) + P.at<float>(1,1)*landModel.at<float>(i,1) + P.at<float>(1,2)*landModel.at<float>(i,2)) +
			landImage.at<float>(i,1) * (P.at<float>(2,0)*landModel.at<float>(i,0) + P.at<float>(2,1)*landModel.at<float>(i,1) + P.at<float>(2,2)*landModel.at<float>(i,2));
	}
	Mat b(2*N+M,1,CV_32F,pB);

	float* pA = new float[2*N*M];
	for (int i=0;i<N;i++){
		for (int j=0;j<M;j++){
			pA[2*i*M+j] = P.at<float>(0,0)*pPC[3*N*j + 3*i] + P.at<float>(0,1)*pPC[3*N*j + 3*i+1] + P.at<float>(0,2)*pPC[3*N*j + 3*i+2] -
				landImage.at<float>(i,0) * (P.at<float>(2,0)*pPC[3*N*j + 3*i] + P.at<float>(2,1)*pPC[3*N*j + 3*i+1] + P.at<float>(2,2)*pPC[3*N*j + 3*i+2]);
			pA[(2*i+1)*M+j] = P.at<float>(1,0)*pPC[3*N*j + 3*i] + P.at<float>(1,1)*pPC[3*N*j + 3*i+1] + P.at<float>(1,2)*pPC[3*N*j + 3*i+2] -
				landImage.at<float>(i,1) * (P.at<float>(2,0)*pPC[3*N*j + 3*i] + P.at<float>(2,1)*pPC[3*N*j + 3*i+1] + P.at<float>(2,2)*pPC[3*N*j + 3*i+2]);
		}
	}
	Mat A0(2*N,M,CV_32F,pA);

	Mat A1;
	vconcat(A0, 100000*Mat::eye(M,M,CV_32F),A1);

	Mat alpha;
	solve(A1,b,alpha,DECOMP_QR);

	Mat thresh1 = (alpha > THRESH)/255;
	Mat thresh2 = (alpha < -THRESH)/255;
	thresh1.convertTo(thresh1,CV_32F);
	thresh2.convertTo(thresh2,CV_32F);
	alpha = THRESH * thresh1 - THRESH * thresh2 + (1 - thresh1 - thresh2).mul(alpha);
	delete pPC;
	delete pB;
	delete pA;
	return alpha;
}


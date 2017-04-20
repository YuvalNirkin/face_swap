/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "FImRenderer.h"
#include <opencv2/calib3d.hpp>

using namespace cv;

FImRenderer::FImRenderer(cv::Mat img)
{
	face_ = new Face();
	img_ = img.clone();
	render_ = new FBRender( img_.cols, img_.rows,true );
	zNear = 50;
	zFar = 10000;
}

// Yuval
void FImRenderer::init(const cv::Mat & img)
{
    img_ = img.clone();
    render_->init(img.cols, img.rows);
}

FImRenderer::~FImRenderer(void)
{
	delete render_;
}
void FImRenderer::loadPLYFile(char* filename, bool useTexture){
	face_->loadPLYModel2(filename);
	if (useTexture && face_->mesh_.colors_){
		delete face_->mesh_.colors_;
		face_->mesh_.colors_ = 0;
		face_->mesh_.colorid = 0;
	}
}

void FImRenderer::loadMesh(cv::Mat shape, cv::Mat tex, cv::Mat faces){
	face_->loadMesh(shape, tex, faces);
}

void FImRenderer::computeTexture(float *rot, float *t, float f){
	if (face_->mesh_.colors_){
		delete face_->mesh_.colors_;
		face_->mesh_.colors_ = 0;
		face_->mesh_.colorid = 0;
	}
	float *vert = face_->mesh_.vertices_;	
	//printf("computeTexture %f %f %f %f %f %f %f %d %d\n",rot[0],rot[1],rot[2],t[0],t[1],t[2],f,img_.rows,img_.cols);
	face_->mesh_.tex_.img_ = img_;

	face_->mesh_.texcoords_ = new float[3*face_->mesh_.nVertices_];
	face_->mesh_.texdepth_ = new float[face_->mesh_.nVertices_];
	float *projPts = face_->mesh_.texcoords_;								//Triangles

	cv::Mat mVert(face_->mesh_.nVertices_,3,CV_32F, vert);
	cv::Mat rVec(3,1,CV_32F, rot);
	cv::Mat tVec(3,1,CV_32F, t);
	cv::Mat rMat(3,3,CV_32F);
	cv::Rodrigues(rVec,rMat);
	cv::Mat P(3,4,CV_32F);
	cv::hconcat(rMat,tVec,P);

	cv::Mat pt3D(4,face_->mesh_.nVertices_,CV_32F);
	cv::vconcat(mVert.t(),cv::Mat::ones(1,face_->mesh_.nVertices_,CV_32F),pt3D);
	float K_[] = { -f, 0.f, img_.cols/2, 0.f, f, img_.rows/2, 0.f, 0.f, 1 };
	cv::Mat k_m( 3, 3, CV_32F,  K_);

	cv::Mat pt3D_2 = k_m * P * pt3D;
	for (int i=0;i<face_->mesh_.nVertices_;i++){
		float z = pt3D_2.at<float>(2,i);
		face_->mesh_.texdepth_[i] = z;
		//if (i < 2)
		//	printf("texdepth_ %f\n",z);
		projPts[i*2] = pt3D_2.at<float>(0,i)/(z*img_.cols);
		projPts[i*2+1] = pt3D_2.at<float>(1,i)/(z*img_.rows);
	}
}

void FImRenderer::computeTexture(cv::Mat rVec, cv::Mat tVec, float f){
	if (face_->mesh_.colors_){
		delete face_->mesh_.colors_;
		face_->mesh_.colors_ = 0;
		face_->mesh_.colorid = 0;
	}
	float *vert = face_->mesh_.vertices_;	
	//printf("computeTexture %f %f %f %f %f %f %f %d %d\n",rot[0],rot[1],rot[2],t[0],t[1],t[2],f,img_.rows,img_.cols);
	face_->mesh_.tex_.img_ = img_;

	face_->mesh_.texcoords_ = new float[3*face_->mesh_.nVertices_];
	face_->mesh_.texdepth_ = new float[face_->mesh_.nVertices_];
	float *projPts = face_->mesh_.texcoords_;								//Triangles

	cv::Mat mVert(face_->mesh_.nVertices_,3,CV_32F, vert);
	cv::Mat rMat(3,3,CV_32F);
	cv::Rodrigues(rVec,rMat);
	cv::Mat P(3,4,CV_32F);
	cv::hconcat(rMat,tVec,P);

	cv::Mat pt3D(4,face_->mesh_.nVertices_,CV_32F);
	cv::vconcat(mVert.t(),cv::Mat::ones(1,face_->mesh_.nVertices_,CV_32F),pt3D);
	float K_[] = { -f, 0.f, img_.cols/2, 0.f, f, img_.rows/2, 0.f, 0.f, 1 };
	cv::Mat k_m( 3, 3, CV_32F,  K_);

	cv::Mat pt3D_2 = k_m * P * pt3D;
	for (int i=0;i<face_->mesh_.nVertices_;i++){
		float z = pt3D_2.at<float>(2,i);
		face_->mesh_.texdepth_[i] = z;
		//if (i < 2)
		//	printf("texdepth_ %f\n",z);
		projPts[i*2] = pt3D_2.at<float>(0,i)/(z*img_.cols);
		projPts[i*2+1] = pt3D_2.at<float>(1,i)/(z*img_.rows);
	}
}

void FImRenderer::loadModel(char* meshFile){
	if (face_->mesh_.texcoords_)
	{
	//printf("loadImageGL\n");
		face_->mesh_.tex_.loadImageGL();
	}
	//printf("loadGeometryGL\n");
	face_->mesh_.loadGeometryGL(true);
	if (meshFile != 0 && strlen(meshFile) > 1)
	face_->savePLYModel(meshFile);
}

void FImRenderer::setIm(cv::Mat inIm){
	img_ = inIm.clone();
}

void
FImRenderer::mapRendering( float *rot, float *t, float f, cv::Mat *img, cv::Mat *mask )
{
	glFlush();
	//setup camera intrinsics
	CvMat *K = cvCreateMat( 3, 3, CV_64FC1 );
	cvSetIdentity( K );
	cvmSet( K, 0, 0, f  );
	cvmSet( K, 1, 1, f  );
	cvmSet( K, 0, 2, img_.cols/2 );
	cvmSet( K, 1, 2, img_.rows/2 );
	render_->loadIntrinGL( K, zNear, zFar, img_.cols, img_.rows );
	cvReleaseMat( &K );

	CvMat *H = cvCreateMat( 4, 4, CV_64FC1 );
	cvSetIdentity( H );
	render_->loadExtrinGL( H );
	cvReleaseMat( &H );

	//update extrinsics and render
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();	

	//Translations
	glTranslatef( t[0], t[1], t[2] );

	//Axis angle
	//rot = { tRx, tRy, tRz } , angle = ||rot|| 
	float angle = sqrt( rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2] );
	glRotatef( angle/3.14159f*180.f, rot[0]/angle, rot[1]/angle, rot[2]/angle );

	render_->mapRendering( face_->mesh_ );
	glPopMatrix();	
	glFlush();
	//glutSwapBuffers ();
	
	//for (int i=0;i<10;i++) {
	/*if ( img  != NULL ) */render_->readFB( *img  );
	/*if ( mask != NULL ) */render_->readDB( *mask );
//}
	//render_->readFB( *img  );
	//render_->readDB( *mask );

	//return inArray;
}
void
FImRenderer::unmapRendering()
{
	render_->unmapRendering();
}
void FImRenderer::render( float *r, float *t, float f, cv::Mat &color, cv::Mat &depth ){
	double mn, mx;
	for (int i=0;i<5;i++){
		mapRendering(r,t,f,&color,&depth);unmapRendering();
		//mapRendering(r,t,f,&color,&depth);unmapRendering();
		cv::minMaxLoc(depth, &mn, &mx);
		if (mx != mn) break;
	}
}

void FImRenderer::resetRenderer(Face* face2, bool resetTexture, float *rot, float *t, float f){
	delete render_;
	delete face_;
	render_ = new FBRender( img_.cols, img_.rows,true );
	face_ = face2;
	if (resetTexture)
		this->computeTexture(rot, t, f);
	loadModel("new.ply");
}

void FImRenderer::computeNormals(){
	face_->estimateNormals();
}

void FImRenderer::copyColors(cv::Mat colors){
	if (face_->mesh_.colors_ == 0) face_->mesh_.colors_ = new float[4*face_->mesh_.nVertices_];
	for (int i=0;i<face_->mesh_.nVertices_;i++){
		face_->mesh_.colors_ [4*i] = colors.at<float>(i,0)/255.0f;
		face_->mesh_.colors_ [4*i+1] = colors.at<float>(i,1)/255.0f;
		face_->mesh_.colors_ [4*i+2] = colors.at<float>(i,2)/255.0f;
		face_->mesh_.colors_ [4*i+3] = 1.0f;
	}
}

void FImRenderer::copyNormals(cv::Mat normals){
	if (face_->mesh_.normals == 0) face_->mesh_.normals = new float[3*face_->mesh_.nVertices_];
	for (int i=0;i<face_->mesh_.nVertices_;i++){
		face_->mesh_.normals [3*i] = normals.at<float>(i,0);
		face_->mesh_.normals [3*i+1] = normals.at<float>(i,1);
		face_->mesh_.normals [3*i+2] = normals.at<float>(i,2);
	}
}
void FImRenderer::copyShape(cv::Mat shape){
	for (int i=0;i<face_->mesh_.nVertices_;i++){
		face_->mesh_.vertices_ [3*i] = shape.at<float>(i,0);
		face_->mesh_.vertices_ [3*i+1] = shape.at<float>(i,1);
		face_->mesh_.vertices_ [3*i+2] = shape.at<float>(i,2);
	}
}
void FImRenderer::copyFaces(cv::Mat faces){
	for (int i=0;i<face_->mesh_.nFaces_;i++){
		face_->mesh_.faces_ [3*i] = faces.at<int>(i,0);
		face_->mesh_.faces_ [3*i+1] = faces.at<int>(i,1);
		face_->mesh_.faces_ [3*i+2] = faces.at<int>(i,2);
	}
}

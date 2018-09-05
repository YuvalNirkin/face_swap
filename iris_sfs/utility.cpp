/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "utility.h"

void write_ply(char* outname, std::vector<cv::Point3f> points){
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << points.size() << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "end_header\n";
	for( int i = 0; i < points.size() ; i++ )
	{
		ply2 << points[i].x << " " << points[i].y << " " << points[i].z << std::endl;
	}
	ply2.close();

}
void write_ply(char* outname, cv::Mat mat_Depth, std::vector<cv::Vec3b> colors){
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << mat_Depth.cols << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "property uchar red\n";
	ply2 << "property uchar green\n";
	ply2 << "property uchar blue\n";
	ply2 << "end_header\n";
	for( int i = 0; i < mat_Depth.cols ; i++ )
	{
		ply2 << mat_Depth.at<double>(0,i) << " " << mat_Depth.at<double>(1,i) << " " << mat_Depth.at<double>(2,i) 
			<< " " << (int)colors[i][2] << " " << (int)colors[i][1] << " " << (int)colors[i][0] << std::endl;
	}
	ply2.close();
}

void write_ply(char* outname, cv::Mat mat_Depth){
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << mat_Depth.cols << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "end_header\n";
	for( int i = 0; i < mat_Depth.cols ; i++ )
	{
		ply2 << mat_Depth.at<double>(0,i) << " " << mat_Depth.at<double>(1,i) << " " << mat_Depth.at<double>(2,i) << std::endl;
	}
	ply2.close();
}

void write_plyFloat(char* outname, cv::Mat mat_Depth){
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << mat_Depth.cols << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "end_header\n";
	for( int i = 0; i < mat_Depth.cols ; i++ )
	{
		ply2 << mat_Depth.at<float>(0,i) << " " << mat_Depth.at<float>(1,i) << " " << mat_Depth.at<float>(2,i) << std::endl;
	}
	ply2.close();
}

void write_ply(char* outname, cv::Mat mat_Depth, cv::Mat mat_Faces){
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << mat_Depth.cols << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "element face " << mat_Faces.cols << std::endl;
	ply2 << "property list uchar int vertex_indices\n";
	ply2 << "end_header\n";
	for( int i = 0; i < mat_Depth.cols ; i++ )
	{
		ply2 << mat_Depth.at<double>(0,i) << " " << mat_Depth.at<double>(1,i) << " " << mat_Depth.at<double>(2,i) <<  std::endl;
	}
	for( int i = 0; i < mat_Faces.cols ; i++ )
	{
		ply2 << "3 " << mat_Faces.at<int>(0,i) << " " << mat_Faces.at<int>(1,i) << " " << mat_Faces.at<int>(2,i) << " " <<  std::endl;
	}
	ply2.close();
}

void write_ply(char* outname, cv::Mat mat_Depth, cv::Mat mat_Color, cv::Mat mat_Faces){
	mat_Color.convertTo(mat_Color,CV_8UC3);
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << mat_Depth.cols << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "property uchar red\n";
	ply2 << "property uchar green\n";
	ply2 << "property uchar blue\n";
	ply2 << "element face " << mat_Faces.cols << std::endl;
	ply2 << "property list uchar int vertex_indices\n";
	ply2 << "end_header\n";
	for( int i = 0; i < mat_Depth.cols ; i++ )
	{
		ply2 << mat_Depth.at<double>(0,i) << " " << mat_Depth.at<double>(1,i) << " " << mat_Depth.at<double>(2,i) << " "
			<<  (int)mat_Color.at<unsigned char>(0,i) << " " << (int)mat_Color.at<unsigned char>(1,i) << " " << (int)mat_Color.at<unsigned char>(2,i) << std::endl;
	}
	for( int i = 0; i < mat_Faces.cols ; i++ )
	{
		ply2 << "3 " << mat_Faces.at<int>(0,i) << " " << mat_Faces.at<int>(1,i) << " " << mat_Faces.at<int>(2,i) << " " <<  std::endl;
	}
	ply2.close();
}

Eigen::Matrix3Xd* toMatrix3Xd(cv::Mat mat){
	Eigen::Matrix3Xd* result = new Eigen::Matrix3Xd(mat.rows,mat.cols);
	for (int i=0;i<mat.cols;i++)
		for (int j=0;j<mat.rows;j++)
			(*result)(j,i) = mat.at<double>(j,i);
	return result;
}

cv::Mat toMat(Eigen::Matrix3Xd emat){
	double* p = new double[emat.rows()*emat.cols()];
	int w = emat.cols();
	for (int i=0;i<emat.rows();i++)
		for (int j=0;j<w;j++)
			p[i*w+j] = emat(i,j);
	cv::Mat result(emat.rows(),emat.cols(),CV_64F,p);
	return result;
}

void qr(cv::Mat input, cv::Mat &q, cv::Mat &r){
	Eigen::Matrix3Xd* A = toMatrix3Xd(input);
	Eigen::HouseholderQR<Eigen::MatrixXd> qr(*A);
	Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
	Eigen::MatrixXd Q = qr.householderQ();

	q = toMat(Q);
	r = toMat(R);
}

int splittext(char* str, char** pos){
	int count = 0;
	char * pch;
	pch = strtok (str," ,\n");
	while (pch != NULL)
	{
		pos[count] =  pch;
		count++;
		pch = strtok (NULL, " ,\n");
	}
	return count;
}

cv::Vec3b avSubMatValue8UC3( const CvPoint2D64f* pt, const cv::Mat* mat ){
	cv::Vec3b color(0,0,0);
	int floorx = (int)floor( pt->x );
	int floory = (int)floor( pt->y );

	if( floorx < 0 || floorx >= (*mat).cols - 1 || 
		floory < 0 || floory >= (*mat).rows - 1 )
		return cv::Vec3b(0,0,0);

	double px = pt->x - floorx;
	double py = pt->y - floory;

	cv::Vec3b tl = (*mat).at<cv::Vec3b>(floory,   floorx  );
	cv::Vec3b tr = (*mat).at<cv::Vec3b>(floory,   floorx+1);
	cv::Vec3b bl = (*mat).at<cv::Vec3b>(floory+1, floorx);
	cv::Vec3b br = (*mat).at<cv::Vec3b>(floory+1, floorx+1);

	color[0] = floor(tl[0]*(1-px)*(1-py) + tr[0]*px*(1-py) + bl[0]*(1-px)*py + br[0]*px*py+0.5);
	color[1] = floor(tl[1]*(1-px)*(1-py) + tr[1]*px*(1-py) + bl[1]*(1-px)*py + br[1]*px*py+0.5);
	color[2] = floor(tl[2]*(1-px)*(1-py) + tr[2]*px*(1-py) + bl[2]*(1-px)*py + br[2]*px*py+0.5);

	return color;
}

cv::Vec3d avSubMatValue8UC3_2( const CvPoint2D64f* pt, const cv::Mat* mat ){
	cv::Vec3d color(0,0,0);
	int floorx = (int)floor( pt->x );
	int floory = (int)floor( pt->y );

	if( floorx < 0 || floorx >= (*mat).cols - 1 || 
		floory < 0 || floory >= (*mat).rows - 1 )
		return cv::Vec3d(0,0,0);

	double px = pt->x - floorx;
	double py = pt->y - floory;

	cv::Vec3b tl = (*mat).at<cv::Vec3b>(floory,   floorx  );
	cv::Vec3b tr = (*mat).at<cv::Vec3b>(floory,   floorx+1);
	cv::Vec3b bl = (*mat).at<cv::Vec3b>(floory+1, floorx);
	cv::Vec3b br = (*mat).at<cv::Vec3b>(floory+1, floorx+1);

	color[0] = tl[0]*(1-px)*(1-py) + tr[0]*px*(1-py) + bl[0]*(1-px)*py + br[0]*px*py;
	color[1] = tl[1]*(1-px)*(1-py) + tr[1]*px*(1-py) + bl[1]*(1-px)*py + br[1]*px*py;
	color[2] = tl[2]*(1-px)*(1-py) + tr[2]*px*(1-py) + bl[2]*(1-px)*py + br[2]*px*py;

	return color;
}

double avSubPixelValue64F( const CvPoint2D64f* pt, const IplImage* img )
{
	int floorx = (int)floor( pt->x );
	int floory = (int)floor( pt->y );

	if( floorx < 0 || floorx >= img->width-1 || 
		floory < 0 || floory >= img->height-1 )
		return 0;

	double px = pt->x - floorx;
	double py = pt->y - floory;

	const int widthStep = img->widthStep/sizeof(double);
	double* pImg = (double*)( img->imageData ) + floory*widthStep + floorx;

	double tl = *( pImg );
	double tr = *( pImg+1 );
	double bl = *( pImg+widthStep );
	double br = *( pImg+widthStep+1 );

	assert( tl == cvGetReal2D( img, floory, floorx ) );

	return tl*(1-px)*(1-py) + tr*px*(1-py) + bl*(1-px)*py + br*px*py;
}

double avSubPixelValue32F( const CvPoint2D64f* pt, const IplImage* img )
{
	int floorx = (int)floor( pt->x );
	int floory = (int)floor( pt->y );

	if( floorx < 0 || floorx >= img->width-1 || 
		floory < 0 || floory >= img->height-1 )
		return 0;

	double px = pt->x - floorx;
	double py = pt->y - floory;

	const int widthStep = img->widthStep/sizeof(float);
	float* pImg = (float*)( img->imageData ) + floory*widthStep + floorx;

	float tl = *( pImg );
	float tr = *( pImg+1 );
	float bl = *( pImg+widthStep );
	float br = *( pImg+widthStep+1 );

	assert( tl == cvGetReal2D( img, floory, floorx ) );

	return (double)( tl*(1-px)*(1-py) + tr*px*(1-py) + bl*(1-px)*py + br*px*py );
}

double avSubPixelValue8U( const CvPoint2D64f* pt, const IplImage* img )
{
	int floorx = (int)floor( pt->x );
	int floory = (int)floor( pt->y );

	if( floorx < 0 || floorx >= img->width-1 || 
		floory < 0 || floory >= img->height-1 )
		return 0;


	double px = pt->x - floorx;
	double py = pt->y - floory;

	unsigned char* pImg = (unsigned char*)( img->imageData + floory*img->widthStep + floorx );

	unsigned char tl = *( pImg );
	unsigned char tr = *( pImg+1 );
	unsigned char bl = *( pImg+img->widthStep );
	unsigned char br = *( pImg+img->widthStep+1 );

	assert( (int)tl == cvGetReal2D( img, floory, floorx ) );

	return (double)( tl*(1-px)*(1-py) + tr*px*(1-py) + bl*(1-px)*py + br*px*py );
}

double avSubMatValue64F( const CvPoint2D64f* pt, const cv::Mat* mat )
{
	int floorx = (int)floor( pt->x );
	int floory = (int)floor( pt->y );

	if( floorx < 0 || floorx >= (*mat).cols - 1 || 
		floory < 0 || floory >= (*mat).rows - 1 )
		return 0;

	double px = pt->x - floorx;
	double py = pt->y - floory;

	double tl = (*mat).at<double>(floory,   floorx  );
	double tr = (*mat).at<double>(floory,   floorx+1);
	double bl = (*mat).at<double>(floory+1, floorx);
	double br = (*mat).at<double>(floory+1, floorx+1);

	return tl*(1-px)*(1-py) + tr*px*(1-py) + bl*(1-px)*py + br*px*py;
}

double avSubMatValue32F( const CvPoint2D64f* pt, const cv::Mat* mat )
{
	int floorx = (int)floor( pt->x );
	int floory = (int)floor( pt->y );

	if( floorx < 0 || floorx >= (*mat).cols - 1 || 
		floory < 0 || floory >= (*mat).rows - 1 )
		return 0;

	double px = pt->x - floorx;
	double py = pt->y - floory;

	double tl = (*mat).at<float>(floory,   floorx  );
	double tr = (*mat).at<float>(floory,   floorx+1);
	double bl = (*mat).at<float>(floory+1, floorx);
	double br = (*mat).at<float>(floory+1, floorx+1);

	return tl*(1-px)*(1-py) + tr*px*(1-py) + bl*(1-px)*py + br*px*py;
}

double avSubMatValue8U( const CvPoint2D64f* pt, const cv::Mat* mat )
{
	int floorx = (int)floor( pt->x );
	int floory = (int)floor( pt->y );

	if( floorx < 0 || floorx >= (*mat).cols - 1 || 
		floory < 0 || floory >= (*mat).rows - 1 )
		return 0;

	double px = pt->x - floorx;
	double py = pt->y - floory;

	double tl = (*mat).at<unsigned char>(floory,   floorx  );
	double tr = (*mat).at<unsigned char>(floory,   floorx+1);
	double bl = (*mat).at<unsigned char>(floory+1, floorx);
	double br = (*mat).at<unsigned char>(floory+1, floorx+1);

	return tl*(1-px)*(1-py) + tr*px*(1-py) + bl*(1-px)*py + br*px*py;
}

void write_ply4(char* outname, cv::Mat mat_Depth,std::vector<cv::Vec3b> colors,std::vector<cv::Vec3i> faces){
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << mat_Depth.cols << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "property uchar red\n";
	ply2 << "property uchar green\n";
	ply2 << "property uchar blue\n";
	ply2 << "element face " << faces.size() << std::endl;
	ply2 << "property list uchar int vertex_indices\n";
	ply2 << "end_header\n";
	for( int i = 0; i < mat_Depth.cols ; i++ )
	{
		ply2 << mat_Depth.at<double>(0,i) << " " << mat_Depth.at<double>(1,i) << " " << mat_Depth.at<double>(2,i) << " " 
			<< (int)colors[i](2) << " " << (int)colors[i](1) << " " << (int)colors[i](0) <<  std::endl;
	}
	for( int i = 0; i < faces.size() ; i++ )
	{
		ply2 << "3 " << faces[i][0] << " " << faces[i][1] << " " << faces[i][2] << " " <<  std::endl;
	}
	ply2.close();
}

//void write_ply(char* outname, bool* visible, cv::Mat mat_Depth,std::vector<cv::Vec3b> colors,cv::Mat faces){
//	std::vector<int> vindices;
//	std::vector<int> vindices_inv;
//	std::vector<int> findices;
//	for (int i=0;i<colors.size();i++) {
//		if (visible[i]) {
//			vindices_inv.push_back(vindices.size());
//			vindices.push_back(i);
//		}
//		else
//			vindices_inv.push_back(-1);
//	}
//	//printf("vindices_inv %d\n",vindices_inv.size());
//	for (int i=0;i<faces.cols;i++) {
//		bool vis = true;
//		for (int j=0;j<3;j++)
//			if (!visible[faces.at<unsigned int>(j,i)]) {
//				vis = false; break;
//			}
//		if (vis) findices.push_back(i);
//	}
//
//	std::ofstream ply2( outname );
//	ply2 << "ply\n";
//	ply2 << "format ascii 1.0\n";
//	ply2 << "element vertex " << vindices.size() << std::endl;
//	ply2 << "property float x\n";
//	ply2 << "property float y\n";
//	ply2 << "property float z\n";
//	ply2 << "property uchar red\n";
//	ply2 << "property uchar green\n";
//	ply2 << "property uchar blue\n";
//	ply2 << "element face " << findices.size() << std::endl;
//	ply2 << "property list uchar int vertex_indices\n";
//	ply2 << "end_header\n";
//	for( int i = 0; i < vindices.size() ; i++ )
//	{
//		int ind = vindices[i];
//		ply2 << mat_Depth.at<double>(0,ind) << " " << mat_Depth.at<double>(1,ind) << " " << mat_Depth.at<double>(2,ind) << " " 
//			<< (int)colors[ind](2) << " " << (int)colors[ind](1) << " " << (int)colors[ind](0) <<  std::endl;
//	}
//	for( int i = 0; i < findices.size() ; i++ )
//	{
//		ply2 << "3 " << vindices_inv[faces.at<unsigned int>(0,findices[i])] << " " << vindices_inv[faces.at<unsigned int>(1,findices[i])] << " " << vindices_inv[faces.at<unsigned int>(2,findices[i])] << " " <<  std::endl;
//	}
//	ply2.close();
//}


void write_plyF(char* outname, cv::Mat mat_Depth,std::vector<cv::Vec3b> colors,int nFaces, unsigned* faces){
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << colors.size() << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "property uchar red\n";
	ply2 << "property uchar green\n";
	ply2 << "property uchar blue\n";
	ply2 << "element face " << nFaces << std::endl;
	ply2 << "property list uchar int vertex_indices\n";
	ply2 << "end_header\n";
	for( int i = 0; i < colors.size() ; i++ )
	{
		ply2 << mat_Depth.at<double>(0,i) << " " << mat_Depth.at<double>(1,i) << " " << mat_Depth.at<double>(2,i) << " " 
			<< (int)colors[i](2) << " " << (int)colors[i](1) << " " << (int)colors[i](0) <<  std::endl;
	}
	for( int i = 0; i < nFaces ; i++ )
	{
		ply2 << "3 " << faces[3*i] << " " << faces[3*i+1] << " " << faces[3*i+2] << " " <<  std::endl;
	}
	ply2.close();
}

void write_ply(char* outname, int count, bool* visible, float* points_){
	std::vector<int> inds;
	for (int i=0;i<count;i++){
		if (visible[i])
			inds.push_back(i);
	}
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << inds.size() << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "end_header\n";
	for( int i = 0; i < inds.size() ; i++ )
	{
		int ind = inds[i];
		ply2 << points_[3*ind] << " " << points_[3*ind+1] << " " << points_[3*ind+2] <<  std::endl;
	}
	ply2.close();
}

void write_ply(char* outname, int count, float* points_){
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << count << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "end_header\n";
	for( int i = 0; i < count ; i++ )
	{
		ply2 << points_[3*i] << " " << points_[3*i+1] << " " << points_[3*i+2] <<  std::endl;
	}
	ply2.close();
}

void write_ply(char* outname, cv::Mat mat_Depth,std::vector<cv::Vec3b> colors,cv::Mat faces){
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << mat_Depth.cols << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "property uchar red\n";
	ply2 << "property uchar green\n";
	ply2 << "property uchar blue\n";
	ply2 << "element face " << faces.cols << std::endl;
	ply2 << "property list uchar int vertex_indices\n";
	ply2 << "end_header\n";
	for( int i = 0; i < mat_Depth.cols ; i++ )
	{
		ply2 << mat_Depth.at<double>(0,i) << " " << mat_Depth.at<double>(1,i) << " " << mat_Depth.at<double>(2,i) << " " 
			<< (int)colors[i](2) << " " << (int)colors[i](1) << " " << (int)colors[i](0) <<  std::endl;
	}
	for( int i = 0; i < faces.cols ; i++ )
	{
		ply2 << "3 " << faces.at<int>(0,i) << " " << faces.at<int>(1,i) << " " << faces.at<int>(2,i) << " " <<  std::endl;
	}
	ply2.close();
}

void write_ply(char* outname, Eigen::Matrix3Xd* mesh){
		std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << mesh->cols() << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "end_header\n";
	for( int i = 0; i < mesh->cols()  ; i++ )
	{
		ply2 <<(*mesh)(0,i) << " " << (*mesh)(1,i) << " " << (*mesh)(2,i) <<  std::endl;
	}
	ply2.close();
}

void write_plyFloat(char* outname, cv::Mat mat_Depth, cv::Mat mat_Color, cv::Mat mat_Faces){
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << mat_Depth.rows << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "property uchar red\n";
	ply2 << "property uchar green\n";
	ply2 << "property uchar blue\n";
	ply2 << "element face " << mat_Faces.rows << std::endl;
	ply2 << "property list uchar int vertex_indices\n";
	ply2 << "end_header\n";
	float colors[3];
	for( int i = 0; i < mat_Depth.rows ; i++ )
	{
		for (int j=0;j<3;j++) {
			colors[j] = mat_Color.at<float>(i,j);
			colors[j] = (colors[j]<0)?0:colors[j];
			colors[j] = (colors[j]>255)?255:colors[j];
		}
		ply2 << mat_Depth.at<float>(i,0) << " " << mat_Depth.at<float>(i,1) << " " << mat_Depth.at<float>(i,2) << " "
			<<  (int)colors[0] << " " << (int)colors[1] << " " << (int)colors[2] << std::endl;
	}
	for( int i = 0; i < mat_Faces.rows ; i++ )
	{
		ply2 << "3 " << mat_Faces.at<int>(i,0) << " " << mat_Faces.at<int>(i,1) << " " << mat_Faces.at<int>(i,2) << " " <<  std::endl;
	}
	ply2.close();
}

cv::Mat skew(cv::Mat v1){
	cv::Mat out(3,3,CV_32F);
	out.at<float>(0,0) = out.at<float>(1,1) = out.at<float>(2,2) = 0;
	out.at<float>(0,1) = -v1.at<float>(2,0);
	out.at<float>(0,2) = v1.at<float>(1,0);
	out.at<float>(1,0) = v1.at<float>(2,0);
	out.at<float>(1,2) = -v1.at<float>(0,0);
	out.at<float>(2,0) = -v1.at<float>(1,0);
	out.at<float>(2,1) = v1.at<float>(0,0);
	return out;
}

void groundScale(cv::Mat input, cv::Mat &output, float bgThresh, float gapPc) {
	cv::Mat mask = (input < bgThresh) & (input > 1 - bgThresh);
	double mn, mx;
	cv::minMaxLoc(input,&mn,&mx,0,0,mask);
	printf("mn, mx: %f %f\n",mn,mx);
	double range = mx - mn;
	mn = mn - gapPc * range;
	mn = (mn > 0)?mn:0;
	mx = mx + gapPc * range;
	mx = (mx < 1)?mx:1;
	range = mx - mn;

	cv::Mat mask1 = (input >= bgThresh)/255;
	mask = mask/255;
	mask.convertTo(mask,input.type());
	mask1.convertTo(mask1,input.type());
	cv::Mat mask2 = 1 - mask - mask1;
	output = mask.mul((input - mn)/range) + mask1 * 1/* + mask2 * mn*/;
	//cv::imshow("out",output); cv::waitKey();
	//output = output*255;
	//output.convertTo(output,CV_8U);
}

cv::Mat findRotation(cv::Mat v1, cv::Mat v2){
	cv::Mat ab = v1.cross(v2);
	//std::cout << "cross " << ab << std::endl;
	float s = sqrt(ab.at<float>(0,0)*ab.at<float>(0,0) + ab.at<float>(1,0)*ab.at<float>(1,0) + ab.at<float>(2,0)*ab.at<float>(2,0));
	if (s == 0)
		return cv::Mat::eye(3,3,CV_32F);

	float c = v1.at<float>(0,0)*v2.at<float>(0,0) + v1.at<float>(1,0)*v2.at<float>(1,0) + v1.at<float>(2,0)*v2.at<float>(2,0);
	cv::Mat sk = skew(ab);
	return cv::Mat::eye(3,3,CV_32F) + sk + sk*sk*(1-c)/(s*s);
}

/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "RenderModel.h"
#include <opencv2/calib3d.hpp>

bool RenderServices::estimateFaceNormals(cv::Mat shape, cv::Mat faces, cv::Mat &normals){
	if (normals.cols == 0) normals = cv::Mat::zeros(faces.rows,shape.cols,shape.type());
	float nx, ny, nz;
	for (int i=0;i<faces.rows;i++){
		triangleNormal(shape,faces,i,nx, ny, nz);
		normals.at<float>(i,0) = nx;
		normals.at<float>(i,1) = ny;
		normals.at<float>(i,2) = nz;
	}
	return true;
}
bool RenderServices::estimateFaceNormals(cv::Mat shape, cv::Mat faces, cv::Mat &normals, cv::Mat &centers, cv::Mat &areas){
	if (normals.cols == 0) normals = cv::Mat::zeros(faces.rows,shape.cols,shape.type());
	if (centers.cols == 0) centers = cv::Mat::zeros(faces.rows,shape.cols,shape.type());
	if (areas.cols == 0) areas = cv::Mat::zeros(faces.rows,1,shape.type());
	float nx, ny, nz, S;
	for (int i=0;i<faces.rows;i++){
		S = triangleNormal(shape,faces,i,nx, ny, nz);
		normals.at<float>(i,0) = nx;
		normals.at<float>(i,1) = ny;
		normals.at<float>(i,2) = nz;
		areas.at<float>(i,0) = S;

		for (int j=0;j<3;j++){
			centers.at<float>(i,j) = 0;
			for (int k=0;k<3;k++){
				int ind = faces.at<int>(i, k);
				centers.at<float>(i,j) += shape.at<float>(ind,j);
			}
			centers.at<float>(i,j) /= 3.0f;
		}
	}
	return true;
}

bool RenderServices::estimateVertexNormals(cv::Mat shape, cv::Mat faces, cv::Mat &normals){
	if (normals.cols != 0) normals.release();
	normals = cv::Mat::zeros(shape.rows,shape.cols,shape.type());
	float nx, ny, nz;
	for (int i=0;i<faces.rows;i++){
		for (int j=0;j<3;j++) {
			triangleNormalFromVertex(shape, faces, i, j, nx, ny, nz);
			int ind = faces.at<int>(i, j);
			normals.at<float>(ind,0) += nx;
			normals.at<float>(ind,1) += ny;
			normals.at<float>(ind,2) += nz;
		}
	}
	
	for (int i=0;i<shape.rows;i++){
		float no = sqrt(normals.at<float>(i,0)*normals.at<float>(i,0)+normals.at<float>(i,1)*normals.at<float>(i,1)+normals.at<float>(i,2)*normals.at<float>(i,2));
		for (int j=0;j<3;j++) normals.at<float>(i,j) /= no;
	}
	return true;
}

bool RenderServices::estimateColor(cv::Mat shape, cv::Mat tex, cv::Mat faces, bool* visible, bool* noShadow, float* render_model, cv::Mat &colors){
	cv::Mat normals;
	return estimateColor(shape, tex, faces, visible, noShadow, render_model, colors, normals);
}
bool RenderServices::estimateColor(cv::Mat shape, cv::Mat tex, cv::Mat faces, bool* visible, bool* noShadow, float* render_model, cv::Mat &colors, cv::Mat &normals){
	estimateVertexNormals(shape, faces, normals);
	if (colors.cols != 0) colors.release();
	colors = cv::Mat::zeros(shape.rows,shape.cols,shape.type());

	float tmpTex[3], tmpColor[3], tmpNormal[3],  tmpR[3], tmpL[3];
	float cc = render_model[RENDER_PARAMS_CONTRAST];
	float r_[3];
	memcpy(r_,render_model+RENDER_PARAMS_R,3*sizeof(float));
	cv::Mat vecR(3,1,CV_32F,r_);
	cv::Mat matR;
	cv::Rodrigues(vecR,matR);
	
	tmpL[0] = cos(render_model[RENDER_PARAMS_LDIR])*sin(render_model[RENDER_PARAMS_LDIR+1]);
	tmpL[1] = sin(render_model[RENDER_PARAMS_LDIR]);
	tmpL[2] = cos(render_model[RENDER_PARAMS_LDIR])*cos(render_model[RENDER_PARAMS_LDIR+1]);
		
	cv::Mat newNormals = matR * normals.t();
	for (int i=0;i<shape.rows;i++){
		if (visible[i]){
			for (int j=0;j<3;j++) {
				tmpTex[j] = tex.at<float>(i,j);
				tmpColor[j] = render_model[RENDER_PARAMS_AMBIENT+j]*tmpTex[j];
				tmpNormal[j] = newNormals.at<float>(j,i);
			} 
			
			if (noShadow[i]) {
				float nl = tmpNormal[0]*tmpL[0] + tmpNormal[1]*tmpL[1] + tmpNormal[2]*tmpL[2];
				float ne = tmpNormal[2];
				if (nl * ne >= 0) {
					if (nl < 0 || ne < 0) {
						nl = -nl;
						ne = -ne;
						for (int k=0;k<3;k++) tmpNormal[k] = -tmpNormal[k];
					}
					
					for (int j=0;j<3;j++) {
						tmpR[j] = 2*nl*tmpNormal[j] - tmpL[j];
						tmpColor[j] += nl*render_model[RENDER_PARAMS_DIFFUSE+j]*tmpTex[j];
					} 
					
					float nR = sqrt(tmpR[0]*tmpR[0] + tmpR[1]*tmpR[1] + tmpR[2]*tmpR[2]);
					float re = tmpR[2]/nR;
					if (re > 0) {
						re = pow(re,RENDER_PARAMS_SHINENESS_DEFAULT);
						for (int j=0;j<3;j++) {
							tmpColor[j] += re*render_model[RENDER_PARAMS_DIFFUSE+j]*RENDER_PARAMS_SPECULAR_DEFAULT;
							//tmpColor[j] += re*render_model[RENDER_PARAMS_DIFFUSE+j]*tmpTex[j];
						} 
					}
					
					for (int j=0;j<3;j++) {
						if (tmpColor[j] < 0) tmpColor[j] = 0;
						if (tmpColor[j] > 255) tmpColor[j] = 255;
					}
				}
			}

			float gray = 0.3*tmpColor[0] + 0.59*tmpColor[1] + 0.11*tmpColor[2];
			for (int j=0;j<3;j++)
				colors.at<float>(i,j) = render_model[RENDER_PARAMS_GAIN+j] * (cc*tmpColor[j]+(1-cc)*gray) + render_model[RENDER_PARAMS_OFFSET+j];
		}
	}
	return true;
}

bool RenderServices::estimatePointColor(cv::Mat &points, cv::Mat &tex, cv::Mat &normals, std::vector<int> &inds, cv::Mat &noShadow, float* render_model, cv::Mat &colors){
	if (colors.cols != 0) colors.release();
	colors = cv::Mat::zeros(points.rows,points.cols,points.type());

	float tmpTex[3], tmpColor[3], tmpNormal[3],  tmpR[3], tmpL[3];
	float cc = render_model[RENDER_PARAMS_CONTRAST];
	float r_[3];
	memcpy(r_,render_model+RENDER_PARAMS_R,3*sizeof(float));
	cv::Mat vecR(3,1,CV_32F,r_);
	cv::Mat matR;
	cv::Rodrigues(vecR,matR);

	tmpL[0] = cos(render_model[RENDER_PARAMS_LDIR])*sin(render_model[RENDER_PARAMS_LDIR+1]);
	tmpL[1] = sin(render_model[RENDER_PARAMS_LDIR]);
	tmpL[2] = cos(render_model[RENDER_PARAMS_LDIR])*cos(render_model[RENDER_PARAMS_LDIR+1]);

	//printf("L: %f %f %f\n",tmpL[0],tmpL[1],tmpL[2]);
	cv::Mat newNormals = matR * normals.t();
	//printf("newNormals %d %d %d\n",newNormals.rows,newNormals.cols,newNormals.type());
	//printf("newNormals done\n");
	for (int i=0;i<points.rows;i++){
		for (int j=0;j<3;j++) {
			tmpTex[j] = tex.at<float>(i,j);
			tmpColor[j] = render_model[RENDER_PARAMS_AMBIENT+j]*tmpTex[j];
			tmpNormal[j] = newNormals.at<float>(j,i);
		} 
		int ind = inds[i];
		if (noShadow.at<unsigned int>(ind,0)) {
			float nl = tmpNormal[0]*tmpL[0] + tmpNormal[1]*tmpL[1] + tmpNormal[2]*tmpL[2];
			float ne = tmpNormal[2];
			if (nl * ne >= 0) {
				if (nl < 0 || ne < 0) {
					nl = -nl;
					ne = -ne;
					for (int k=0;k<3;k++) tmpNormal[k] = -tmpNormal[k];
				}

				for (int j=0;j<3;j++) {
					tmpR[j] = 2*nl*tmpNormal[j] - tmpL[j];
					tmpColor[j] += nl*render_model[RENDER_PARAMS_DIFFUSE+j]*tmpTex[j];
				} 

				float nR = sqrt(tmpR[0]*tmpR[0] + tmpR[1]*tmpR[1] + tmpR[2]*tmpR[2]);
				float re = tmpR[2]/nR;
				if (re > 0) {
					re = pow(re,RENDER_PARAMS_SHINENESS_DEFAULT);
					for (int j=0;j<3;j++) {
						tmpColor[j] += re*render_model[RENDER_PARAMS_DIFFUSE+j]*RENDER_PARAMS_SPECULAR_DEFAULT;
						//tmpColor[j] += re*render_model[RENDER_PARAMS_DIFFUSE+j]*tmpTex[j];
					} 
				}

				for (int j=0;j<3;j++) {
					if (tmpColor[j] < 0) tmpColor[j] = 0;
					if (tmpColor[j] > 255) tmpColor[j] = 255;
				}
			}

		}
		
		float gray = 0.3*tmpColor[0] + 0.59*tmpColor[1] + 0.11*tmpColor[2];
		for (int j=0;j<3;j++)
			colors.at<float>(i,j) = render_model[RENDER_PARAMS_GAIN+j] * (cc*tmpColor[j]+(1-cc)*gray) + render_model[RENDER_PARAMS_OFFSET+j];
	}
	newNormals.release();
	//printf("colors done\n");
	return true;
}

bool RenderServices::triangleNormalFromVertex(cv::Mat shape, cv::Mat faces, int face_id, int vertex_id, float &nx, float &ny, float &nz) {
	int ind0 = faces.at<int>(face_id, vertex_id);
	int ind1 = faces.at<int>(face_id, (vertex_id+1)%3);
	int ind2 = faces.at<int>(face_id, (vertex_id+2)%3);

	float a[3],b[3],v[3];
	for (int j=0;j<3;j++){
		a[j] = shape.at<float>(ind1,j) - shape.at<float>(ind0,j);
		b[j] = shape.at<float>(ind2,j) - shape.at<float>(ind0,j);
	}
	v[0] = a[1]*b[2] - a[2]*b[1];
	v[1] = a[2]*b[0] - a[0]*b[2];
	v[2] = a[0]*b[1] - a[1]*b[0];
	float no = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
	float dp = a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
	float la = sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
	float lb = sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
	float alpha = acos(dp/(la*lb));

	nx = alpha * v[0]/no;
	ny = alpha * v[1]/no;
	nz = alpha * v[2]/no;
	return true;
}

float RenderServices::triangleNormal(cv::Mat shape, cv::Mat faces, int face_id, float &nx, float &ny, float &nz) {
	int vertex_id = 0;
	int ind0 = faces.at<int>(face_id, vertex_id);
	int ind1 = faces.at<int>(face_id, (vertex_id+1)%3);
	int ind2 = faces.at<int>(face_id, (vertex_id+2)%3);

	float a[3],b[3],v[3];
	for (int j=0;j<3;j++){
		a[j] = shape.at<float>(ind1,j) - shape.at<float>(ind0,j);
		b[j] = shape.at<float>(ind2+j) - shape.at<float>(ind0,j);
	}
	v[0] = a[1]*b[2] - a[2]*b[1];
	v[1] = a[2]*b[0] - a[0]*b[2];
	v[2] = a[0]*b[1] - a[1]*b[0];
	float no = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
	nx = v[0]/no;
	ny = v[1]/no;
	nz = v[2]/no;
	return no*no/2;
}

bool RenderServices::estimatePointColor(cv::Mat &points, cv::Mat &tex, cv::Mat &normals, std::vector<int> &inds, cv::Mat &visible, cv::Mat &noShadow, float* render_model, cv::Mat &colors){
	if (colors.cols != 0) colors.release();
	colors = cv::Mat::zeros(points.rows,points.cols,points.type());

	float tmpTex[3], tmpColor[3], tmpNormal[3],  tmpR[3], tmpL[3];
	float cc = render_model[RENDER_PARAMS_CONTRAST];
	float r_[3];
	memcpy(r_,render_model+RENDER_PARAMS_R,3*sizeof(float));
	cv::Mat vecR(3,1,CV_32F,r_);
	cv::Mat matR;
	cv::Rodrigues(vecR,matR);

	tmpL[0] = cos(render_model[RENDER_PARAMS_LDIR])*sin(render_model[RENDER_PARAMS_LDIR+1]);
	tmpL[1] = sin(render_model[RENDER_PARAMS_LDIR]);
	tmpL[2] = cos(render_model[RENDER_PARAMS_LDIR])*cos(render_model[RENDER_PARAMS_LDIR+1]);

	//printf("L: %f %f %f\n",tmpL[0],tmpL[1],tmpL[2]);
	cv::Mat newNormals = matR * normals.t();
	//printf("newNormals %d %d %d\n",newNormals.rows,newNormals.cols,newNormals.type());
	//printf("newNormals done\n");
	for (int i=0;i<points.rows;i++){
		int ind = inds[i];
		if (visible.at<unsigned int>(ind,0) == 0){
			colors.at<float>(i,0) = colors.at<float>(i,1) = colors.at<float>(i,2) = 0;
			continue;
		}
		for (int j=0;j<3;j++) {
			tmpTex[j] = tex.at<float>(i,j);
			tmpColor[j] = render_model[RENDER_PARAMS_AMBIENT+j]*tmpTex[j];
			tmpNormal[j] = newNormals.at<float>(j,i);
		} 
		if (noShadow.at<unsigned int>(ind,0)) {
			float nl = tmpNormal[0]*tmpL[0] + tmpNormal[1]*tmpL[1] + tmpNormal[2]*tmpL[2];
			float ne = tmpNormal[2];
			if (nl * ne >= 0) {
				if (nl < 0 || ne < 0) {
					nl = -nl;
					ne = -ne;
					for (int k=0;k<3;k++) tmpNormal[k] = -tmpNormal[k];
				}

				for (int j=0;j<3;j++) {
					tmpR[j] = 2*nl*tmpNormal[j] - tmpL[j];
					tmpColor[j] += nl*render_model[RENDER_PARAMS_DIFFUSE+j]*tmpTex[j];
				} 

				float nR = sqrt(tmpR[0]*tmpR[0] + tmpR[1]*tmpR[1] + tmpR[2]*tmpR[2]);
				float re = tmpR[2]/nR;
				if (re > 0) {
					re = pow(re,RENDER_PARAMS_SHINENESS_DEFAULT);
					for (int j=0;j<3;j++) {
						tmpColor[j] += re*render_model[RENDER_PARAMS_DIFFUSE+j]*RENDER_PARAMS_SPECULAR_DEFAULT;
						//tmpColor[j] += re*render_model[RENDER_PARAMS_DIFFUSE+j]*tmpTex[j];
					} 
				}

				for (int j=0;j<3;j++) {
					if (tmpColor[j] < 0) tmpColor[j] = 0;
					if (tmpColor[j] > 255) tmpColor[j] = 255;
				}
			}

		}
		
		float gray = 0.3*tmpColor[0] + 0.59*tmpColor[1] + 0.11*tmpColor[2];
		for (int j=0;j<3;j++)
			colors.at<float>(i,j) = render_model[RENDER_PARAMS_GAIN+j] * (cc*tmpColor[j]+(1-cc)*gray) + render_model[RENDER_PARAMS_OFFSET+j];
	}
	newNormals.release();
	//printf("colors done\n");
	return true;
}

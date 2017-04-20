/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cfloat>
#include <Eigen/Dense>
#include "utility.h"
//#include <pcl/common/common_headers.h>
//#include <pcl/surface/gp3.h>

#define MESH_COLOR	1
#define MESH_NORMAL	2

#define PROP_X		0
#define PROP_Y		1
#define PROP_Z		2
#define PROP_R		3
#define PROP_G		4
#define PROP_B		5
#define PROP_NX		6
#define PROP_NY		7
#define PROP_NZ		8

using Eigen::Matrix3Xd;

class MeshModel
{
public:
	int nVertices;
	int nFaces;
	int type;

	float* vertices_;
	int* faces_;
	unsigned char* colors_;
	float* normals_;

	MeshModel(char* ply_file);
	MeshModel(Matrix3Xd m);
	MeshModel();
	//MeshModel(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);
	//MeshModel(pcl::PointCloud<pcl::PointNormal>::Ptr cloud);
	bool save2File(char* ply_file );

	//void toPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr);
	Matrix3Xd* toMatrix3Xd();
	//void toPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr);
	//void copyNormals(pcl::PointCloud<pcl::Normal>::Ptr normals);
	//void copyMesh(pcl::PolygonMesh mesh);
	//pcl::PointXYZ centerPoint();
	void translate(double tx, double ty, double tz);
	~MeshModel(void);
};

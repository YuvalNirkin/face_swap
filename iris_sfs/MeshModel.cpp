/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "MeshModel.h"
//#include <boost/thread/thread.hpp>

MeshModel::MeshModel(){
	nVertices = nFaces = type = -1;
	vertices_ = 0;
	faces_ = 0;
	colors_ = 0;
	normals_ = 0;
	int state = 0;
}

//MeshModel::MeshModel(pcl::PointCloud<pcl::PointNormal>::Ptr cloud){
//	faces_ = 0;
//	nFaces = -1;
//	nVertices = cloud->points.size();
//	type = MESH_NORMAL;
//	vertices_ = new float[3*nVertices];
//	normals_ = new float[3*nVertices];
//	for (int i=0;i<nVertices;i++){
//		normals_[3*i] = -cloud->points[i].normal_x;
//		normals_[3*i+1] = -cloud->points[i].normal_y;
//		normals_[3*i+2] = -cloud->points[i].normal_z;
//		vertices_[3*i] = cloud->points[i].x;
//		vertices_[3*i+1] = cloud->points[i].y;
//		vertices_[3*i+2] = cloud->points[i].z;
//	}
//}

//MeshModel::MeshModel(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud){
//	faces_ = 0;
//	nFaces = -1;
//	nVertices = cloud->points.size();
//	type = MESH_COLOR | MESH_NORMAL;
//	vertices_ = new float[3*nVertices];
//	normals_ = new float[3*nVertices];
//	colors_ = new unsigned char[3*nVertices];
//	for (int i=0;i<nVertices;i++){
//		normals_[3*i] = -cloud->points[i].normal_x;
//		normals_[3*i+1] = -cloud->points[i].normal_y;
//		normals_[3*i+2] = -cloud->points[i].normal_z;
//		vertices_[3*i] = cloud->points[i].x;
//		vertices_[3*i+1] = cloud->points[i].y;
//		vertices_[3*i+2] = cloud->points[i].z;
//		boost::uint32_t rgb = *reinterpret_cast<boost::uint32_t*>(&(cloud->points[i].rgb));
//		colors_[3*i] = (unsigned char)(rgb >> 16);
//		colors_[3*i+1] = (unsigned char)((rgb  & 0x0000FF00) >> 8);
//		colors_[3*i + 2] = (unsigned char)(rgb & 0x000000FF);
//	}
//}

MeshModel::MeshModel(char* ply_file){
	nVertices = nFaces = type = -1;
	vertices_ = 0;
	faces_ = 0;
	colors_ = 0;
	normals_ = 0;
	int state = 0;

	char str[250];
	char* pos[10];
	unsigned char prop_count = 0;
	unsigned char props[9];
	int count;
	int vcount = 0;
	int fcount = 0;
	int i;

	FILE* file = fopen(ply_file,"r");
	if (!file) return;
	while (!feof(file) && state <= 4){
		fgets(str,250,file);
		count = splittext(str, pos);
		if (count < 1 || (strcmp(pos[0],"comment") == 0)) continue;
		
		switch (state){
			case 0:								// at beginning
				if (count != 3 || (strcmp(pos[0],"element") != 0) || (strcmp(pos[1],"vertex") != 0)) continue;
				nVertices = atoi(pos[2]);
				if (nVertices < 1) {
					nVertices = -1; return;
				}
				vertices_ = new float[3*nVertices];
				state = 1;
				type = 0;
				break;
			case 1:								// get properties
				if (strcmp(pos[0],"end_header") == 0) state = 3;
				else if (strcmp(pos[0],"element") == 0){
					if (strcmp(pos[1],"face") == 0){
						state = 2;
						nFaces = atoi(pos[2]);
						faces_ = new int[3*nFaces];
					}
				}
				else if (count == 3 && (strcmp(pos[0],"property") == 0)){
					if (strcmp(pos[2],"x") == 0) {
						props[prop_count] = PROP_X;
						prop_count++;
					}
					if (strcmp(pos[2],"y") == 0){
						props[prop_count] = PROP_Y;
						prop_count++;
					}
					if (strcmp(pos[2],"z") == 0){
						props[prop_count] = PROP_Z;
						prop_count++;
					}
					if (strcmp(pos[2],"nx") == 0) {
						type += MESH_NORMAL;
						normals_ = new float[3*nVertices];
						props[prop_count] = PROP_NX;
						prop_count++;
					}
					if (strcmp(pos[2],"ny") == 0){
						props[prop_count] = PROP_NY;
						prop_count++;
					}
					if (strcmp(pos[2],"nz") == 0){
						props[prop_count] = PROP_NZ;
						prop_count++;
					}
					if (strcmp(pos[2],"red") == 0){
						type += MESH_COLOR;
						colors_ = new unsigned char[3*nVertices];
						props[prop_count] = PROP_R;
						prop_count++;
					}
					if (strcmp(pos[2],"green") == 0){
						props[prop_count] = PROP_G;
						prop_count++;
					}
					if (strcmp(pos[2],"blue") == 0){
						props[prop_count] = PROP_B;
						prop_count++;
					}
				}
				break;
			case 2:
				if (strcmp(pos[0],"end_header") == 0) state = 3;
				break;
			case 3:
				for (i = 0; i < prop_count; i++){
					switch (props[i]){
						case PROP_X:
							vertices_[vcount*3] = atof(pos[i]); break;
						case PROP_Y:
							vertices_[vcount*3+1] = atof(pos[i]); break;
						case PROP_Z:
							vertices_[vcount*3+2] = atof(pos[i]); break;
						case PROP_NX:
							normals_[vcount*3] = atof(pos[i]); break;
						case PROP_NY:
							normals_[vcount*3+1] = atof(pos[i]); break;
						case PROP_NZ:
							normals_[vcount*3+2] = atof(pos[i]); break;
						case PROP_R:
							colors_[vcount*3] = atoi(pos[i]); break;
						case PROP_G:
							colors_[vcount*3+1] = atoi(pos[i]); break;
						case PROP_B:
							colors_[vcount*3+2] = atoi(pos[i]); break;
					}
				}
				vcount++;
				if (vcount == nVertices) {
					if (nFaces > 0)
						state = 4;
					else 
						state = 5;
				}
				break;
			case 4:
				faces_[3*fcount] = atoi(pos[1]);
				faces_[3*fcount+1] = atoi(pos[2]);
				faces_[3*fcount+2] = atoi(pos[3]);
				fcount++;
				break;
		}
	}

	fclose(file);



	//std::ifstream file( fileName );
 //   if ( !file ) return invalidPLYFile();

	//std::stringstream buffer;
 //   buffer << file.rdbuf();
 //   file.close();

	//std::string &buf = buffer.str();
	//if ( buf.substr( 0, 3 ) != "ply" ) return invalidPLYFile();

	//size_t pos;
	//pos = buf.find( "element vertex" );
	//if ( pos == std::string::npos) return invalidPLYFile();
	//buffer.seekg( pos + 14 );
	//buffer >> nVertices_;
	//vertices_ = new float[3*nVertices_]; 
	//buffer.ignore( 1024, '\n' );		//potential comments



	//pos = buf.find( "element face" );
	//if ( pos == std::string::npos) return invalidPLYFile();
	//buffer.seekg( pos + 12 );
	//buffer >> mesh_.nFaces_;
	//mesh_.faces_ = new unsigned[3*mesh_.nFaces_];

	//pos = buf.find( "end_header" );
	//buffer.seekg( pos );
	//buffer.ignore( 1024, '\n' );

	////Vertices
	//for ( unsigned i=0, idx=0; i<mesh_.nVertices_; ++i )
	//{	
	//	buffer >> mesh_.vertices_[idx++];	//x
	//	mesh_.vertices_[idx-1] += MODEL_TX;
	//	mesh_.vertices_[idx-1] *= MODEL_SCALE;
	//	buffer >> mesh_.vertices_[idx++];	//y
	//	mesh_.vertices_[idx-1] += MODEL_TY;
	//	mesh_.vertices_[idx-1] *= MODEL_SCALE;
	//	buffer >> mesh_.vertices_[idx++];	//z
	//	mesh_.vertices_[idx-1] += MODEL_TZ;
	//	mesh_.vertices_[idx-1] *= MODEL_SCALE;
	//	buffer.ignore( 1024, '\n' );		//potential comments
	//}

	////Faces
	//unsigned nEdges;
	//for ( unsigned i=0, idx=0; i<mesh_.nFaces_; ++i )
	//{
	//	buffer >> nEdges;
	//	if ( nEdges != 3 ) return invalidPLYFile();
	//	buffer >> mesh_.faces_[idx++];		//v1
	//	buffer >> mesh_.faces_[idx++];		//v2
	//	buffer >> mesh_.faces_[idx++];		//v3
	//	buffer.ignore( 1024, '\n' );		//potential comments
	//}

	//return 1;
}
	
bool MeshModel::save2File(char* ply_file){
	FILE* file = fopen( ply_file, "w");
	if ( !file )
    {
		std::cerr << "Creation Error\n";
        return false;
    }

	fprintf( file, "ply\n");
	fprintf( file, "format ascii 1.0\n" );
	fprintf( file, "element vertex %d\n", nVertices );
	fprintf( file, "property float x\nproperty float y\nproperty float z\n" );
	if (type & MESH_COLOR)
		fprintf( file, "property uchar red\nproperty uchar green\nproperty uchar blue\n");
	if (type & MESH_NORMAL)
		fprintf( file, "property float nx\nproperty float ny\nproperty float nz\n");
	if (nFaces > 0){
		fprintf( file, "element face %d\n",nFaces);
		fprintf( file, "property list uchar int vertex_indices\n");
	}
	fprintf( file, "end_header\n");

	for (int i=0;i<nVertices;i++){
		fprintf(file, "%f %f %f", vertices_[3*i], vertices_[3*i+1], vertices_[3*i+2]);
		if (type & MESH_COLOR)
			fprintf(file, " %d %d %d", colors_[3*i], colors_[3*i+1], colors_[3*i+2]);
		if (type & MESH_NORMAL)
			fprintf(file, " %f %f %f", normals_[3*i], normals_[3*i+1], normals_[3*i+2]);
		fprintf( file, "\n");
	}

	if (faces_){
		for (int i=0;i<nFaces;i++){
			fprintf(file, "3 %d %d %d\n", faces_[3*i], faces_[3*i+1], faces_[3*i+2]);
		}
	}
	fclose(file);
	return true;
}
	
MeshModel::~MeshModel(void){
}

//void MeshModel::toPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr){
//	for (int i=0;i<nVertices;i++){
//		pcl::PointXYZRGB point;
//		point.x = vertices_[3*i];
//		point.y = vertices_[3*i+1];
//		point.z = vertices_[3*i+2];
//		boost::uint32_t rgb = (static_cast<boost::uint32_t>(colors_[3*i]) << 16 |
//			  static_cast<boost::uint32_t>(colors_[3*i+1]) << 8 | static_cast<boost::uint32_t>(colors_[3*i+2]));
//		point.rgb = *reinterpret_cast<float*>(&rgb);
//		cloud_ptr->points.push_back (point);
//	}
//	cloud_ptr->width = nVertices;
//	cloud_ptr->height = 1;
//}

//void MeshModel::toPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr){
//	for (int i=0;i<nVertices;i++){
//		pcl::PointXYZ point;
//		point.x = vertices_[3*i];
//		point.y = vertices_[3*i+1];
//		point.z = vertices_[3*i+2];
//		cloud_ptr->points.push_back (point);
//	}
//	cloud_ptr->width = nVertices;
//	cloud_ptr->height = 1;
//}

//void MeshModel::copyNormals(pcl::PointCloud<pcl::Normal>::Ptr normals){
//	if (normals_ != 0) delete normals_;
//	normals_ = new float[3*normals->points.size()];
//	for (int i=0;i<normals->points.size();i++){
//		normals_[3*i] = -normals->points[i].normal_x;
//		normals_[3*i+1] = -normals->points[i].normal_y;
//		normals_[3*i+2] = -normals->points[i].normal_z;
//	}
//	type = type | MESH_NORMAL;
//}

//void MeshModel::copyMesh(pcl::PolygonMesh mesh){
//	if (faces_ != 0) delete faces_;
//	faces_ = new int[3*mesh.polygons.size()];
//	nFaces = mesh.polygons.size();
//	for (int i=0;i<mesh.polygons.size();i++){
//		faces_[3*i] = mesh.polygons[i].vertices[0];
//		faces_[3*i+1] = mesh.polygons[i].vertices[1];
//		faces_[3*i+2] = mesh.polygons[i].vertices[2];
//	}
//}

//pcl::PointXYZ MeshModel::centerPoint(){
//	if (nVertices == 0)
//		return pcl::PointXYZ(0,0,0);
//
//	pcl::PointXYZ result(0,0,0);
//	for (int i=0;i<nVertices;i++){
//		result.x += vertices_[3*i];
//		result.y += vertices_[3*i+1];
//		result.z += vertices_[3*i+2];
//	}
//	result.x = result.x/nVertices;
//	result.y = result.y/nVertices;
//	result.z = result.z/nVertices;
//	return result;
//}

void MeshModel::translate(double tx, double ty, double tz){
	if (nVertices == 0)
		return;

	for (int i=0;i<nVertices;i++){
		vertices_[3*i] += tx;
		vertices_[3*i+1] += ty;
		vertices_[3*i+2] += tz;
	}
}

Matrix3Xd* MeshModel::toMatrix3Xd(){
	Matrix3Xd* result = new Matrix3Xd(3,nVertices);
	for (int i=0;i<nVertices;i++)
		for (int j=0;j<3;j++)
			(*result)(j,i) = vertices_[3*i+j];
	return result;
}

MeshModel::MeshModel(Matrix3Xd m){
	nVertices = m.cols();
	type = 0;
	nFaces = 0;
	vertices_ = new float[3*nVertices];
	for (int i=0;i<nVertices;i++)
		for (int j=0;j<3;j++)
			vertices_[3*i+j] = m(j,i);
}

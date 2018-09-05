#include "face_swap/basel_3dmm.h"
#include "face_swap/utilities.h"
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  // debug

// Boost
#include <boost/filesystem.hpp>

// HDF5
#include <H5Cpp.h>

using namespace boost::filesystem;

namespace face_swap
{
    void Mesh::save_ply(const Mesh& mesh, const std::string & ply_file)
    {
        std::ofstream ply(ply_file);
        ply.precision(6);

        // Write header
        ply << "ply" << std::endl;
        ply << "format ascii 1.0" << std::endl;
        ply << "element vertex " << mesh.vertices.rows << std::endl;
        ply << "property float x" << std::endl;
        ply << "property float y" << std::endl;
        ply << "property float z" << std::endl;
        ply << "property uchar red" << std::endl;
        ply << "property uchar green" << std::endl;
        ply << "property uchar blue" << std::endl;
        ply << "element face " << mesh.faces.rows << std::endl;
        ply << "property list uchar int vertex_indices" << std::endl;
        ply << "end_header" << std::endl;

        // Write vertices
        float* vert_data = (float*)mesh.vertices.data;
        unsigned char* color_data = mesh.colors.data;
        for (int i = 0; i < mesh.vertices.rows; ++i)
        {
            ply << *vert_data++ << " ";
            ply << *vert_data++ << " ";
            ply << *vert_data++ << " ";
            ply << (int)*color_data++ << " ";
            ply << (int)*color_data++ << " ";
            ply << (int)*color_data++ << std::endl;
        }

        // Write faces
        unsigned short* faces_data = (unsigned short*)mesh.faces.data;
        for (int i = 0; i < mesh.faces.rows; ++i)
        {
            ply << "3 " << (int)*faces_data++ << " ";
            ply << (int)*faces_data++ << " ";
            ply << (int)*faces_data++ << std::endl;
        }
    }

    cv::Mat readH5Dataset(const H5::H5File& file, const std::string& datasetName)
    {
        cv::Mat out;

        // Open the specified dataset in the file
        H5::DataSet dataset = file.openDataSet(datasetName);

        // Get dataset info
        H5T_class_t type_class = dataset.getTypeClass();
        H5::DataSpace filespace = dataset.getSpace();
        hsize_t dims[2];    // dataset dimensions
        int rank = filespace.getSimpleExtentDims(dims);

        // Read dataset
        int sizes[2] = { (int)dims[0], (int)dims[1] };
        out.create(rank, sizes, CV_32FC1);
        dataset.read(out.data, H5::PredType::NATIVE_FLOAT, H5::DataSpace(rank, dims), filespace);

        return out;
        /*
        DSetCreatPropList cparms = dataset.getCreatePlist();
        if (H5D_CHUNKED == cparms.getLayout())
        {
        // Get chunking information: rank and dimensions
        rank = cparms.getChunk(2, dims);

        // Define the memory space to read a chunk.
        H5::DataSpace cspace(rank, dims);

        // Define chunk in the file (hyperslab) to read.
        }
        else
        {
        }
        */
    }

    Mesh Basel3DMM::sample(const cv::Mat & shape_coefficients, 
        const cv::Mat & tex_coefficients)
    {
        Mesh mesh;
        mesh.faces = faces;

        cv::Mat s = shape_coefficients.mul(shapeEV);
        cv::Mat t = tex_coefficients.mul(texEV);

        mesh.vertices = shapePC * s + shapeMU;
        mesh.colors = texPC * t + texMU;

        int total_vertices = mesh.vertices.rows / 3;
        mesh.vertices = mesh.vertices.reshape(0, total_vertices);
        mesh.colors = mesh.colors.reshape(0, total_vertices);
        mesh.colors.convertTo(mesh.colors, CV_8U);

        return mesh;
    }

    Mesh Basel3DMM::sample(const cv::Mat& shape_coefficients,
        const cv::Mat& tex_coefficients, const cv::Mat& expr_coefficients)
    {
        Mesh mesh;
        mesh.faces = faces;

        cv::Mat s = shape_coefficients.mul(shapeEV);
        cv::Mat t = tex_coefficients.mul(texEV);
        cv::Mat e = expr_coefficients.mul(exprEV);

        mesh.vertices = shapePC * s + shapeMU + exprPC * e + exprMU;
        mesh.colors = texPC * t + texMU;

        int total_vertices = mesh.vertices.rows / 3;
        mesh.vertices = mesh.vertices.reshape(0, total_vertices);
        mesh.colors = mesh.colors.reshape(0, total_vertices);
        mesh.colors.convertTo(mesh.colors, CV_8U);

        return mesh;
    }

    Basel3DMM Basel3DMM::load(const std::string & model_file)
    {
        Basel3DMM basel_3dmm;

        try
        {
            // Turn off the auto-printing when failure occurs so that we can
            // handle the errors appropriately
            H5::Exception::dontPrint();

            // Open the specified file and the specified dataset in the file
            H5::H5File file(model_file.c_str(), H5F_ACC_RDONLY);
            cv::Mat faces = readH5Dataset(file, "/faces");
            basel_3dmm.shapeMU = readH5Dataset(file, "/shapeMU");
            basel_3dmm.shapePC = readH5Dataset(file, "/shapePC");
            basel_3dmm.shapeEV = readH5Dataset(file, "/shapeEV");
            basel_3dmm.texMU = readH5Dataset(file, "/texMU");
            basel_3dmm.texPC = readH5Dataset(file, "/texPC");
            basel_3dmm.texEV = readH5Dataset(file, "/texEV");
            basel_3dmm.exprMU = readH5Dataset(file, "/expMU");
            basel_3dmm.exprPC = readH5Dataset(file, "/expPC");
            basel_3dmm.exprEV = readH5Dataset(file, "/expEV");

            // Convert faces to unsigned int
            float* faces_data = (float*)faces.data;
            int faces_size = faces.total();
            basel_3dmm.faces.create(faces.size(), CV_16U);
            unsigned short* out_faces_data = (unsigned short*)basel_3dmm.faces.data;
            for (int i = 0; i < faces_size; ++i)
                *out_faces_data++ = (unsigned short)(*faces_data++);
        }
        catch (H5::DataSetIException error)
        {
            throw std::runtime_error(error.getDetailMsg());
        }

        return basel_3dmm;
    }

}   // namespace face_swap

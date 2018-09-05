#include "face_swap/face_swap_engine.h"
#include "face_swap/utilities.h"
//#include "face_swap/face_swap_c_interface.h"

#include <iostream>
#include<cmath>
#include<boost/python/module.hpp>
#include<boost/python/def.hpp>
#include<boost/python/extract.hpp>
#include<boost/python/numpy.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

struct FaceData
{
	FaceData(np::ndarray img = np::empty(p::make_tuple(0), np::dtype::get_builtin<float>()),
		np::ndarray seg = np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())) :
		img(img),
		seg(seg),
		scaled_img(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		scaled_seg(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		cropped_img(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		cropped_seg(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		scaled_landmarks(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		cropped_landmarks(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		bbox(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		scaled_bbox(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		shape_coefficients(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		tex_coefficients(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		expr_coefficients(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		vecR(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		vecT(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		K(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		shape_coefficients_flipped(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		tex_coefficients_flipped(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		expr_coefficients_flipped(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		vecR_flipped(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>())),
		vecT_flipped(np::empty(p::make_tuple(0), np::dtype::get_builtin<float>()))
	{
	}

	// Input
	np::ndarray img;
	np::ndarray seg;

	// Intermediate pipeline data
	np::ndarray scaled_img;
	np::ndarray scaled_seg;
	np::ndarray cropped_img;
	np::ndarray cropped_seg;
	np::ndarray scaled_landmarks;
	np::ndarray cropped_landmarks;
	np::ndarray bbox;
	np::ndarray scaled_bbox;
	np::ndarray shape_coefficients, tex_coefficients, expr_coefficients;
	np::ndarray vecR, vecT, K;

	// Flipped image data
	np::ndarray shape_coefficients_flipped, tex_coefficients_flipped, expr_coefficients_flipped;
	np::ndarray vecR_flipped, vecT_flipped;

	// Processing parameters
	bool enable_seg = true;
	int max_bbox_res = 0;
};

void convert(const np::ndarray& in, cv::Mat& out)
{
	if (in.get_nd() == 0) return;
	np::dtype type = in.get_dtype();
	int base_type = 0, cv_type = 0;
	if (type == np::dtype::get_builtin<float>()) base_type = CV_32F;
	else if (type == np::dtype::get_builtin<int>()) base_type = CV_32S;
	else if (type == np::dtype::get_builtin<double>()) base_type = CV_64F;
	cv_type = CV_MAKE_TYPE(base_type, in.get_nd() == 3 ? in.get_nd() : 1);
	int rows = in.shape(0), cols = 1;
	if (in.get_nd() > 1) cols = in.shape(1);
	out = cv::Mat(rows, cols, cv_type, in.get_data());
}

void convert(cv::Mat& in, np::ndarray& out)
{
	if (in.empty()) return;
	if ((void*)in.data == (void*)out.get_data()) return;

	// Get shape
	p::tuple shape;
	if (in.channels() > 1) shape = p::make_tuple(in.rows, in.cols, in.channels());
	else if (in.cols > 1) shape = p::make_tuple(in.rows, in.cols);
	else shape = p::make_tuple(in.rows);

	// Get type
	np::dtype type = np::dtype::get_builtin<unsigned char>();
	unsigned char depth = in.type() & CV_MAT_DEPTH_MASK;
	switch (depth) {
	case CV_8U:  type = np::dtype::get_builtin<unsigned char>(); break;
	case CV_8S:  type = np::dtype::get_builtin<char>(); break;
	case CV_16U: type = np::dtype::get_builtin<unsigned short>(); break;
	case CV_16S: type = np::dtype::get_builtin<short>(); break;
	case CV_32S: type = np::dtype::get_builtin<int>(); break;
	case CV_32F: type = np::dtype::get_builtin<float>(); break;
	case CV_64F: type = np::dtype::get_builtin<double>(); break;
	default:     type = np::dtype::get_builtin<unsigned char>(); break;
	}
	
	// Create ndarray
	//std::cout << "shape = " << p::extract<char const *>(p::str(shape)) << std::endl;//
	//std::cout << "type = " << p::extract<char const *>(p::str(type)) << std::endl;//
	out = np::empty(shape, type);

	if (in.isContinuous())
		memcpy(out.get_data(), in.data, in.total() * in.elemSize());
	else
	{
		char* out_data = out.get_data();
		int row_size = in.elemSize() * in.cols;
		for (int r = 0; r < in.rows; ++r)
		{
			memcpy(out_data, in.ptr<unsigned char>(r), row_size);
			out_data += row_size;
		}
	}
}

void convert(const np::ndarray& in, std::vector<cv::Point>& out)
{
	if (in.get_nd() != 2 || in.shape(0) != 68 || in.shape(1) != 2 ||
		in.get_dtype() != np::dtype::get_builtin<int>()) return;
	out.assign(((cv::Point*)in.get_data()), ((cv::Point*)in.get_data()) + 68);
}

void convert(std::vector<cv::Point>& in, np::ndarray& out)
{
	if (in.empty()) return;
	if (out.get_nd() != 2 || out.shape(0) != 68 || out.shape(1) != 2 ||
		out.get_dtype() !=  np::dtype::get_builtin<int>())
		out = np::empty(p::make_tuple(68, 2), np::dtype::get_builtin<int>());
	memcpy(out.get_data(), in.data(), 68 * 2 * sizeof(int));
}

void convert(const np::ndarray& in, cv::Rect& out)
{
	if (in.get_nd() != 1 || in.shape(0) != 4 ||
		in.get_dtype() != np::dtype::get_builtin<int>()) return;
	memcpy(&(out.x), in.get_data(), 4 * sizeof(int));
}

void convert(cv::Rect& in, np::ndarray& out)
{
	if (out.get_nd() != 1 || out.shape(0) != 4 ||
		out.get_dtype() != np::dtype::get_builtin<int>())
		out = np::empty(p::make_tuple(4), np::dtype::get_builtin<int>());
	memcpy(out.get_data(), &(in.x), 4 * sizeof(int));
}

template<typename T1, typename T2>
void convert_face_data(T1& in_face_data, T2& out_face_data)
{
	convert(in_face_data.img, out_face_data.img);
	convert(in_face_data.seg, out_face_data.seg);
	convert(in_face_data.scaled_img, out_face_data.scaled_img);
	convert(in_face_data.scaled_seg, out_face_data.scaled_seg);
	convert(in_face_data.cropped_img, out_face_data.cropped_img);
	convert(in_face_data.cropped_seg, out_face_data.cropped_seg);
	convert(in_face_data.scaled_landmarks, out_face_data.scaled_landmarks);
	convert(in_face_data.cropped_landmarks, out_face_data.cropped_landmarks);
	convert(in_face_data.bbox, out_face_data.bbox);
	convert(in_face_data.scaled_bbox, out_face_data.scaled_bbox);
	convert(in_face_data.shape_coefficients, out_face_data.shape_coefficients);
	convert(in_face_data.tex_coefficients, out_face_data.tex_coefficients);
	convert(in_face_data.expr_coefficients, out_face_data.expr_coefficients);
	convert(in_face_data.vecR, out_face_data.vecR);
	convert(in_face_data.vecT, out_face_data.vecT);
	convert(in_face_data.K, out_face_data.K);
	convert(in_face_data.shape_coefficients_flipped, out_face_data.shape_coefficients_flipped);
	convert(in_face_data.tex_coefficients_flipped, out_face_data.tex_coefficients_flipped);
	convert(in_face_data.expr_coefficients_flipped, out_face_data.expr_coefficients_flipped);
	convert(in_face_data.vecR_flipped, out_face_data.vecR_flipped);
	convert(in_face_data.vecT_flipped, out_face_data.vecT_flipped);
	in_face_data.enable_seg = out_face_data.enable_seg;
	in_face_data.max_bbox_res = out_face_data.max_bbox_res;
}

class FaceSwap
{
public:
	FaceSwap(const std::string& landmarks_path, const std::string& model_3dmm_h5_path,
		const std::string& model_3dmm_dat_path, const std::string& reg_model_path,
		const std::string& reg_deploy_path, const std::string& reg_mean_path,
		const std::string& seg_model_path, const std::string& seg_deploy_path,
		bool generic = false, bool with_expr = true, bool with_gpu = true,
		int gpu_device_id = 0)
	{
		m_fs = face_swap::FaceSwapEngine::createInstance(
			landmarks_path, model_3dmm_h5_path,
			model_3dmm_dat_path, reg_model_path,
			reg_deploy_path, reg_mean_path,
			seg_model_path, seg_deploy_path,
			generic, with_expr, with_gpu, gpu_device_id);
	}

	bool process(FaceData& face_data)
	{
		face_swap::FaceData cpp_face_data;
		convert_face_data(face_data, cpp_face_data);
		bool res = m_fs->process(cpp_face_data);
		convert_face_data(cpp_face_data, face_data);

		return res;
	}

	np::ndarray swap(FaceData& src_data, FaceData& tgt_data)
	{
		face_swap::FaceData cpp_src_data, cpp_tgt_data;
		convert_face_data(src_data, cpp_src_data);
		convert_face_data(tgt_data, cpp_tgt_data);
		cv::Mat rendered_img = m_fs->swap(cpp_src_data, cpp_tgt_data);
		convert_face_data(cpp_src_data, src_data);
		convert_face_data(cpp_tgt_data, tgt_data);

		// Output render image
		p::tuple out_shape = p::make_tuple(rendered_img.rows, rendered_img.cols, rendered_img.channels());
		np::ndarray out = np::empty(out_shape, np::dtype::get_builtin<unsigned char>());
		memcpy(out.get_data(), rendered_img.data, rendered_img.total() * rendered_img.elemSize());
		return out;
	}

private:
	std::shared_ptr<face_swap::FaceSwapEngine> m_fs;
};


bool read_face_data(const std::string& path, FaceData& face_data)
{
	face_swap::FaceData io_face_data;
	bool res = face_swap::readFaceData(path, io_face_data);
	convert_face_data(io_face_data, face_data);

	return res;
}

bool write_face_data(const std::string& path, FaceData& face_data,
	bool overwrite = false)
{
	face_swap::FaceData io_face_data;
	convert_face_data(face_data, io_face_data);
	return face_swap::writeFaceData(path, io_face_data, overwrite);
}

BOOST_PYTHON_MODULE(face_swap_py)
{
	np::initialize();

	p::class_<FaceData>("FaceData", p::init<p::optional<np::ndarray, np::ndarray>>())
		.def_readwrite("img", &FaceData::img)
		.def_readwrite("seg", &FaceData::seg)
		.def_readwrite("scaled_img", &FaceData::scaled_img)
		.def_readwrite("scaled_seg", &FaceData::scaled_seg)
		.def_readwrite("cropped_img", &FaceData::cropped_img)
		.def_readwrite("cropped_seg", &FaceData::cropped_seg)
		.def_readwrite("scaled_landmarks", &FaceData::scaled_landmarks)
		.def_readwrite("cropped_landmarks", &FaceData::cropped_landmarks)
		.def_readwrite("bbox", &FaceData::bbox)
		.def_readwrite("scaled_bbox", &FaceData::scaled_bbox)
		.def_readwrite("shape_coefficients", &FaceData::shape_coefficients)
		.def_readwrite("tex_coefficients", &FaceData::tex_coefficients)
		.def_readwrite("expr_coefficients", &FaceData::expr_coefficients)
		.def_readwrite("vecR", &FaceData::vecR)
		.def_readwrite("vecT", &FaceData::vecT)
		.def_readwrite("K", &FaceData::K)
		.def_readwrite("shape_coefficients_flipped", &FaceData::shape_coefficients_flipped)
		.def_readwrite("tex_coefficients_flipped", &FaceData::tex_coefficients_flipped)
		.def_readwrite("expr_coefficients_flipped", &FaceData::expr_coefficients_flipped)
		.def_readwrite("vecR_flipped", &FaceData::vecR_flipped)
		.def_readwrite("vecT_flipped", &FaceData::vecT_flipped)
		.def_readwrite("enable_seg", &FaceData::enable_seg)
		.def_readwrite("max_bbox_res", &FaceData::max_bbox_res)
		;

	p::class_<FaceSwap>("FaceSwap", p::init<std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, p::optional<bool, bool, bool, int>>())
		.def("process", &FaceSwap::process)
		.def("swap", &FaceSwap::swap)
		;

	p::def("read_face_data", read_face_data);
	p::def("write_face_data", write_face_data);
}
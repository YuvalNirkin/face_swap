// std
#include <iostream>
#include <exception>
#include <fstream>

// Boost
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// face_swap
#include <face_swap/face_swap_engine.h>

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::runtime_error;
using namespace boost::program_options;
using namespace boost::filesystem;

int main(int argc, char* argv[])
{
	// Parse command line arguments
	string input_path, seg_path;
	string output_path, landmarks_path;
	string model_3dmm_h5_path, model_3dmm_dat_path;
	string reg_model_path, reg_deploy_path, reg_mean_path;
	string seg_model_path, seg_deploy_path;
    string cfg_path;
    bool generic, with_expr, with_gpu;
    unsigned int gpu_device_id, verbose, pyramid_num;
	try {
		options_description desc("Allowed options");
		desc.add_options()
			("help,h", "display the help message")
            ("verbose,v", value<unsigned int>(&verbose)->default_value(0), "output debug information [0, 4]")
			("input,i", value<string>(&input_path)->required(), "image path")
			("output,o", value<string>(&output_path)->required(), "output path")
            ("segmentation,s", value<string>(&seg_path), "segmentation path")
			("landmarks,l", value<string>(&landmarks_path)->required(), "path to landmarks model file")
            ("model_3dmm_h5", value<string>(&model_3dmm_h5_path)->required(), "path to 3DMM file (.h5)")
            ("model_3dmm_dat", value<string>(&model_3dmm_dat_path)->required(), "path to 3DMM file (.dat)")
            ("reg_model,r", value<string>(&reg_model_path)->required(), "path to 3DMM regression CNN model file (.caffemodel)")
            ("reg_deploy,d", value<string>(&reg_deploy_path)->required(), "path to 3DMM regression CNN deploy file (.prototxt)")
            ("reg_mean,m", value<string>(&reg_mean_path)->required(), "path to 3DMM regression CNN mean file (.binaryproto)")
			("seg_model", value<string>(&seg_model_path), "path to face segmentation CNN model file (.caffemodel)")
			("seg_deploy", value<string>(&seg_deploy_path), "path to face segmentation CNN deploy file (.prototxt)")
            ("generic,g", value<bool>(&generic)->default_value(false), "use generic model without shape regression")
            ("expressions,e", value<bool>(&with_expr)->default_value(true), "with expressions")
			("gpu", value<bool>(&with_gpu)->default_value(true), "toggle GPU / CPU")
			("gpu_id", value<unsigned int>(&gpu_device_id)->default_value(0), "GPU's device id")
            ("cfg", value<string>(&cfg_path)->default_value("test_flip.cfg"), "configuration file (.cfg)")
			;
		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).
			positional(positional_options_description().add("input", -1)).run(), vm);

        if (vm.count("help")) {
            cout << "Usage: test_flip [options]" << endl;
            cout << desc << endl;
            exit(0);
        }

        // Read config file
        std::ifstream ifs(vm["cfg"].as<string>());
        store(parse_config_file(ifs, desc), vm);

        notify(vm);
		if (!is_regular_file(input_path)) throw error("input must be a path to an image!");
		if (!seg_path.empty() && !is_regular_file(seg_path))
			throw error("segmentation must be a path to a file!");
		if (!is_regular_file(landmarks_path)) throw error("landmarks must be a path to a file!");
        if (!is_regular_file(model_3dmm_h5_path)) throw error("model_3dmm_h5 must be a path to a file!");
        if (!is_regular_file(model_3dmm_dat_path)) throw error("model_3dmm_dat must be a path to a file!");
        if (!is_regular_file(reg_model_path)) throw error("reg_model must be a path to a file!");
        if (!is_regular_file(reg_deploy_path)) throw error("reg_deploy must be a path to a file!");
        if (!is_regular_file(reg_mean_path)) throw error("reg_mean must be a path to a file!");
		if (!seg_model_path.empty() && !is_regular_file(seg_model_path))
			throw error("seg_model must be a path to a file!");
		if (!seg_deploy_path.empty() && !is_regular_file(seg_deploy_path))
			throw error("seg_deploy must be a path to a file!");
	}
	catch (const error& e) {
        cerr << "Error while parsing command-line arguments: " << e.what() << endl;
        cerr << "Use --help to display a list of options." << endl;
		exit(1);
	}

	try
	{
        // Initialize face swap
		std::shared_ptr<face_swap::FaceSwapEngine> fs = 
			face_swap::FaceSwapEngine::createInstance(
				landmarks_path, model_3dmm_h5_path, model_3dmm_dat_path, reg_model_path,
				reg_deploy_path, reg_mean_path, seg_model_path, seg_deploy_path,
				generic, with_expr, with_gpu, gpu_device_id);

        // Read input image
        cv::Mat img = cv::imread(input_path);

        // Read segmentation or initialize segmentation model
        cv::Mat seg;
		if (seg_model_path.empty() || seg_deploy_path.empty())
		{
			if(!seg_path.empty())
				seg = cv::imread(seg_path, cv::IMREAD_GRAYSCALE);
		}

		// Flip image
		cv::Mat img_flipped;
		cv::flip(img, img_flipped, 1);

		// Process images
		face_swap::FaceData face_data, face_flipped_data;
		face_data.img = img;
		face_flipped_data.img = img_flipped;
		
		std::cout << "Processing original image..." << std::endl;
		fs->process(face_data);
		std::cout << "Processing flipped image..." << std::endl;
		fs->process(face_flipped_data);

		// Print results
		std::cout << "vecR = " << face_data.vecR << std::endl;
		std::cout << "vecR_flipped = " << face_flipped_data.vecR << std::endl;
		std::cout << "vecT = " << face_data.vecT << std::endl;
		std::cout << "vecT_flipped = " << face_flipped_data.vecT << std::endl;
		std::cout << "K = " << face_data.K << std::endl;
		std::cout << "K_flipped = " << face_flipped_data.K << std::endl;
		std::cout << "shape_coefficients = " << face_data.shape_coefficients << std::endl;
		std::cout << "shape_coefficients_flipped = " << face_flipped_data.shape_coefficients << std::endl;
		std::cout << "expr_coefficients = " << face_data.expr_coefficients << std::endl;
		std::cout << "expr_coefficients_flipped = " << face_flipped_data.expr_coefficients << std::endl;

		// Write output to file
		/*path out_file_path = output_path;
		path out_dir_path = output_path;
		if (is_directory(output_path))
		{
			path outputName = (path(input_path).stem() += ".jpg");
			out_file_path = path(output_path) /= outputName;
		}
		else out_dir_path = path(output_path).parent_path();
		cv::imwrite(out_file_path.string(), rendered_img);*/
	}
	catch (std::exception& e)
	{
		cerr << e.what() << endl;
		return 1;
	}

	return 0;
}


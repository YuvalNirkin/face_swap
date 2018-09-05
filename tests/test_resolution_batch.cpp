// std
#include <iostream>
#include <exception>
#include <fstream>

// Boost
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/regex.hpp>
#include <boost/timer/timer.hpp>

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

const std::string IMAGE_FILTER =
"(.*\\.(bmp|dib|jpeg|jpg|jpe|jp2|png|pbm|pgm|ppm|sr|ras))";

void nchoose2(const std::vector<string>& in, 
    std::vector<std::pair<string, string>>& out)
{
    size_t n = in.size();
    out.reserve(n*(n - 1));
    int i, j;
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            if(i != j) out.push_back(std::make_pair(in[i], in[j]));
        }
    }
}

void getImagesFromDir(const std::string& dir_path, std::vector<std::string>& img_paths)
{
    boost::regex filter(IMAGE_FILTER);
    boost::smatch what;
    directory_iterator end_itr; // Default ctor yields past-the-end
    for (directory_iterator it(dir_path); it != end_itr; ++it)
    {
        // Skip if not a file
        if (!boost::filesystem::is_regular_file(it->status())) continue;

        // Get extension
        std::string ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        // Skip if no match
        if (!boost::regex_match(ext, what, filter)) continue;

        img_paths.push_back(it->path().string());
    }
}


void logError(std::ofstream& log, const std::pair<string, string>& img_pair, 
    const string& msg, bool write_to_file = true)
{
    std::cerr << "Error: " << msg << std::endl;
    if (write_to_file)
    {
        log << img_pair.first << "," << img_pair.second <<
            ",Error: " << msg << std::endl;
    } 
}

cv::Mat pyramidsTest(std::shared_ptr<face_swap::FaceSwapEngine>& fs,
	const cv::Mat& img, unsigned int pyramid_num = 4)
{
	// Generate image pyramids
	std::vector<cv::Mat> pyramids(pyramid_num);
	pyramids[0] = img;
	for (int i = 1; i < pyramid_num; ++i)
		cv::pyrDown(pyramids[i - 1], pyramids[i]);

	// Process image pyramids
	std::vector<cv::Mat> rendered_pyramids(pyramid_num);
	std::vector<face_swap::FaceData> pyramids_data(pyramid_num);
	for (int i = 0; i < pyramid_num; ++i)
	{
		std::cout << "Prcoessing pyramid " << i << "..." << std::endl;
		pyramids_data[i].img = pyramids[i];
		fs->process(pyramids_data[i]);
		rendered_pyramids[i] = fs->renderFaceData(pyramids_data[i], 1 << i);
	}

	// Concatenate rendered pyramid to a single image
	cv::Mat rendered_img = rendered_pyramids[0];
	std::string text = std::to_string(pyramids_data[0].bbox.width) + " X " + std::to_string(pyramids_data[0].bbox.height);
	cv::putText(rendered_pyramids[0], text, cv::Point(10, rendered_pyramids[0].rows - 10), cv::FONT_HERSHEY_PLAIN, 10, cv::Scalar(0, 255, 0), 10);
	for (int i = 1; i < pyramid_num; ++i)
	{
		cv::resize(rendered_pyramids[i], rendered_pyramids[i], rendered_pyramids[i - 1].size());
		std::string text = std::to_string(pyramids_data[i].bbox.width) + " X " + std::to_string(pyramids_data[i].bbox.height);
		cv::putText(rendered_pyramids[i], text, cv::Point(10, rendered_pyramids[0].rows - 10), cv::FONT_HERSHEY_PLAIN, 10, cv::Scalar(0, 255, 0), 10);
		cv::vconcat(rendered_img, rendered_pyramids[i], rendered_img);
	}

	return rendered_img;
}

//cv::Mat resolutionsTest(std::shared_ptr<face_swap::FaceSwapEngine>& fs,
//	const cv::Mat& img, const std::vector<unsigned int>& resolutions,
//	unsigned int pyramid_num = 4)
//{
//	// Generate image pyramids
//	std::vector<cv::Mat> pyramids(pyramid_num);
//	pyramids[0] = img;
//	for (int i = 1; i < pyramid_num; ++i)
//		cv::pyrDown(pyramids[i - 1], pyramids[i]);
//
//	// Find initial resolution where a face is detected
//	std::vector<face_swap::FaceData> pyramids_data(pyramid_num);
//	cv::Mat initial_img;
//	face_swap::FaceData initial_img_data;
//	for (int i = 0; i < pyramid_num; ++i)
//	{
//		std::cout << "Prcoessing pyramid " << i << "..." << std::endl;
//		fs->process(pyramids[i], pyramids_data[i]);
//		if (pyramids_data[i].landmarks.empty()) continue;
//		initial_img = pyramids[i];
//		initial_img_data = pyramids_data[i];
//		break;
//	}
//
//	// Check that a face was actually found
//	if (initial_img_data.landmarks.empty())
//		return fs->renderFaceData(img, initial_img_data);
//
//	// Sort resolutions in descending order
//	std::vector<unsigned int> resolutions_sorted = resolutions;
//	std::sort(resolutions_sorted.begin(), resolutions_sorted.end(), std::greater<unsigned int>());
//
//	// Remove resolutions that are greater than the initial image
//	resolutions_sorted.erase(std::remove_if(resolutions_sorted.begin(), resolutions_sorted.end(),
//		[initial_img_data](const int& x) { return x > initial_img_data.bbox.width; }),
//		resolutions_sorted.end());
//
//	// Render image pyramids by specified resolutions
//	std::vector<cv::Mat> rendered_pyramids(resolutions_sorted.size());
//	std::vector<face_swap::FaceData> res_pyramids_data(resolutions_sorted.size());
//	for(int i = 0; i < resolutions_sorted.size(); ++i)
//	{
//		std::cout << "Prcoessing resolution pyramid " << i << "..." << std::endl;
//		int res = resolutions_sorted[i];
//		float scale = float(res) / float(initial_img_data.bbox.width);
//		cv::Mat temp;
//		
//		//cv::resize(initial_img, temp, cv::Size(), scale, scale, cv::INTER_CUBIC);
//		cv::Size target_size((int)std::round(initial_img.cols*scale), (int)std::round(initial_img.rows*scale));
//		for (int j = 0; j < pyramids.size(); ++j)
//		{
//			if (pyramids[j].cols < target_size.width) continue;
//			cv::resize(pyramids[j], temp, target_size, 0.0, 0.0, cv::INTER_CUBIC);
//			break;
//		}
//
//		fs->process(temp, res_pyramids_data[i]);
//		rendered_pyramids[i] = fs->renderFaceData(temp, res_pyramids_data[i], 1.0f / scale);
//	}
//
//	// Concatenate rendered pyramid to a single image
//	cv::Mat rendered_img = rendered_pyramids[0];
//	std::string text = std::to_string(res_pyramids_data[0].bbox.width) + " X " + std::to_string(res_pyramids_data[0].bbox.height);
//	cv::putText(rendered_pyramids[0], text, cv::Point(10, rendered_pyramids[0].rows - 10), cv::FONT_HERSHEY_PLAIN, 10, cv::Scalar(0, 255, 0), 10);
//	for (int i = 1; i < rendered_pyramids.size(); ++i)
//	{
//		cv::resize(rendered_pyramids[i], rendered_pyramids[i], rendered_pyramids[i - 1].size(), cv::INTER_CUBIC);
//		std::string text = std::to_string(res_pyramids_data[i].bbox.width) + " X " + std::to_string(res_pyramids_data[i].bbox.height);
//		cv::putText(rendered_pyramids[i], text, cv::Point(10, rendered_pyramids[0].rows - 10), cv::FONT_HERSHEY_PLAIN, 10, cv::Scalar(0, 255, 0), 10);
//		cv::vconcat(rendered_img, rendered_pyramids[i], rendered_img);
//	}
//
//	return rendered_img;
//}

int main(int argc, char* argv[])
{
	// Parse command line arguments
    string input_path, seg_path, output_path, landmarks_path;
	string model_3dmm_h5_path, model_3dmm_dat_path;
	string reg_model_path, reg_deploy_path, reg_mean_path;
	string seg_model_path, seg_deploy_path;
    string log_path, cfg_path;
    bool generic, with_expr, with_gpu;
    unsigned int gpu_device_id, verbose, pyramid_num;
	std::vector<unsigned int> resolutions;
	try {
		options_description desc("Allowed options");
		desc.add_options()
			("help,h", "display the help message")
            ("verbose,v", value<unsigned int>(&verbose)->default_value(0), "output debug information [0, 4]")
			("input,i", value<string>(&input_path)->required(), "path to input directory or image pairs list")
			("output,o", value<string>(&output_path)->required(), "output directory")
            ("segmentations,s", value<string>(&seg_path)->default_value(""), "segmentations directory")
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
			("pyramids,p", value<unsigned int>(&pyramid_num)->default_value(4), "number of pyramids")
			("resolutions", value<std::vector<unsigned int>>(&resolutions), "list of resolutions")
            ("log", value<string>(&log_path)->default_value("test_resolution_batch_log.csv"), "log file path")
            ("cfg", value<string>(&cfg_path)->default_value("test_resolution_batch.cfg"), "configuration file (.cfg)")
			;
		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).
			positional(positional_options_description().add("input", -1)).run(), vm);

        if (vm.count("help")) {
            cout << "Usage: test_resolution_batch [options]" << endl;
            cout << desc << endl;
            exit(0);
        }

        // Read config file
        std::ifstream ifs(vm["cfg"].as<string>());
        store(parse_config_file(ifs, desc), vm);

        notify(vm);

        if (!(is_regular_file(input_path) || is_directory(input_path)))
            throw error("input must be a path to input directory or image pairs list!");
        if(!seg_path.empty() && !is_directory(seg_path))
            throw error("segmentations must be a path to segmentations directory!");
        if ( !is_directory(output_path))
            throw error("output must be a path to a directory!");
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
        // Initialize log file
        std::ofstream log;
        if (verbose > 0)
            log.open(log_path);

        // Parse image paths
		std::vector<string> img_paths;
		getImagesFromDir(input_path, img_paths);

		// Initialize face swap
		std::shared_ptr<face_swap::FaceSwapEngine> fs =
			face_swap::FaceSwapEngine::createInstance(
				landmarks_path, model_3dmm_h5_path, model_3dmm_dat_path, reg_model_path,
				reg_deploy_path, reg_mean_path, seg_model_path, seg_deploy_path,
				generic, with_expr, with_gpu, gpu_device_id);

        // Initialize timer
        boost::timer::cpu_timer timer;
        float total_time = 0.0f, fps = 0.0f;
        int frame_counter = 0;

        // For each image pair
        string prev_src_path, prev_tgt_path;
        cv::Mat img, rendered_img;
        for (const auto& img_path : img_paths)
        {
            // Check if output image already exists
            path outputName = (path(img_path).stem() += ".jpg");
            string curr_output_path = (path(output_path) /= outputName).string();
            if (is_regular_file(curr_output_path))
            {
                std::cout << "Skipping: " << path(img_path).filename() << std::endl;
                continue;
            }
            std::cout << "Processing: " << path(img_path).filename() << std::endl;

            // Read source and target images
			img = cv::imread(img_path);

            // Read source and target segmentations
            cv::Mat seg;
            if (!(!(seg_model_path.empty() || seg_deploy_path.empty()) || seg_path.empty()))
            {
                string seg_path = (path(seg_path) /= (path(img_path).stem() += ".png")).string();
                if (is_regular_file(seg_path))
                    seg = cv::imread(seg_path, cv::IMREAD_GRAYSCALE);
            }

            // Start measuring time
            timer.start();

			// Run test
			rendered_img = pyramidsTest(fs, img, pyramid_num);
			//if (resolutions.empty())
			//	rendered_img = pyramidsTest(fs, img, pyramid_num);
			//else rendered_img = resolutionsTest(fs, img, resolutions, pyramid_num);

            // Stop measuring time
            timer.stop();

            // Write output to file
            std::cout << "Writing " << outputName << " to output directory." << std::endl;
            cv::imwrite(curr_output_path, rendered_img);

            // Print current fps
            total_time += (timer.elapsed().wall*1.0e-9);
            fps = (++frame_counter) / total_time;
            std::cout << "total_time = " << total_time << std::endl;
            std::cout << "fps = " << fps << std::endl;
        }
	}
	catch (std::exception& e)
	{
		cerr << e.what() << endl;
		return 1;
	}

	return 0;
}


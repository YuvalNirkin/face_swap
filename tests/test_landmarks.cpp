#include "face_swap/face_detection_landmarks.h"
#include "face_swap/landmarks_utilities.h"

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

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::runtime_error;
using namespace boost::program_options;
using namespace boost::filesystem;

const std::string IMAGE_FILTER =
"(.*\\.(bmp|dib|jpeg|jpg|jpe|jp2|png|pbm|pgm|ppm|sr|ras))";


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

void parseInputPaths(const std::vector<std::string>& input_paths, std::vector<std::string>& img_paths)
{
	boost::regex filter(IMAGE_FILTER);
	boost::smatch what;

	// For each input path
	for (size_t i = 0; i < input_paths.size(); ++i)
	{
		const string& input_path = input_paths[i];

		// Get extension
		std::string ext = path(input_path).extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

		if (is_regular_file(input_path) && boost::regex_match(ext, what, filter))
			img_paths.push_back(input_path);
		else if (is_directory(input_path))
			getImagesFromDir(input_path, img_paths);
		else throw error("input must be contain paths to image files or input directories!");
	}
}

int main(int argc, char* argv[])
{
	// Parse command line arguments
	std::vector<string> input_paths;
    string output_path, landmarks_path, cfg_path;
	bool preview;
	try {
		options_description desc("Allowed options");
		desc.add_options()
			("help,h", "display the help message")
			("input,i", value<std::vector<string>>(&input_paths)->required(), "input paths")
            ("output,o", value<string>(&output_path), "output directory")
			("landmarks,l", value<string>(&landmarks_path)->required(), "path to landmarks model")
			("preview,p", value<bool>(&preview)->default_value(true), "toggle preview")
			("cfg", value<string>(&cfg_path)->default_value("test_landmarks.cfg"), "configuration file (.cfg)")
			;
		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).
			positional(positional_options_description().add("input", -1)).run(), vm);

        if (vm.count("help")) {
            cout << "Usage: test_landmarks [options]" << endl;
            cout << desc << endl;
            exit(0);
        }

        // Read config file
        std::ifstream ifs(vm["cfg"].as<string>());
        store(parse_config_file(ifs, desc), vm);

        notify(vm);

        if (!output_path.empty() && !is_directory(output_path))
            throw error("output must be a path to a directory!");
		if (!is_regular_file(landmarks_path))
			throw error("landmarks must be a path to a file!");
	}
	catch (const error& e) {
        cerr << "Error while parsing command-line arguments: " << e.what() << endl;
        cerr << "Use --help to display a list of options." << endl;
		exit(1);
	}

	try
	{
		// Parse image paths
		std::vector<string> img_paths;
		parseInputPaths(input_paths, img_paths);

		// Initialize face detection and landmarks
		std::shared_ptr<face_swap::FaceDetectionLandmarks> lms =
			face_swap::FaceDetectionLandmarks::create(landmarks_path);

		// Initialize timer
		boost::timer::cpu_timer timer;
		float det_delta_time = 0.0f, lms_delta_time = 0.0f;
        
        // For each image
        for (const string& img_path : img_paths)
        {
			std::cout << "Processing " << path(img_path).filename() << "..." << std::endl;

			// Read input image
            cv::Mat img = cv::imread(img_path);

			// Start measuring time
			timer.start();

			// Detect faces and extract landmarks
			std::vector<face_swap::Face> faces;
			lms->process(img, faces);

			// Stop measuring time
			timer.stop();

			// Print current timing statistics
			lms_delta_time += (timer.elapsed().wall*1.0e-9 - lms_delta_time)*0.1f;
			std::cout << "Landmarks timing = " << lms_delta_time << "s (" <<
				(1.0f / lms_delta_time) << " fps)" << std::endl;

			if(preview)
			{
				// Render landmarks
				cv::Mat out = img.clone();
				face_swap::render(out, faces);
				cv::imshow("test_landmarks", out);
				cv::waitKey(1);
			}
        }
	}
	catch (std::exception& e)
	{
		cerr << e.what() << endl;
		return 1;
	}

	return 0;
}


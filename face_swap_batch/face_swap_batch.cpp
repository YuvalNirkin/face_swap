// std
#include <iostream>
#include <exception>

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

// sfl
#include <sfl/sequence_face_landmarks.h>
#include <sfl/utilities.h>

// face_swap
#include <face_swap/face_swap.h>

// OpenGL
#include <GL/glew.h>

// Qt
#include <QApplication>
#include <QOpenGLContext>
#include <QOffscreenSurface>
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

void readImagePairsFromFile(const std::string& csv_file,
    std::vector<std::pair<string, string>>& out)
{
    std::ifstream file(csv_file);
    std::pair<string, string> img_pair;
    while (file.good())
    {
        std::getline(file, img_pair.first, ',');
        std::getline(file, img_pair.second, '\n');
        if (img_pair.first.empty() || img_pair.second.empty()) return;
        out.push_back(img_pair);
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

int main(int argc, char* argv[])
{
	// Parse command line arguments
    string input_path, seg_path, output_path, landmarks_path;
	string model_3dmm_h5_path, model_3dmm_dat_path;
	string reg_model_path, reg_deploy_path, reg_mean_path;
	string seg_model_path, seg_deploy_path;
    string log_path, cfg_path;
    bool generic, with_expr, with_gpu;
    unsigned int gpu_device_id, verbose;
	try {
		options_description desc("Allowed options");
		desc.add_options()
			("help,h", "display the help message")
            ("verbose,v", value<unsigned int>(&verbose)->default_value(0), "output debug information")
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
            ("log", value<string>(&log_path)->default_value("face_swap_batch_log.csv"), "log file path")
            ("cfg", value<string>(&cfg_path)->default_value("face_swap_batch.cfg"), "configuration file (.cfg)")
			;
		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).
			positional(positional_options_description().add("input", -1)).run(), vm);

        if (vm.count("help")) {
            cout << "Usage: face_swap_batch [options]" << endl;
            cout << desc << endl;
            exit(0);
        }

        // Read config file
        ifstream ifs(vm["cfg"].as<string>());
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
        // Intialize OpenGL context
        QApplication a(argc, argv);

        QSurfaceFormat surfaceFormat;
        surfaceFormat.setMajorVersion(1);
        surfaceFormat.setMinorVersion(5);

        QOpenGLContext openGLContext;
        openGLContext.setFormat(surfaceFormat);
        openGLContext.create();
        if (!openGLContext.isValid()) return -1;

        QOffscreenSurface surface;
        surface.setFormat(surfaceFormat);
        surface.create();
        if (!surface.isValid()) return -2;

        openGLContext.makeCurrent(&surface);

        // Initialize GLEW
        GLenum err = glewInit();
        if (GLEW_OK != err)
        {
            // Problem: glewInit failed, something is seriously wrong
            fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
            throw std::runtime_error("Failed to initialize GLEW!");
        }

        // Initialize log file
        std::ofstream log;
        if (verbose > 0)
            log.open(log_path);

        // Parse image pairs
        std::vector<std::pair<string, string>> img_pairs;
        if (is_directory(input_path))
        {
            std::vector<string> img_paths;
            getImagesFromDir(input_path, img_paths);
            nchoose2(img_paths, img_pairs);
        }
        else readImagePairsFromFile(input_path, img_pairs);

        // Initialize face swap
		face_swap::FaceSwap fs(landmarks_path, model_3dmm_h5_path, model_3dmm_dat_path,
            reg_model_path, reg_deploy_path, reg_mean_path, generic, with_expr,
			with_gpu, (int)gpu_device_id);
		if (!(seg_model_path.empty() || seg_deploy_path.empty()))
			fs.setSegmentationModel(seg_model_path, seg_deploy_path);

        // Initialize timer
        boost::timer::cpu_timer timer;
        float total_time = 0.0f, fps = 0.0f;
        int frame_counter = 0;

        // For each image pair
        string prev_src_path, prev_tgt_path;
        cv::Mat source_img, target_img, rendered_img;
        for (const auto& img_pair : img_pairs)
        {
            // Check if output image already exists
            path outputName = (path(img_pair.first).stem() += "_") +=
                (path(img_pair.second).stem() += ".jpg");
            string curr_output_path = (path(output_path) /= outputName).string();
            if (is_regular_file(curr_output_path))
            {
                std::cout << "Skipping: " << path(img_pair.first).filename() <<
                    " -> " << path(img_pair.second).filename() << std::endl;
                continue;
            }
            std::cout << "Face swapping: " << path(img_pair.first).filename() <<
                " -> " << path(img_pair.second).filename() << std::endl;

            // Read source and target images
            if(prev_src_path != img_pair.first)
                source_img = cv::imread(img_pair.first);
            if (prev_tgt_path != img_pair.second)
                target_img = cv::imread(img_pair.second);

            // Read source and target segmentations
            cv::Mat source_seg, target_seg;
            if (!(fs.isSegmentationModelInit() || seg_path.empty()))
            {
                string src_seg_path = (path(seg_path) /= 
                    (path(img_pair.first).stem() += ".png")).string();
                string tgt_seg_path = (path(seg_path) /= 
                    (path(img_pair.second).stem() += ".png")).string();

                if (is_regular_file(src_seg_path) && prev_src_path != img_pair.first)
                    source_seg = cv::imread(src_seg_path, cv::IMREAD_GRAYSCALE);
                if (is_regular_file(tgt_seg_path) && prev_tgt_path != img_pair.second)
                    target_seg = cv::imread(tgt_seg_path, cv::IMREAD_GRAYSCALE);
            }

            // Start measuring time
            timer.start();

            // Set source and target
            if (prev_src_path != img_pair.first)
            {
                if (!fs.setSource(source_img, source_seg))
                {
                    logError(log, img_pair, "Failed to find faces in source image!", verbose);
                    prev_src_path = "";
                    continue;
                }
            }
                
            if (prev_tgt_path != img_pair.second)
            {
                if (!fs.setTarget(target_img, target_seg))
                {
                    logError(log, img_pair, "Failed to find faces in target image!", verbose);
                    prev_tgt_path = "";
                    continue;
                }
            }
                
            prev_src_path = img_pair.first;
            prev_tgt_path = img_pair.second;

            // Do face swap
            rendered_img = fs.swap();
            if (rendered_img.empty())
            {
                logError(log, img_pair, "Face swap failed!", verbose);
                continue;
            }

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

            // Debug
            if (verbose > 0)
            {
                // Write projected meshes
                string debug_src_mesh_path = (path(output_path) /=
                    (path(curr_output_path).stem() += "_src_mesh.jpg")).string();
                cv::Mat debug_src_mesh_img = fs.debugSourceMesh();
                cv::imwrite(debug_src_mesh_path, debug_src_mesh_img);

                string debug_tgt_mesh_path = (path(output_path) /=
                    (path(curr_output_path).stem() += "_tgt_mesh.jpg")).string();
                cv::Mat debug_tgt_mesh_img = fs.debugTargetMesh();
                cv::imwrite(debug_tgt_mesh_path, debug_tgt_mesh_img);

                // Write meshes
                string src_ply_path = (path(output_path) /=
                    (path(curr_output_path).stem() += "_src_mesh.ply")).string();
                string tgt_ply_path = (path(output_path) /=
                    (path(curr_output_path).stem() += "_tgt_mesh.ply")).string();
				face_swap::Mesh::save_ply(fs.getSourceMesh(), src_ply_path);
				face_swap::Mesh::save_ply(fs.getTargetMesh(), tgt_ply_path);

                // Write landmarks render
                string debug_src_landmarks_path = (path(output_path) /=
                    (path(curr_output_path).stem() += "_src_landmarks.jpg")).string();
                cv::Mat debug_src_landmarks_img = fs.debugSourceLandmarks();
                cv::imwrite(debug_src_landmarks_path, debug_src_landmarks_img);

                string debug_tgt_landmarks_path = (path(output_path) /=
                    (path(curr_output_path).stem() += "_tgt_landmarks.jpg")).string();
                cv::Mat debug_tgt_landmarks_img = fs.debugTargetLandmarks();
                cv::imwrite(debug_tgt_landmarks_path, debug_tgt_landmarks_img);

                // Write rendered image
                string debug_render_path = (path(output_path) /=
                    (path(curr_output_path).stem() += "_render.jpg")).string();
                cv::Mat debug_render_img = fs.debugRender();
                cv::imwrite(debug_render_path, debug_render_img);
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


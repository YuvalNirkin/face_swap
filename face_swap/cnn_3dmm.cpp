#include "face_swap/cnn_3dmm.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  // debug

#include <exception>
#include <fstream>

#include <H5Cpp.h>

using std::string;
using namespace caffe;

namespace face_swap
{
CNN3DMM::CNN3DMM(const string& deploy_file, const string& caffe_model_file,
    const std::string& mean_file, bool init_cnn, bool with_gpu, int gpu_device_id) :
    m_num_channels(0), m_with_gpu(with_gpu)
{
    if (!init_cnn) return;

    // Initialize device mode
    if (with_gpu)
    {
        Caffe::SetDevice(gpu_device_id);
        Caffe::set_mode(Caffe::GPU);
    }
    else Caffe::set_mode(Caffe::CPU);

    // Set test mode
    //Caffe::set_phase(Caffe::TEST);

    // Load the network
    m_net.reset(new Net<float>(deploy_file, caffe::TEST));
    m_net->CopyTrainedLayersFrom(caffe_model_file);

    CHECK_EQ(m_net->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(m_net->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = m_net->input_blobs()[0];
    m_num_channels = input_layer->channels();
    CHECK(m_num_channels == 3 || m_num_channels == 1)
            << "Input layer should have 1 or 3 channels.";
    m_input_size = cv::Size(input_layer->width(), input_layer->height());

    // Load mean file
    m_mean = readMean(mean_file);
}

void CNN3DMM::process(const cv::Mat& img,
    cv::Mat& shape_coefficients, cv::Mat& tex_coefficients)
{
    // Prepare input data
    cv::Mat img_processed = preprocess(img);
    copyInputData(img_processed);
	const vector<Blob<float>*>& output_blobs = m_net->Forward();

    // Output results
	const std::vector<int>& shape = m_net->blob_by_name("fc_ftnew")->shape();
	float* featues = m_net->blob_by_name("fc_ftnew")->mutable_cpu_data();
    shape_coefficients = cv::Mat_<float>(99, 1, featues).clone();
    tex_coefficients = cv::Mat_<float>(99, 1, featues + 99).clone();
}

CNN3DMM::~CNN3DMM()
{
}

void CNN3DMM::wrapInputLayer(std::vector<cv::Mat>& input_channels)
{
    Blob<float>* input_layer = m_net->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();

#ifdef CPU_ONLY
    float* input_data = input_layer->mutable_cpu_data();
#else
    float* input_data = input_layer->mutable_gpu_data();
#endif

    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }
}

cv::Mat CNN3DMM::readMean(const std::string & mean_file) const
{
    // Read binary proto file
    caffe::BlobProto blob_proto;
    caffe::ReadProtoFromBinaryFileOrDie(mean_file, &blob_proto);

    // Convert from BlobProto to Blob<float>
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), m_num_channels)
        << "Number of channels of mean file doesn't match input layer.";

    // The format of the mean file is planar 32-bit float BGR or grayscale
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < m_num_channels; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    // Merge the separate channels into a single image
    cv::Mat mean;
    cv::merge(channels, mean);

    return mean;
}

cv::Mat CNN3DMM::preprocess(const cv::Mat& img)
{
    // Convert the input image to the input image format of the network.
    cv::Mat sample;
    if (img.channels() == 3 && m_num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && m_num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && m_num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && m_num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != m_input_size)
        cv::resize(sample, sample_resized, m_input_size, 0, 0, cv::INTER_CUBIC);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (m_num_channels == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    //cv::Mat sample_normalized;
    //cv::subtract(sample_float, mean_, sample_normalized);

    // Normalize the image by substracting the mean
    //subtractMean_c3(sample_float);
    //cv::Mat sample_normalized = sample_float;
    sample_float -= m_mean;

    return sample_float;
}

void CNN3DMM::copyInputData(cv::Mat& img)
{
    Blob<float>* input_layer = m_net->input_blobs()[0];
    float* buf = new float[input_layer->count()];

    int r, c, cnl, channels = img.channels();
    float* img_data = nullptr;
    float* buf_data = buf;
    for(cnl = 0; cnl < channels; ++cnl)
    {
        img_data = ((float*)img.data) + cnl;
        for(r = 0; r < img.rows; ++r)
        {
            for(c = 0; c < img.cols; ++c)
            {
                *buf_data++ = *img_data;
                img_data += channels;
            }
        }
    }

    if(m_with_gpu)
        caffe_copy(input_layer->count(), buf, input_layer->mutable_gpu_data());
    else 
        caffe_copy(input_layer->count(), buf, input_layer->mutable_cpu_data());

    delete[] buf;
}

}   // namespace face_swap

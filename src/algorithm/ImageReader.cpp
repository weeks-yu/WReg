#include "ImageReader.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

ImageReader::ImageReader()
{
	filename = "";
	fileInput = nullptr;
}

ImageReader::~ImageReader()
{

}

bool ImageReader::getNextFrame(cv::Mat &rgb, cv::Mat &depth, double &timestamp)
{
	if (rgbStream.empty() || depthStream.empty())
	{
		return false;
	}

	std::string strColor = rgbStream.front();
	rgbStream.pop();
	rgb = cv::imread(strColor);

	std::string strDepth = depthStream.front();
	depthStream.pop();
	depth = cv::imread(strDepth, -1);

	return true;
}

void ImageReader::registerColorToDepth(const cv::Mat &rgb, const cv::Mat &depth, cv::Mat &rgbRegistered)
{
	throw std::exception("Exception : not implemented.");
}

void ImageReader::registerDepthToColor(const cv::Mat &rgb, const cv::Mat &depth, cv::Mat &depthRegistered)
{
	throw std::exception("Exception : not implemented.");
}

bool ImageReader::create(const char* filename_)
{
	filename = std::string(filename_);
	std::string input = filename + "/read.txt";
	fileInput = new std::ifstream(input);
	if (fileInput != nullptr)
		return true;
	return false;
}

void ImageReader::start()
{
	std::string line;
	while (fileInput != nullptr && std::getline(*fileInput, line))
	{
		int pos = line.find(' ');
		std::string strColor = filename + "/" + line.substr(0, pos);
		std::string strDepth = filename + "/" + line.substr(pos + 1, line.length() - pos - 1);
		rgbStream.push(strColor);
		depthStream.push(strDepth);
	}
	return;
}

void ImageReader::stop()
{
	fileInput->close();
	return;
}
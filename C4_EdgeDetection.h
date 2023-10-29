#pragma once

#include "myImgProcess.hpp"
#include "C2_GraytransAFilter.h"
//enum class BoarderType;
enum class C4Session
{
	None,
	Edge,
	Canny,
	LOG,
	Hough
};
enum class GradOperator
{
	ROBERTS,
	PREWITT,
	SOBEL
};
//enum class BoarderType;
//BoarderType cool = BoarderType::CONSTANT;

cv::Mat RobertsFilter(const cv::Mat& src, cv::Mat& dst, const BoarderType& type = BoarderType::CONSTANT, int constant = 0);

cv::Mat PreWittFilter(const cv::Mat& src, cv::Mat& dst, const BoarderType& type = BoarderType::CONSTANT, int constant = 0);

cv::Mat Sobel(const cv::Mat& src, cv::Mat& dst, const BoarderType& type = BoarderType::CONSTANT, int constant = 0);

std::pair<cv::Mat, cv::Mat> getLoGVector(int size, double sigma);
void LoGFilter(const cv::Mat& src, cv::Mat& dst, int kernelSize, double sigmaX);

cv::Mat nms(const cv::Mat& intensity, const cv::Mat& gradient);
cv::Mat DualThreshold(const cv::Mat& src, double threshold1, double threshold2);
cv::Mat trackHysteresis(const cv::Mat& src);
void Canny(cv::Mat src, cv::Mat& edge, double threshold1, double threshold2,
	const GradOperator gradientOper = GradOperator::SOBEL);

//cv::Mat src, gray;
void hough_change(const cv::Mat& src, const cv::Mat& gray);

void HoughLines(cv::Mat edge, std::vector<cv::Vec2f>& lines,
	double rho, double theta, int threshold,
	double srn = 0, double stn = 0,
	double min_theta = 0, double max_theta = CV_PI);
void HoughCircles(cv::Mat image, std::vector<cv::Vec3f>& circles,
	int method, double dp, double minDist,
	double param1 = 100, double param2 = 100,
	int minRadius = 0, int maxRadius = 0);
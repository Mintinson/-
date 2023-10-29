#include "C4_EdgeDetection.h"
//#include "myImgProcess.hpp"
//#include "C2_GraytransAFilter.h"


cv::Mat RobertsFilter(const cv::Mat& src, cv::Mat& dst, const BoarderType& type, int constant)
{
	uchar* srcData = src.data;
	int srcStep = src.step;
	int channels = src.channels();
	int dstRows = src.rows;
	int dstCols = src.cols;

	cv::Mat tmp(dstRows, dstCols, src.type());
	int dataType = src.type() == CV_8UC1 ? CV_32FC1 : CV_32FC3;
	//std::cout << CV_16FC1 << std::endl;
	//std::cout << dataType << std::endl;
	cv::Mat phase(dstRows, dstCols, dataType);
	//uchar* phaseData = phase.data;
	uchar* dstData = tmp.data;
	int dstStep = tmp.step;


	std::unique_ptr<double[]> tempVals(new double[dstRows * dstCols * channels]);
	double maxVal = 0;

	for (int c = 0; c != channels; c++)
	{
		for (int row = 0; row != dstRows - 1; ++row)
		{
			for (int col = 0; col != dstCols - 1; ++col)
			{
				double x = -1 * (*(srcData + row * dstStep + col * channels + c))
					+ *(srcData + (row + 1) * dstStep + (col + 1) * channels + c);
				double y = -1 * (*(srcData + row * dstStep + (col + 1) * channels + c))
					+ *(srcData + (row + 1) * dstStep + col * channels + c);
				tempVals[row * dstStep + col * channels + c] = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
				if (tempVals[row * dstStep + col * channels + c] > maxVal) maxVal = tempVals[row * dstStep + col * channels + c];
				//if (dataType CV_32FC1
				//if (dataType CV_32FC1
				if (dataType == CV_32FC1) phase.at<float>(row, col) = std::atan2f(y, x);
				else phase.at<cv::Vec3f>(row, col)[c] = std::atan2f(y, x);
			}
		}
	}
	for (int c = 0; c != channels; c++)
	{
		for (int row = 0; row != dstRows; ++row)
		{
			for (int col = 0; col != dstCols; ++col)
			{

				dstData[row * dstStep + col * channels + c] = tempVals[row * dstStep + col * channels + c] / maxVal * 255;
			}
		}
	}
	dst = tmp;
	return phase;
}

cv::Mat PreWittFilter(const cv::Mat& src, cv::Mat& dst, const BoarderType& type, int constant)
{
	uchar* srcData = src.data;
	int srcStep = src.step;
	int channels = src.channels();
	int dstRows = src.rows;
	int dstCols = src.cols;

	cv::Mat tmp(dstRows, dstCols, src.type());
	int dataType = src.type() == CV_8UC1 ? CV_32FC1 : CV_32FC3;
	//std::cout << CV_16FC1 << std::endl;
	//std::cout << dataType << std::endl;
	cv::Mat phase(dstRows, dstCols, dataType);
	//uchar* phaseData = phase.data;
	uchar* dstData = tmp.data;
	int dstStep = tmp.step;

	std::unique_ptr<double[]> tempVals(new double[dstRows * dstCols * channels]);
	double maxVal = 0;


	for (int c = 0; c != channels; c++)
	{
		for (int row = 0; row != dstRows; ++row)
		{
			for (int col = 0; col != dstCols; ++col)
			{
				double x = 0;
				double y = 0;
				for (int i = 0; i < 3; ++i)
				{
					x += -1 * boarderPixel(srcData, dstStep, channels, c, row - 1, dstRows, col - 1 + i, dstCols, type, constant)
						+ boarderPixel(srcData, dstStep, channels, c, row + 1, dstRows, col - 1 + i, dstCols, type, constant);
					y += -1 * boarderPixel(srcData, dstStep, channels, c, row + i - 1, dstRows, col - 1, dstCols, type, constant)
						+ boarderPixel(srcData, dstStep, channels, c, row + i - 1, dstRows, col + 1, dstCols, type, constant);
				}
				tempVals[row * dstStep + col * channels + c] = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
				if (tempVals[row * dstStep + col * channels + c] > maxVal) maxVal = tempVals[row * dstStep + col * channels + c];
				//if (dataType CV_32FC1
				if (dataType == CV_32FC1) phase.at<float>(row, col) = std::atan2f(y, x);
				else phase.at<cv::Vec3f>(row, col)[c] = std::atan2f(y, x);
			}
		}
	}
	// 归一化处理
	for (int c = 0; c != channels; c++)
	{
		for (int row = 0; row != dstRows; ++row)
		{
			for (int col = 0; col != dstCols; ++col)
			{

				dstData[row * dstStep + col * channels + c] = tempVals[row * dstStep + col * channels + c] / maxVal * 255;
			}
		}
	}

	dst = tmp;
	return phase;
}

cv::Mat Sobel(const cv::Mat& src, cv::Mat& dst, const BoarderType& type, int constant)
{
	// uchar *srcData = src.data;
	// int srcStep = src.step;
	int channels = src.channels();
	int dstRows = src.rows;
	int dstCols = src.cols;

	cv::Mat tmp(dstRows, dstCols, src.type());
	int dataType = src.type() == CV_8UC1 ? CV_32FC1 : CV_32FC3;
	cv::Mat phase(dstRows, dstCols, dataType);
	uchar* dstData = tmp.data;
	int dstStep = tmp.step;

	MatElements srcData(src);

	std::unique_ptr<double[]> tempVals(new double[dstRows * dstCols * channels]);
	double maxVal = 0;
	std::vector<int> coef{1, 2, 1};

	for (int c = 0; c != channels; c++)
	{
		for (int row = 0; row != dstRows; ++row)
		{
			for (int col = 0; col != dstCols; ++col)
			{
				double x = 0;
				double y = 0;
				for (int i = 0; i < 3; ++i)
				{
					y += -coef[i] * srcData.boarderPixel(row - 1, col - 1 + i, c, type, cv::Scalar(0, 0, 0)) + coef[i] * srcData.boarderPixel(row + 1, col - 1 + i, c, type, cv::Scalar(0, 0, 0));
					x += -coef[i] * srcData.boarderPixel(row + i - 1, col - 1, c, type, cv::Scalar(0, 0, 0)) + coef[i] * srcData.boarderPixel(row + i - 1, col + 1, c, type, cv::Scalar(0, 0, 0));
				}
				tempVals[row * dstStep + col * channels + c] = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
				if (tempVals[row * dstStep + col * channels + c] > maxVal) maxVal = tempVals[row * dstStep + col * channels + c];
				if (dataType == CV_32FC1)
					phase.at<float>(row, col) = 1.0 * y / x;
				else
					phase.at<cv::Vec3f>(row, col)[c] = 1.0 * y / x;
			}
		}
	}
	// 归一化处理
	for (int c = 0; c != channels; c++)
	{
		for (int row = 0; row != dstRows; ++row)
		{
			for (int col = 0; col != dstCols; ++col)
			{
			
				dstData[row * dstStep + col * channels + c] = tempVals[row * dstStep + col * channels + c] / maxVal * 255;
			}
		}
	}

	dst = tmp;
	return phase;
}

// 将 LoG滤波分成x和y方向滤波的线性迭代，生成X方向和Y方向的核向量
std::pair<cv::Mat, cv::Mat> getLoGVector(int size, double sigma)
{
	//分配内存
	cv::Mat vectorX(cv::Size(size, 1), CV_32FC1);
	cv::Mat vectorY(cv::Size(1, size), CV_32FC1);
	double sqSigma = sigma * sigma;
	double base = std::exp(-1 / (2 * sqSigma));
	for (int i = 0; i < size; ++i)
	{
		double distance = (i - size / 2) * (i - size / 2);
		//vectorX.at<float>(i, 0) = std::pow(base, distance);
		vectorX.at<float>(i) = std::exp(-distance / (2*sqSigma));
		vectorY.at<float>(i) = (distance / sqSigma - 1) * vectorX.at<float>(i);
	}
	return std::make_pair(vectorX, vectorY);
	
}
void LoGFilter(const cv::Mat& src, cv::Mat& dst, int kernelSize, double sigma)
{
	auto vectors = getLoGVector(kernelSize, sigma);
	
	cv::Mat convXY, convYX;
	// 向X后Y
	cv::filter2D(src, convXY, CV_32FC1, vectors.first);
	cv::filter2D(convXY, convXY, CV_32FC1, vectors.second);

	// 先Y后X
	cv::filter2D(src, convYX, CV_32FC1, vectors.second.t());
	cv::filter2D(convYX, convYX, CV_32FC1, vectors.first.t());
	
	// 线性叠加
	cv::add(convXY, convYX, dst);
}

cv::Mat nms(const cv::Mat& intensity, const cv::Mat& gradient)
{
	uchar* srcData = intensity.data;
	int step = intensity.step;
	int srcRows = intensity.rows;
	int srcCols = intensity.cols;
	int channels = intensity.channels();

	cv::Mat res(intensity.clone());
	uchar* dstData = res.data;
	for (int c = 0; c < channels; ++c)
	{
		for (int row = 0; row < srcRows; ++row)
		{
			for (int col = 0; col < srcCols; ++col)
			{
				float dir = std::atan(gradient.at<float>(row, col)) * 180 / CV_PI;  // 获得梯度信息
				dir = dir < 0 ? dir + 180 : dir;
				uchar pixel1, pixel2;
				int rowOffset = !((dir >= 0 && dir <= 22.5) || (dir >= 157.5 && dir <= 180));
				int symbol = -1 + 2 * (dir >= 0 && dir <= 90);
				int colOffset = !(dir >= 67.5 && dir <= 112.5);
				// 沿梯度找到两个相邻像素
				pixel1 = boarderCornerPixel(srcData, step, channels, c,
					row - rowOffset, srcRows, col - symbol * colOffset, srcCols, BoarderType::CONSTANT, 0);
				pixel2 = boarderCornerPixel(srcData, step, channels, c,
					row + rowOffset, srcRows, col + symbol * colOffset, srcCols, BoarderType::CONSTANT, 0);
				uchar curPixel = srcData[row * step + col * channels + c];
				// 不满足条件，则置为0
				if (curPixel < pixel1 || curPixel < pixel2) dstData[row * step + col * channels + c] = 0;
			}
		}
	}
	return res;

}
// 双阈值
cv::Mat DualThreshold(const cv::Mat& src, double threshold1, double threshold2)
{
	uchar* srcData = src.data;
	int step = src.step;
	int srcRows = src.rows;
	int srcCols = src.cols;
	int channels = src.channels();

	cv::Mat res = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	uchar* dstData = res.data;
	for (int c = 0; c != channels; c++)
	{
		for (int row = 0; row != srcRows; ++row)
		{
			for (int col = 0; col != srcCols; ++col)
			{
				int idx = row * step + col * channels + c;
				if (srcData[idx] > threshold1) dstData[idx] = 255;
				else if (srcData[idx] > threshold2) dstData[idx] = 20;  // 这里置为20是为了后续链接的时候方便处理
			}
		}
	}
	return res;
}
cv::Mat trackHysteresis(const cv::Mat& src)
{
	uchar* srcData = src.data;
	int step = src.step;
	int srcRows = src.rows;
	int srcCols = src.cols;
	int channels = src.channels();

	cv::Mat res(src.clone());
	uchar* dstData = res.data;
	int width = 2;
	for (int c = 0; c != channels; c++)
	{
		for (int row = width; row != srcRows - width; ++row)
		{
			for (int col = width; col != srcCols - width; ++col)
			{
				if (srcData[row * step + col + channels + c] == 20)
				{
					bool finished = false;
					// 8 领域判断
					for (int i = -width; i != width + 1 && !finished; ++i)
					{
						for (int j = -width; j != width + 1 && !finished; ++j)
						{
							if (srcData[(row + i) * step + (col + j) * channels + c] == 255)
							{
								dstData[row * step + col + channels + c] == 255;
								finished = true;
							}
						}
					}

				}
			}
		}
	}
	return res;
}
void Canny(cv::Mat src, cv::Mat& edge, double threshold1, double threshold2,
	const GradOperator gradientOper)
{
	GaussFilter(src.clone(), src, cv::Size(3, 3), 1.6);
	cv::Mat gradient;
	cv::Mat res;
	cv::Mat intensity;
	cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	switch (gradientOper)
	{
	case GradOperator::SOBEL:
		gradient = Sobel(src, intensity);
		break;
	default:
		break;
	}
	res = nms(intensity, gradient);  // 非极大值抑制
	edge = DualThreshold(res, threshold1, threshold2);  // 双阈值处理第一步
	edge = trackHysteresis(edge);   // 双阈值第二步，处理介于阈值中间的像素
}
void HoughLines(cv::Mat edge, std::vector<cv::Vec2f>& lines,
	double rho, double theta, int threshold,
	double srn, double stn,
	double minTheta, double maxTheta)
{
	double maxRho =2* std::sqrt(std::pow(edge.rows, 2)+std::pow(edge.cols, 2));
	//maxRho = edge.rows + edge.cols;
	double lengthR = maxRho / rho, lengthTheta = maxTheta / theta;
	//std::cout << maxRho << "  " << lengthR << " " << lengthTheta << std::endl;
	std::vector<std::vector<int>> accumulator(lengthR, std::vector<int>(lengthTheta, 0));
	//std::transform()
	MatElements src(edge);
	for (int c = 0; c != src.channels; ++c)
	{
		for (int row = 0; row != src.rows; ++row)
		{
			for (int col = 0; col != src.cols; ++col)
			{
				if (*src.at(row, col, c) > 0)
				{
					for (double angle = minTheta; angle < maxTheta; angle += theta)
					{
						int indexR = col * std::cos(angle) + row * std::sin(angle) + maxRho / 2;
						int indexTheta = angle / theta;
						accumulator[indexR][indexTheta]++;

					}
				}
			}
		}
	}

	lines.clear();
	for (int r = 0; r != accumulator.size(); ++r)
	{
		for (int angle = 0; angle != accumulator[r].size(); ++angle)
		{
			//lines.emplace_back(r * rho, angle * theta);
			if (accumulator[r][angle] >= threshold) lines.emplace_back(r * rho - maxRho / 2, angle * theta);
		}
	}
	//}//std::for_each(accumulator.begin(), accumulator.end(), [](auto& vec) {std::sort(vec.begin(), vec.end()); });

	
}

void HoughCircles(cv::Mat image, std::vector<cv::Vec3f>& circles,
	int method, double dp, double minDist,
	double param1, double param2,
	int minRadius, int maxRadius)
{
	circles.clear();
	cv::Mat edge;
	cv::Canny(image, edge, param1 / 2, param1);

	MatElements src(edge), img(image);
	std::vector<std::vector<int>> accmulator(src.rows * src.cols, { 0, 0, 0 });
	std::vector<double> gradOper{1, 2, 1};
	// std::vector<double> gradOPer{4, 10, 4};
	for (int c = 0; c != src.channels; ++c)
	{
		for (int row = 1; row != src.rows - 1; ++row)
		{
			for (int col = 1; col != src.cols - 1; ++col)
			{
				if (/*row >= 150 && col >= 10 &&*/ *src.at(row, col, c) > 0)
				{
					double dx = gradOper[0] * *img.at(row + 1, col - 1, c) + gradOper[1] * *img.at(row + 1, col, c) + gradOper[2] * *img.at(row + 1, col + 1, c) - (gradOper[0] * *img.at(row - 1, col - 1, c) + gradOper[1] * *img.at(row - 1, col, c) + gradOper[2] * *img.at(row - 1, col + 1, c));
					double dy = gradOper[0] * *img.at(row - 1, col + 1, c) + gradOper[1] * *img.at(row, col + 1, c) + gradOper[2] * *img.at(row + 1, col + 1, c) - (gradOper[0] * *img.at(row - 1, row - 1, c) + gradOper[1] * *img.at(row, col - 1, c) + gradOper[2] * *img.at(row + 1, col - 1, c));
					double k = dy / dx;  // 获得梯度信息
					if (abs(dx) <= 1e-3)
					{
						for (int j = 0; j < src.cols; ++j)
						{
							if (accmulator[row * src.cols + j][2] == 0)
							{
								accmulator[row * src.cols + j][0] = row;
								accmulator[row * src.cols + j][1] = j;
								accmulator[row * src.cols + j][2] = 1;
							}
							else
								accmulator[row * src.cols + j][2] += 1;
						}
					}
					else
					{
						if (accmulator[row * src.cols + col][2] == 0)
						{
							accmulator[row * src.cols + col][0] = row;
							accmulator[row * src.cols + col][1] = col;
							accmulator[row * src.cols + col][2] = 1;
						}
						else
							accmulator[row * src.cols + col][2]++;
						for (int i = row + 1; i < src.rows && i < row + 50; ++i)
						{
							int j = round(i - row) * k + col;
							if (j >= src.cols || j < 0)
								break;
							if (accmulator[i * src.cols + j][2] == 0)
							{
								accmulator[i * src.cols + j][0] = i;
								accmulator[i * src.cols + j][1] = j;
								accmulator[i * src.cols + j][2] = 1;
							}
							else
								accmulator[i * src.cols + j][2]++;
						}
						for (int i = row - 1; i >= 0 && i > row - 50; --i)
						{
							int j = round(i - row) * k + col;

							if (j >= src.cols || j < 0)
								break;
							if (accmulator[i * src.cols + j][2] == 0)
							{
								accmulator[i * src.cols + j][0] = i;
								accmulator[i * src.cols + j][1] = j;
								accmulator[i * src.cols + j][2] = 1;
							}
							else
								accmulator[i * src.cols + j][2]++;
						}
					}
					// return;
				}
			}
		}
	}
	std::sort(accmulator.begin(), accmulator.end(), [](auto& lhs, auto& rhs)
		{ return lhs[2] > rhs[2]; });  // 排序，便于后期删除低于阈值的情况
	accmulator.erase(std::find_if(accmulator.begin(), accmulator.end(), [param2](auto& vec)
		{ return vec[2] < param2; }),
		accmulator.end());   // 剔除低于阈值的情况
	for (auto& point : accmulator)  // 对可能的圆心，判断距离是否过近，太近则只保留最大点
	{
		bool satisfied = true;
		for (int j = circles.size() - 1; j >= 0; j--)
		{
			if (std::sqrt(quadraticSum(point[0] - circles[j][0], point[1] - circles[j][1])) <= minDist)
			{
				satisfied = false;
				break;
			}
		}
		if (satisfied)
			circles.emplace_back(point[0], point[1], point[2]);
	}
	// 计算半径
	double maxLength = sqrt(quadraticSum(src.rows, src.cols));
	std::vector<std::vector<std::vector<int>>> radiusAccumulator(circles.size(), std::vector<std::vector<int>>(maxLength, std::vector<int>{0, 0}));
	for (int c = 0; c != src.channels; ++c)
	{
		for (int row = 0; row != src.rows; ++row)
		{
			for (int col = 0; col != src.cols; ++col)
			{
				if (*(src.at(row, col, c)))
				{
					for (int i = 0; i != circles.size(); ++i)
					{
						double radius = std::sqrt(quadraticSum(row - circles[i][0], col - circles[i][1]));
						radiusAccumulator[i][radius][0] = radius;
						radiusAccumulator[i][radius][1]++;
					}
				}
			}
		}
	}

	for (int i = 0; i < circles.size(); ++i)
	{
		circles[i][2] = (*std::max_element(radiusAccumulator[i].cbegin(), radiusAccumulator[i].cend(),
			[](auto& lhs, auto& rhs)
			{ return lhs[1] < rhs[1]; }))[0];
	}
}

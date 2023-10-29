#pragma once
#define _USE_MATH_DEFINES // 使用math.h中的M_PI宏定义需要
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

//#include "C1_Fundamental.h"
//#include "C2_GraytransAFilter.h"
#include "C3_FFT.h"
//#include "C4_EdgeDetection.h"

constexpr long double PI = M_PI;
constexpr long double PI_2 = M_PI_2;
constexpr long double E = M_E;


enum class BoarderType
{
	CONSTANT,
	NEAREST,
	REFLECT,
};




class StatisticType
{
public:
	explicit StatisticType(double mu, double sigma): _mu(mu), _sigma(sigma) {}
	double mu() const { return _mu; }
	double sigma() const { return _sigma; }

	friend std::ostream& operator<<(std::ostream& os, const StatisticType& rhs)
	{
		os << "Mu: " << rhs.mu() << " Sigma: " << rhs.sigma();
		return os;
	}
private:
	double _mu;
	double _sigma;
};


class POINT {
public:
	int x = 0;
	int y = 0;

	POINT() = default;
	POINT(const int x, const int y) : x(x), y(y) {}
	POINT(const cv::Point& p) : x(p.y), y(p.x) {}
	operator cv::Point() const { return cv::Point(y, x); }
};

class MAT {
public:
	friend std::ostream& operator<<(std::ostream& os, const MAT& mat)
	{
		for (auto& row : mat.mat) {
			for (auto ele : row) {
				os << ele << "\t";
			}
			os << "\n";
		}
		return os;
	}
	friend MAT crossProduct(const MAT& lhs, const MAT& rhs)
	{
		if (lhs.n != rhs.m)
		{
			std::cerr << "error ! mat1 must have the same n as mat2' m" << std::endl;
			return MAT(0, 0);
		}
		MAT res(lhs.m, rhs.n);
		for (int row = 0; row < res.m; ++row)
		{
			for (int col = 0; col < res.n; ++col)
			{
				double sum_res = 0;
				for (int i = 0; i < lhs.n; ++i)
				{
					sum_res += lhs[row][i] * rhs[i][col];
				}
				res[row][col] = sum_res;
			}
		}
		return res;
	}
	MAT(int M, int N) : m(M), n(N), mat(M, std::vector<double>(N)) {}
	MAT(int M, int N, double* array[]) : m(M), n(N) {
		for (size_t i = 0; i < m; i++) {
			std::vector<double> tmp;
			for (size_t j = 0; j < n; j++) {
				tmp.push_back(array[i][j]);
			}
			mat.push_back(tmp);
		}
	}
	MAT(const std::vector<std::vector<double>>& mmat)
		: m(mmat.size()), n(mmat.back().size()), mat(mmat) {}
	MAT(int M, int N, const std::vector<double>& mmat) : m(M), n(N), mat(m, std::vector<double>(n))
	{
		for (int row = 0; row != m; ++row)
		{
			for (int col = 0; col != n; ++col)
			{
				mat[row][col] = mmat[row * n + col];
			}
		}
	}
	void insert(uint pos, const std::vector<double>& row) {
		if (pos >= m || row.size() < n) {
			std::cerr << "mat is a " << m << " x " << n << " matrix" << std::endl;
			return;
		}
		mat[pos].assign(row.cbegin(), row.cbegin() + n);
	}
	void swap(size_t row1, size_t row2) { std::swap(mat[row1], mat[row2]); }
	void mul_k_to_row(size_t orig_row, size_t dst_row, double k) {
		for (size_t i = 0; i < n; ++i) {
			mat[dst_row][i] += mat[orig_row][i] * k;
		}
	}
	double sum() const
	{
		double res{ 0 };
		for (auto& row : mat)
		{
			for (auto& ele : row)
			{
				res += ele;
			}
		}
		return res;
	}

	std::vector<double>& operator[](size_t n) { return mat[n]; }
	const std::vector<double>& operator[](size_t n) const { return mat[n]; }

private:
	int m, n;
	std::vector<std::vector<double>> mat;
};

template<typename T>
class orderVector
{
public:
	using sizeType = typename std::vector<T>::size_type;
	orderVector() = default;
	orderVector(sizeType sz, bool isAsc = true)
		: v(sz), isAscend(isAsc)
	{

	}
	void push_back(T&& ele)
	{
		if (v.size() == 0) v.push_back(ele);
		else
		{
			v.push_back(ele);
			auto idx = v.size() - 1;
			for (int i = v.size() - 2; i >= 0; --i)
			{
				if (v[i] <= v[idx]) break;
				else
				{
					std::swap(v[i], v[idx]);
					idx = i;
				}
			}
		}
	}
	void clear()
	{
		v.clear();
	}
	const T at(sizeType idx) const
	{
		return v.at(idx);
	}
private:
	std::vector<T> v;
	bool isAscend = true;
};

//long double 

/**
 * @brief 矩阵数组存放类，负责保存一些常用的操作
 */
struct MatElements
{
	uchar* data = nullptr;
	int step;
	int channels;
	int rows;
	int cols;

	MatElements() = default;
	MatElements(const cv::Mat& mat)
		: data(mat.data), step(mat.step), channels(mat.channels()), rows(mat.rows), cols(mat.cols)
	{
	}
	void getData(cv::Mat& mat)
	{
		data = mat.data;
		step = mat.step;
		channels = mat.channels();
		rows = mat.rows;
		cols = mat.cols;
	}
	/**
	 * @brief 根据不同的边界类型，获得不一样的边界点
	 **/
	uchar boarderPixel(int row, int col, int c, const BoarderType& type, cv::Scalar constant)
	{
		switch (type)
		{
		case BoarderType::CONSTANT:
			if (row < 0 || row >= rows || col < 0 || col >= cols)
				return constant[c];
			else
				return *(data + row * step + channels * col + c);
			break;
		case BoarderType::NEAREST:
			col = std::max(col, 0);
			col = std::min(col, cols - 1);
			row = std::max(row, 0);
			row = std::min(row, rows - 1);
			return *(data + row * step + channels * col + c);
			break;
		case BoarderType::REFLECT:
			if (row < 0)
				row = 0 - row;
			else if (row >= rows)
				row = rows - (row - rows) - 1;
			if (col < 0)
				col = 0 - col;
			else if (col >= cols)
				col = cols - (col - cols) - 1;
			return *(data + row * step + channels * col + c);
			break;
		default:
			break;
		}
		return *(data + row * step + channels * col + c);
	}
	// 访问某个通道，某行，某列像素
	uchar* at(int row, int col, int channel) { return (data + row * step + channels * col + channel); }
	const uchar* at(int row, int col, int channel) const { return (data + row * step + channels * col + channel); }
};

/*
* 平方和
*/
inline double quadraticSum(double a, double b)
{
	return std::pow(a, 2) + std::pow(b, 2);
}

/**
* @ 将直线极坐标表达式换算成两点式
*/
inline std::pair<cv::Point2f, cv::Point2f> rhoTheta2Points(double rho, double theta, double length)
{
	cv::Point2f pt1, pt2;                                  //定义两点p1和p2
	double a = cos(theta), b = sin(theta);           //a:cos  b:sin
	//以x0和y0作为参照点，求出(x1, y1)和(x2, y2)
	double x0 = a * rho, y0 = b * rho;
	pt1.x = cvRound(x0 - length/2 * (-b));
	pt1.y = cvRound(y0 - length/2 * (a));
	pt2.x = cvRound(x0 + length/2 * (-b));
	pt2.y = cvRound(y0 + length/2 * (a));
	return std::make_pair(pt1, pt2);
}

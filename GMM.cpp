#include "GMM.h"
#include <vector>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
using namespace std;
using namespace cv;

Gauss::Gauss() {
	//初始值都设置为0
	mean = Vec3f(0, 0, 0);
	covmat.create(3, 3, CV_64FC1);
	covmat.setTo(0);
}
//计算高斯概率-hd
double Gauss::gauss(double u, double sigma, double x) {
	double t = (-0.5) * (x - u) * (x - u) / (sigma * sigma);
	return 1.0 / sigma * exp(t);
}
//向高斯模型中加入样例
void Gauss::addsample(Vec3f _color) {
	samples.push_back(_color);
}
//计算高斯模型中的均值和协方差-hd
void Gauss::learn() {
	//计算均值
	Vec3f sum = 0;
	int sz = (int)samples.size();
	for (int i = 0; i < sz; i++) sum += samples[i];
	mean = sum / sz;
	//并行化计算协方差
	cv::parallel_for_(cv::Range(0, 3), [&](const cv::Range& range) {
		for (int i = range.start; i < range.end; i++) {
			for (int j = 0; j < 3; j++) {
				double sum = 0;
				for (int cnt = 0; cnt < sz; cnt++) sum += (samples[cnt][i] - mean[i]) * (samples[cnt][j] - mean[j]);
				covmat.at<double>(i, j) = sum / sz;
			}
		}
		});
}
//计算高斯模型的概率

double Gauss::possibility(const Vec3f & mean, const Mat& covMat, Vec3f color) {
	// 计算 color 和 mean 的差值
	double diff[3];
	diff[0] = color[0] - mean[0];
	diff[1] = color[1] - mean[1];
	diff[2] = color[2] - mean[2];

	// 将差值数组转换为矩阵
	Mat diffMat = Mat(1, 3, CV_64FC1, diff);

	// 计算概率
	Mat ans = diffMat * covMat.inv() * diffMat.t();
	double mul = (-0.5) * ans.at<double>(0, 0);
	return 1.0 / sqrt(determinant(covMat)) * exp(mul);
}
//对两个参数进行离散化处理
void Gauss::discret(vector<double> &_sigma, vector<double> &_delta) {
	_sigma.clear();
	_delta.clear();
	for (double i = 0.1; i <= 6; i += (6.0 / 30)) {
		_delta.push_back(i);
	}
	for (double i = 0.1; i <= 2; i += (2.0 / 10)) {
		_sigma.push_back(i);
	}
}

//构造GMM，从 model 中读取参数并存储-hd
GMM::GMM(Mat& _model) {
	//GMM模型有13*K项数据，一个权重，三个均值和九个协方差
	//如果模型为空，则创建一个新的
	if (_model.empty())	{
		_model.create(1, 13*K, CV_64FC1);
		_model.setTo(Scalar(0));
	}
	model = _model;
	//存储顺序为权重、均值和协方差
	coefs = model.ptr<double>(0);
	mean = coefs+K;
	cov = mean+3*K;
	//如果某个项的权重不为0，则计算其协方差的逆和行列式
	for (int i = 0; i < K; ++i)
		if (coefs[i] > 0) calcuInvAndDet(i);
}
// 计算某个颜色属于某个高斯成分的可能性（高斯概率）-hd
double GMM::possibility(int componentIndex, const Vec3d color) const {
	double probability = 0;

	// 检查高斯成分的权重是否大于零
	if (coefs[componentIndex] > 0) {
		// 计算颜色与高斯成分均值的差异向量
		Vec3d diff = color;
		double* meanPtr = mean + 3 * componentIndex; // 获取高斯成分均值的指针
		diff[0] -= meanPtr[0];
		diff[1] -= meanPtr[1];
		diff[2] -= meanPtr[2];

		// 计算差异向量的平方形式
		double mult = diff[0] * (diff[0] * covInv[componentIndex][0][0] + diff[1] * covInv[componentIndex][1][0] + diff[2] * covInv[componentIndex][2][0])
			+ diff[1] * (diff[0] * covInv[componentIndex][0][1] + diff[1] * covInv[componentIndex][1][1] + diff[2] * covInv[componentIndex][2][1])
			+ diff[2] * (diff[0] * covInv[componentIndex][0][2] + diff[1] * covInv[componentIndex][1][2] + diff[2] * covInv[componentIndex][2][2]);

		// 计算高斯概率密度函数值
		probability = 1.0 / sqrt(covDet[componentIndex]) * exp(-0.5 * mult);
	}

	return probability;
}

//计算数据项的权重-hd
double GMM::totalWeight(const Vec3d _color)const{ 
	//求当前颜色属于当前 GMM 的总概率的加权和
	double res = 0;
	for (int ci = 0; ci < K; ci++)
		res += coefs[ci] * possibility(ci, _color);
	return res;
}
//计算颜色是属于哪个高斯概率最高的颜色项-hd
int GMM::choice(const Vec3d color) const {
	int k = 0;
	double max1= 0;
	for (int i=0;i<K;i++){
		double p = possibility(i, color);
		if (p>=max1){
			k=i;
			max1=p;
		}
	}
	return k;
}
//学习之前对数据进行初始化
void GMM::startLearning() {
	//对要用的中间变量赋0
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < 3; j++) //3代表三个颜色：RGB
			sums[i][j] = 0;
		for (int p = 0; p < 3; p++) {
			for (int q = 0; q < 3; q++) {
				prods[i][p][q] = 0;
			}
		}
		sampleCounts[i] = 0;
	}
	totalSampleCount = 0;
}
//添加单个的点
void GMM::addSample(int _i, const Vec3d _color) {
	//改变中间变量的值
	for (int i = 0; i < 3; i++) {
		sums[_i][i] += _color[i];
		for (int j = 0; j < 3; j++)
			prods[_i][i][j] += _color[i] * _color[j];
	}
	sampleCounts[_i]++;
	totalSampleCount++;
}

// GMM 模型学习结束函数，用于计算每个高斯分布的参数-hd
void GMM::finishLearning() {
	// 添加一个小的常数以避免数值问题，在计算协方差矩阵的过程中可能会遇到行列式接近0的情况
	const double variance = 0.01;

	// 使用 OpenMP 并行化处理，每个线程计算一个高斯分布的参数
#pragma omp parallel for
	for (int i = 0; i < K; i++) {
		// 获取第 i 个高斯分布的样本数量
		int n = sampleCounts[i];

		// 如果样本数量为 0，将该高斯分布的权重设为 0
		if (n == 0) {
			coefs[i] = 0;
		}
		else {
			// 计算高斯分布的权重，公式为样本数量除以总样本数量
			coefs[i] = 1.0 * n / totalSampleCount;

			// 计算均值
			double* m = mean + 3 * i;
			for (int j = 0; j < 3; j++) {
				m[j] = sums[i][j] / n;
			}

			// 计算协方差
			double* c = cov + 9 * i;
			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					c[p * 3 + q] = prods[i][p][q] / n - m[p] * m[q];
				}
			}

			// 计算协方差矩阵的行列式
			double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) -
				c[1] * (c[3] * c[8] - c[5] * c[6]) +
				c[2] * (c[3] * c[7] - c[4] * c[6]);

			// 如果行列式太小，加入一些噪音
			if (dtrm <= std::numeric_limits<double>::epsilon()) {
				c[0] += variance;
				c[4] += variance;
				c[8] += variance;
			}

			// 计算协方差矩阵的逆和行列式的值
			calcuInvAndDet(i);
		}
	}
}


// 计算协方差矩阵的逆和行列式的值-hd
void GMM::calcuInvAndDet(int _i) {
	if (coefs[_i] > 0) {
		double* c = cov + 9 * _i; // 指向第 _i 个高斯分布的协方差矩阵的指针

		// 计算行列式的值
		double dtrm = covDet[_i] = c[0] * (c[4] * c[8] - c[5] * c[7]) -
			c[1] * (c[3] * c[8] - c[5] * c[6]) +
			c[2] * (c[3] * c[7] - c[4] * c[6]);

		// 使用行列式计算协方差矩阵的逆矩阵
		covInv[_i][0][0] = (c[4] * c[8] - c[5] * c[7]) / dtrm;
		covInv[_i][1][0] = -(c[3] * c[8] - c[5] * c[6]) / dtrm;
		covInv[_i][2][0] = (c[3] * c[7] - c[4] * c[6]) / dtrm;

		covInv[_i][0][1] = -(c[1] * c[8] - c[2] * c[7]) / dtrm;
		covInv[_i][1][1] = (c[0] * c[8] - c[2] * c[6]) / dtrm;
		covInv[_i][2][1] = -(c[0] * c[7] - c[1] * c[6]) / dtrm;

		covInv[_i][0][2] = (c[1] * c[5] - c[2] * c[4]) / dtrm;
		covInv[_i][1][2] = -(c[0] * c[5] - c[2] * c[3]) / dtrm;
		covInv[_i][2][2] = (c[0] * c[4] - c[1] * c[3]) / dtrm;
	}
}




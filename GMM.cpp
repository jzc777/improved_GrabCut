#include "GMM.h"
#include <vector>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
using namespace std;
using namespace cv;

Gauss::Gauss() {
	//��ʼֵ������Ϊ0
	mean = Vec3f(0, 0, 0);
	covmat.create(3, 3, CV_64FC1);
	covmat.setTo(0);
}
//�����˹����-hd
double Gauss::gauss(double u, double sigma, double x) {
	double t = (-0.5) * (x - u) * (x - u) / (sigma * sigma);
	return 1.0 / sigma * exp(t);
}
//���˹ģ���м�������
void Gauss::addsample(Vec3f _color) {
	samples.push_back(_color);
}
//�����˹ģ���еľ�ֵ��Э����-hd
void Gauss::learn() {
	//�����ֵ
	Vec3f sum = 0;
	int sz = (int)samples.size();
	for (int i = 0; i < sz; i++) sum += samples[i];
	mean = sum / sz;
	//���л�����Э����
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
//�����˹ģ�͵ĸ���

double Gauss::possibility(const Vec3f & mean, const Mat& covMat, Vec3f color) {
	// ���� color �� mean �Ĳ�ֵ
	double diff[3];
	diff[0] = color[0] - mean[0];
	diff[1] = color[1] - mean[1];
	diff[2] = color[2] - mean[2];

	// ����ֵ����ת��Ϊ����
	Mat diffMat = Mat(1, 3, CV_64FC1, diff);

	// �������
	Mat ans = diffMat * covMat.inv() * diffMat.t();
	double mul = (-0.5) * ans.at<double>(0, 0);
	return 1.0 / sqrt(determinant(covMat)) * exp(mul);
}
//����������������ɢ������
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

//����GMM���� model �ж�ȡ�������洢-hd
GMM::GMM(Mat& _model) {
	//GMMģ����13*K�����ݣ�һ��Ȩ�أ�������ֵ�;Ÿ�Э����
	//���ģ��Ϊ�գ��򴴽�һ���µ�
	if (_model.empty())	{
		_model.create(1, 13*K, CV_64FC1);
		_model.setTo(Scalar(0));
	}
	model = _model;
	//�洢˳��ΪȨ�ء���ֵ��Э����
	coefs = model.ptr<double>(0);
	mean = coefs+K;
	cov = mean+3*K;
	//���ĳ�����Ȩ�ز�Ϊ0���������Э������������ʽ
	for (int i = 0; i < K; ++i)
		if (coefs[i] > 0) calcuInvAndDet(i);
}
// ����ĳ����ɫ����ĳ����˹�ɷֵĿ����ԣ���˹���ʣ�-hd
double GMM::possibility(int componentIndex, const Vec3d color) const {
	double probability = 0;

	// ����˹�ɷֵ�Ȩ���Ƿ������
	if (coefs[componentIndex] > 0) {
		// ������ɫ���˹�ɷ־�ֵ�Ĳ�������
		Vec3d diff = color;
		double* meanPtr = mean + 3 * componentIndex; // ��ȡ��˹�ɷ־�ֵ��ָ��
		diff[0] -= meanPtr[0];
		diff[1] -= meanPtr[1];
		diff[2] -= meanPtr[2];

		// �������������ƽ����ʽ
		double mult = diff[0] * (diff[0] * covInv[componentIndex][0][0] + diff[1] * covInv[componentIndex][1][0] + diff[2] * covInv[componentIndex][2][0])
			+ diff[1] * (diff[0] * covInv[componentIndex][0][1] + diff[1] * covInv[componentIndex][1][1] + diff[2] * covInv[componentIndex][2][1])
			+ diff[2] * (diff[0] * covInv[componentIndex][0][2] + diff[1] * covInv[componentIndex][1][2] + diff[2] * covInv[componentIndex][2][2]);

		// �����˹�����ܶȺ���ֵ
		probability = 1.0 / sqrt(covDet[componentIndex]) * exp(-0.5 * mult);
	}

	return probability;
}

//�����������Ȩ��-hd
double GMM::totalWeight(const Vec3d _color)const{ 
	//��ǰ��ɫ���ڵ�ǰ GMM ���ܸ��ʵļ�Ȩ��
	double res = 0;
	for (int ci = 0; ci < K; ci++)
		res += coefs[ci] * possibility(ci, _color);
	return res;
}
//������ɫ�������ĸ���˹������ߵ���ɫ��-hd
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
//ѧϰ֮ǰ�����ݽ��г�ʼ��
void GMM::startLearning() {
	//��Ҫ�õ��м������0
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < 3; j++) //3����������ɫ��RGB
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
//��ӵ����ĵ�
void GMM::addSample(int _i, const Vec3d _color) {
	//�ı��м������ֵ
	for (int i = 0; i < 3; i++) {
		sums[_i][i] += _color[i];
		for (int j = 0; j < 3; j++)
			prods[_i][i][j] += _color[i] * _color[j];
	}
	sampleCounts[_i]++;
	totalSampleCount++;
}

// GMM ģ��ѧϰ�������������ڼ���ÿ����˹�ֲ��Ĳ���-hd
void GMM::finishLearning() {
	// ���һ��С�ĳ����Ա�����ֵ���⣬�ڼ���Э�������Ĺ����п��ܻ���������ʽ�ӽ�0�����
	const double variance = 0.01;

	// ʹ�� OpenMP ���л�����ÿ���̼߳���һ����˹�ֲ��Ĳ���
#pragma omp parallel for
	for (int i = 0; i < K; i++) {
		// ��ȡ�� i ����˹�ֲ�����������
		int n = sampleCounts[i];

		// �����������Ϊ 0�����ø�˹�ֲ���Ȩ����Ϊ 0
		if (n == 0) {
			coefs[i] = 0;
		}
		else {
			// �����˹�ֲ���Ȩ�أ���ʽΪ����������������������
			coefs[i] = 1.0 * n / totalSampleCount;

			// �����ֵ
			double* m = mean + 3 * i;
			for (int j = 0; j < 3; j++) {
				m[j] = sums[i][j] / n;
			}

			// ����Э����
			double* c = cov + 9 * i;
			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					c[p * 3 + q] = prods[i][p][q] / n - m[p] * m[q];
				}
			}

			// ����Э������������ʽ
			double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) -
				c[1] * (c[3] * c[8] - c[5] * c[6]) +
				c[2] * (c[3] * c[7] - c[4] * c[6]);

			// �������ʽ̫С������һЩ����
			if (dtrm <= std::numeric_limits<double>::epsilon()) {
				c[0] += variance;
				c[4] += variance;
				c[8] += variance;
			}

			// ����Э���������������ʽ��ֵ
			calcuInvAndDet(i);
		}
	}
}


// ����Э���������������ʽ��ֵ-hd
void GMM::calcuInvAndDet(int _i) {
	if (coefs[_i] > 0) {
		double* c = cov + 9 * _i; // ָ��� _i ����˹�ֲ���Э��������ָ��

		// ��������ʽ��ֵ
		double dtrm = covDet[_i] = c[0] * (c[4] * c[8] - c[5] * c[7]) -
			c[1] * (c[3] * c[8] - c[5] * c[6]) +
			c[2] * (c[3] * c[7] - c[4] * c[6]);

		// ʹ������ʽ����Э�������������
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




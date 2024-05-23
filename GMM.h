#ifndef GMM_H_
#define GMM_H_
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>

class Gauss {
public:
	Gauss();
	static double gauss(const double, const double, const double);
	static double possibility(const cv::Vec3f&, const cv::Mat&, cv::Vec3f);//算高斯模型的概率
	static void discret(std::vector<double>&, std::vector<double>&);  // 对两个参数进行离散化处理
	void addsample(cv::Vec3f);  // 向高斯模型中添加样本
	void learn();  // 根据样本计算高斯模型的均值和协方差
	cv::Vec3f getmean()const { return mean; }
	cv::Mat getcovmat()const { return covmat; }
private:
	cv::Vec3f mean;
	cv::Mat covmat;
	std::vector<cv::Vec3f> samples;

};

class GMM {
public:

	static const int K = 5;//GMM数量
	GMM(cv::Mat& _model);//从model读取参数并存储
	double possibility(int, const cv::Vec3d) const;  // 计算某个颜色属于某个组件的概率
	double totalWeight(const cv::Vec3d) const;  // 计算数据项的权重
	int choice(const cv::Vec3d) const;  // 颜色属于哪个组件（概率最高的项）
	void startLearning();  
	void addSample(int, const cv::Vec3d);  // 添加单个样本点
	void finishLearning();  
private:
	void calcuInvAndDet(int);//求协方差矩阵的逆和行列式的值
	cv::Mat model;
	double* coefs, * mean, * cov;//权重、均值和协方差
	double covInv[K][3][3];	//协方差的逆
	double covDet[K];	//行列式值
	double sums[K][3];
	double prods[K][3][3];
	int sampleCounts[K];
	int totalSampleCount;
};
#endif

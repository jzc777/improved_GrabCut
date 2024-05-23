#ifndef GMM_H_
#define GMM_H_
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>

class Gauss {
public:
	Gauss();
	static double gauss(const double, const double, const double);
	static double possibility(const cv::Vec3f&, const cv::Mat&, cv::Vec3f);//���˹ģ�͵ĸ���
	static void discret(std::vector<double>&, std::vector<double>&);  // ����������������ɢ������
	void addsample(cv::Vec3f);  // ���˹ģ�����������
	void learn();  // �������������˹ģ�͵ľ�ֵ��Э����
	cv::Vec3f getmean()const { return mean; }
	cv::Mat getcovmat()const { return covmat; }
private:
	cv::Vec3f mean;
	cv::Mat covmat;
	std::vector<cv::Vec3f> samples;

};

class GMM {
public:

	static const int K = 5;//GMM����
	GMM(cv::Mat& _model);//��model��ȡ�������洢
	double possibility(int, const cv::Vec3d) const;  // ����ĳ����ɫ����ĳ������ĸ���
	double totalWeight(const cv::Vec3d) const;  // �����������Ȩ��
	int choice(const cv::Vec3d) const;  // ��ɫ�����ĸ������������ߵ��
	void startLearning();  
	void addSample(int, const cv::Vec3d);  // ��ӵ���������
	void finishLearning();  
private:
	void calcuInvAndDet(int);//��Э���������������ʽ��ֵ
	cv::Mat model;
	double* coefs, * mean, * cov;//Ȩ�ء���ֵ��Э����
	double covInv[K][3][3];	//Э�������
	double covDet[K];	//����ʽֵ
	double sums[K][3];
	double prods[K][3][3];
	int sampleCounts[K];
	int totalSampleCount;
};
#endif

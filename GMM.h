#ifndef GMM_H_
#define GMM_H_
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>

class Gauss {
public:
	Gauss();
	static double gauss(const double, const double, const double);
	//���˹ģ�͵ĸ���
	static double possibility(const cv::Vec3f&, const cv::Mat&, cv::Vec3f);

	//����������ɢ������
	static void discret(std::vector<double>&, std::vector<double>&);//delta range from [0,6], sigma range from [0,delta/3]
	//���˹ģ���м���һ������
	void addsample(cv::Vec3f);
	//��������ģ�ͣ������˹ģ���еľ�ֵ��Э����
	void learn();
	cv::Vec3f getmean()const { return mean; }
	cv::Mat getcovmat()const { return covmat; }
private:
	cv::Vec3f mean;	//��˹��ֵ
	cv::Mat covmat;//Э����
	std::vector<cv::Vec3f> samples;
	
};

class GMM {
public:
	//��˹ģ�͵����������������е�ʵ�֣�Ϊ5
	static const int K = 5;
	//GMM�Ĺ��캯������ model �ж�ȡ�������洢
	GMM(cv::Mat& _model);
	//����ĳ����ɫ����ĳ������Ŀ����ԣ���˹���ʣ�
	double possibility(int, const cv::Vec3d) const;
	//����������Ȩ��
	double totalWeight(const cv::Vec3d) const;
	//����һ����ɫӦ���������ĸ��������˹������ߵ��
	int choice(const cv::Vec3d) const;
	//ѧϰ֮ǰ�����ݽ��г�ʼ��
	void startLearning();
	//��ӵ����ĵ�
	void addSample(int, const cv::Vec3d);
	//������ӵ����ݣ������µĲ������
	void finishLearning();
private:
	//����Э���������������ʽ��ֵ
	void calcuInvAndDet(int);
	//�洢GMMģ��
	cv::Mat model;
	//GMMģ���У�ÿ����˹�ֲ���Ȩ�ء���ֵ��Э����
	double *coefs, *mean, *cov;
	//�洢Э������棬���ڼ���
	double covInv[K][3][3];
	//�洢Э���������ʽֵ
	double covDet[K];
	//����ѧϰ�����б����м����ݵı���
	double sums[K][3];
	double prods[K][3][3];
	int sampleCounts[K];
	int totalSampleCount;
};
#endif

#include "GMM.h"
#ifndef CUTGRAPH_H_
#define CUTGRAPH_H_
#include "graph.h"
#include <iostream>
#include <limits>
#include <vector>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
using namespace cv;
using namespace std;

class GraphCut {
private:
	Graph<double, double, double>* graph;
public:
	GraphCut();
	GraphCut(int, int);
	int addVertex();
	double maxFlow();
	void addVertexWeights(int, double, double);
	void addEdges(int, int, double);
	bool isSourceSegment(int);
};

enum
{
	GC_WITH_RECT = 0,
	GC_WITH_MASK = 1,
	GC_CUT = 2
};
enum { //���ֱ�ǩ����
	MUST_BGD = 0,
	MUST_FGD = 1,
	MAYBE_BGD = 2,
	MAYBE_FGD = 3
};

class GrabCut2D
{
public:
	void GrabCut(cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel, cv::InputOutputArray _fgdModel,
		int iterCount, int mode);

	~GrabCut2D(void);
};



// �������ĵĹ�ʽ5�õ�Betaֵ-hd
static double calculateBeta(const Mat& _img) {
	double beta;
	double totalDiff = 0;

	// ����ͼ���ÿ�����أ�����ÿ������������������֮�����ɫ�����ƽ����
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x); // ��ǰ���ص���ɫֵ

			// ����
			if (x > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y, x - 1);
				totalDiff += diff.dot(diff);
			}

			// ���Ϸ���
			if (y > 0 && x > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x - 1);
				totalDiff += diff.dot(diff);
			}

			// �Ϸ���
			if (y > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x);
				totalDiff += diff.dot(diff);
			}

			// ���Ϸ���
			if (y > 0 && x < _img.cols - 1) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x + 1);
				totalDiff += diff.dot(diff);
			}
		}
	}

	// ���������еĹ�ʽ�������ܵ���ɫ����ƽ���͵�����
	totalDiff *= 2;

	// ����ܵ���ɫ����ƽ���ͺ�С�������� beta Ϊ 0��������� 0 �����
	if (totalDiff <= std::numeric_limits<double>::epsilon()) {
		beta = 0;
	}
	else {
		// ���򣬼��� beta ��ֵ
		// ��ʽ: beta = 1 / (2 * mean(color difference^2))
		// ���� mean(color difference^2) = totalDiff / (8 * number_of_pixels - 6 * (width + height) + 4)
		beta = 1.0 / (2 * totalDiff / (8 * _img.cols * _img.rows - 6 * _img.cols - 6 * _img.rows + 4));
	}

	return beta;
}

// ����һ�������Ȩ��
double calculateWeight(const Vec3d& color1, const Vec3d& color2, double beta, double gamma) {
	Vec3d diff = color1 - color2;
	return gamma * exp(-beta * diff.dot(diff));
}

// �����������ص�Ȩ�ز����µľ������洢���ⷽ���Ȩ��
static void calculateNeighborWeights(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, Mat& _r, Mat& _dr, Mat& _d, Mat& _dl, double _beta, double _gamma) {
	const double gammaDiv = _gamma / std::sqrt(2.0);

	// ����������󣬴�С������ͼ����ͬ������Ϊ˫���ȸ�����
	_l.create(_img.size(), CV_64FC1);
	_ul.create(_img.size(), CV_64FC1);
	_u.create(_img.size(), CV_64FC1);
	_ur.create(_img.size(), CV_64FC1);
	_r.create(_img.size(), CV_64FC1);
	_dr.create(_img.size(), CV_64FC1);
	_d.create(_img.size(), CV_64FC1);
	_dl.create(_img.size(), CV_64FC1);

	// ���л�����ÿ�����أ�����˸������Ȩ�ز�
	cv::parallel_for_(cv::Range(0, _img.rows * _img.cols), [&](const cv::Range& range) {
		for (int idx = range.start; idx < range.end; idx++) {
			int y = idx / _img.cols;
			int x = idx % _img.cols;
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x);

			// ����
			if (x - 1 >= 0) {
				_l.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y, x - 1), _beta, _gamma);
			}
			else {
				_l.at<double>(y, x) = 0;
			}

			// ���Ϸ���
			if (x - 1 >= 0 && y - 1 >= 0) {
				_ul.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y - 1, x - 1), _beta, gammaDiv);
			}
			else {
				_ul.at<double>(y, x) = 0;
			}

			// �Ϸ���
			if (y - 1 >= 0) {
				_u.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y - 1, x), _beta, _gamma);
			}
			else {
				_u.at<double>(y, x) = 0;
			}

			// ���Ϸ���
			if (x + 1 < _img.cols && y - 1 >= 0) {
				_ur.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y - 1, x + 1), _beta, gammaDiv);
			}
			else {
				_ur.at<double>(y, x) = 0;
			}

			// �ҷ���
			if (x + 1 < _img.cols) {
				_r.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y, x + 1), _beta, _gamma);
			}
			else {
				_r.at<double>(y, x) = 0;
			}

			// ���·���
			if (x + 1 < _img.cols && y + 1 < _img.rows) {
				_dr.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y + 1, x + 1), _beta, gammaDiv);
			}
			else {
				_dr.at<double>(y, x) = 0;
			}

			// �·���
			if (y + 1 < _img.rows) {
				_d.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y + 1, x), _beta, _gamma);
			}
			else {
				_d.at<double>(y, x) = 0;
			}

			// ���·���
			if (x - 1 >= 0 && y + 1 < _img.rows) {
				_dl.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y + 1, x - 1), _beta, gammaDiv);
			}
			else {
				_dl.at<double>(y, x) = 0;
			}
		}
		});
}

// �����������ص�Ȩ�ز�����˸�����
static void calculateNeighborWeights(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma) {
	Mat _r, _dr, _d, _dl;
	calculateNeighborWeights(_img, _l, _ul, _u, _ur, _r, _dr, _d, _dl, _beta, _gamma);
}

// ��������������� mask��������϶��Ǳ����������ڿ�����ǰ����-hd
static void initMaskWithRect(Mat& _mask, Size _imgSize, Rect _rect) {
	// ����һ����ͼ���С��ͬ��������󣬳�ʼֵ����Ϊȷ������ (MUST_BGD)
	_mask.create(_imgSize, CV_8UC1);
	_mask.setTo(MUST_BGD);

	// ȷ�����ο����ʼ����ͼ��Χ��
	if (_rect.x < 0) {
		_rect.x = 0;
	}
	if (_rect.y < 0) {
		_rect.y = 0;
	}

	// ȷ�����ο�Ŀ�Ⱥ͸߶���ͼ��Χ��
	if (_rect.x + _rect.width > _imgSize.width) {
		_rect.width = _imgSize.width - _rect.x;
	}
	if (_rect.y + _rect.height > _imgSize.height) {
		_rect.height = _imgSize.height - _rect.y;
	}

	// �����ο��ڵ���������Ϊ����ǰ�� (MAYBE_FGD)
	_mask(_rect).setTo(Scalar(MAYBE_FGD));
}

// ���� kmeans ������ʼ�� GMM ģ��-hd
static void initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM) {
	const int kmeansItCount = 5; // k-means �ĵ���������Ϊ5
	Mat bgdLabels, fgdLabels; // ���ڴ洢������ǰ�������ı�ǩ����
	vector<Vec3f> bgdSamples, fgdSamples; // ���ڴ洢������ǰ������������
	Point p;

	// ����ͼ���ÿ������
	for (p.y = 0; p.y < img.rows; p.y++) {
		for (p.x = 0; p.x < img.cols; p.x++) {
			// �����Ĥ mask �еĶ�Ӧ����ֵΪ������MUST_BGD �� MAYBE_BGD��
			if (mask.at<uchar>(p) == MUST_BGD || mask.at<uchar>(p) == MAYBE_BGD)
				bgdSamples.push_back((Vec3f)img.at<Vec3b>(p)); // �������ص���ɫֵ���� bgdSamples
			else
				fgdSamples.push_back((Vec3f)img.at<Vec3b>(p)); // ������� fgdSamples
		}
	}

	// ����������ת��Ϊ Mat �������� k-means
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	// �Ա����������� k-means ����
	kmeans(_bgdSamples, GMM::K, bgdLabels,
		TermCriteria(TermCriteria::COUNT, kmeansItCount, 0.0), 0, KMEANS_PP_CENTERS);

	// ��ǰ������ת��Ϊ Mat �������� k-means
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	// ��ǰ���������� k-means ����
	kmeans(_fgdSamples, GMM::K, fgdLabels,
		TermCriteria(TermCriteria::COUNT, kmeansItCount, 0.0), 0, KMEANS_PP_CENTERS);

	// ѧϰ���� GMM ģ��
	bgdGMM.startLearning();
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	bgdGMM.finishLearning();

	// ѧϰǰ�� GMM ģ��
	fgdGMM.startLearning();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.finishLearning();
}


//�ϲ���������1�Ͳ���2����ÿ�����ط��䵽GMM�������ĸ�˹ģ�ͣ�������GMM����ֵ��
static void assignAndLearnGMMS(const Mat& _img, const Mat& _mask, GMM& _bgdGMM, GMM& _fgdGMM, Mat& _partIndex) {
	_bgdGMM.startLearning();
	_fgdGMM.startLearning();

	Point p;
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(p);
			uchar t = _mask.at<uchar>(p);
			int componentIndex;

			// ������ p ���ڱ���GMM ģ���е��ĸ���˹�ɷֵ������洢�� _partIndex ������
			if (t == MUST_BGD || t == MAYBE_BGD) {
				componentIndex = _bgdGMM.choice(color);
				_partIndex.at<int>(p) = componentIndex;
				_bgdGMM.addSample(componentIndex, _img.at<Vec3b>(p));
			}
			else {
				componentIndex = _fgdGMM.choice(color);
				_partIndex.at<int>(p) = componentIndex;
				_fgdGMM.addSample(componentIndex, _img.at<Vec3b>(p));
			}
		}
	}

	_bgdGMM.finishLearning();
	_fgdGMM.finishLearning();
}


//���ݵõ��Ľ������ͼ��ʹ�����̸����ֳɵĿ� Done
static void getGraph(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur, GraphCut& _graph) {
	int vCount = _img.cols*_img.rows;
	int eCount = 2 * (4 * vCount - 3 * _img.cols - 3 * _img.rows + 2);
	_graph = GraphCut(vCount, eCount);
	Point p;
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			int vNum = _graph.addVertex();
			Vec3b color = _img.at<Vec3b>(p);
			double wSource = 0, wSink = 0;
			if (_mask.at<uchar>(p) == MAYBE_BGD || _mask.at<uchar>(p) == MAYBE_FGD) {
				wSource = -log(_bgdGMM.totalWeight(color)); //�����������ڱ����ĸ��ʡ�
				wSink = -log(_fgdGMM.totalWeight(color));  //������������ǰ���ĸ��ʡ�
			}
			else if (_mask.at<uchar>(p) == MUST_BGD) wSink = _lambda;
			else wSource = _lambda;
			_graph.addVertexWeights(vNum, wSource, wSink);
			if (p.x > 0) {
				double w = _l.at<double>(p);
				_graph.addEdges(vNum, vNum - 1, w);
			}
			if (p.x > 0 && p.y > 0) {
				double w = _ul.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols - 1, w);
			}
			if (p.y > 0) {
				double w = _u.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols, w);
			}
			if (p.x < _img.cols - 1 && p.y > 0) {
				double w = _ur.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols + 1, w);
			}
		}
	}
}
// ���зָ�-hd
static void estimateSegmentation(GraphCut& _graph, Mat& _mask) {
	// ���������
	_graph.maxFlow();

	// ������������е�ÿ������
	Point p;
	for (p.y = 0; p.y < _mask.rows; p.y++) {
		for (p.x = 0; p.x < _mask.cols; p.x++) {
			// ��������еĵ�ǰ������ MAYBE_BGD �� MAYBE_FGD
			if (_mask.at<uchar>(p) == MAYBE_BGD || _mask.at<uchar>(p) == MAYBE_FGD) {
				// ���������Ƿ�����Դ��
				if (_graph.isSourceSegment(p.y * _mask.cols + p.x)) {
					// ����ǣ��������е�����ֵ��Ϊ MAYBE_FGD
					_mask.at<uchar>(p) = MAYBE_FGD;
				}
				else {
					// ������ǣ��������е�����ֵ��Ϊ MAYBE_BGD
					_mask.at<uchar>(p) = MAYBE_BGD;
				}
			}
		}
	}
}

GrabCut2D::~GrabCut2D(void) {}
//GrabCut����-hd
void GrabCut2D::GrabCut(InputArray _img, InputOutputArray _mask, Rect rect,
	InputOutputArray _bgdModel, InputOutputArray _fgdModel,
	int iterCount, int mode) {

	// ��������ͼ��
	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();

	// 1. ��Сͼ��������Լӿ촦���ٶ�
	Mat smallImg, smallMask;
	float scaleFactor = 1.1; // ������Ҫ������������
	resize(img, smallImg, Size(img.cols / scaleFactor, img.rows / scaleFactor));
	resize(mask, smallMask, Size(mask.cols / scaleFactor, mask.rows / scaleFactor), 0, 0, INTER_NEAREST);

	// ��ʼ�����룬��ģʽΪ GC_WITH_RECT����ʹ�þ��ο�
	if (mode == GC_WITH_RECT) {
		Rect smallRect(rect.x / scaleFactor, rect.y / scaleFactor, rect.width / scaleFactor, rect.height / scaleFactor);
		initMaskWithRect(smallMask, smallImg.size(), smallRect);
	}

	// 2. ��ʼ�� GMM ģ��
	GMM bgdGMM(bgdModel), fgdGMM(fgdModel);

	// ʹ�� k-means ��ʼ�� GMM
	if (mode == GC_WITH_RECT || mode == GC_WITH_MASK) {
		initGMMs(smallImg, smallMask, bgdGMM, fgdGMM);
	}

	if (iterCount <= 0) return;

	// ����ͼ�ı�Ȩ��
	const double gamma = 50;
	const double beta = calculateBeta(smallImg);
	Mat leftW, upleftW, upW, uprightW;
	calculateNeighborWeights(smallImg, leftW, upleftW, upW, uprightW, beta, gamma);

	// 3. ������ȡ����mask
	Mat compIdxs(smallImg.size(), CV_32SC1);//��ͨ��32λ
	const double lambda = 100;
	for (int i = 0; i < iterCount; i++) {
		GraphCut graph;
		assignAndLearnGMMS(smallImg, smallMask, bgdGMM, fgdGMM, compIdxs);
		//learnGMMs(smallImg, smallMask, bgdGMM, fgdGMM, compIdxs);
		getGraph(smallImg, smallMask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
		estimateSegmentation(graph, smallMask);
	}

	// 4. ������������Ŵ��ԭʼ��С
	resize(smallMask, mask, mask.size(), 0, 0, INTER_NEAREST);
}


#endif

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
enum { //四种标签分类
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



// 根据论文的公式5得到Beta值-hd
static double calculateBeta(const Mat& _img) {
	double beta;
	double totalDiff = 0;

	// 遍历图像的每个像素，计算每个像素与其相邻像素之间的颜色差异的平方和
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x); // 当前像素的颜色值

			// 左方向
			if (x > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y, x - 1);
				totalDiff += diff.dot(diff);
			}

			// 左上方向
			if (y > 0 && x > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x - 1);
				totalDiff += diff.dot(diff);
			}

			// 上方向
			if (y > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x);
				totalDiff += diff.dot(diff);
			}

			// 右上方向
			if (y > 0 && x < _img.cols - 1) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x + 1);
				totalDiff += diff.dot(diff);
			}
		}
	}

	// 根据论文中的公式，计算总的颜色差异平方和的两倍
	totalDiff *= 2;

	// 如果总的颜色差异平方和很小，则设置 beta 为 0，避免除以 0 的情况
	if (totalDiff <= std::numeric_limits<double>::epsilon()) {
		beta = 0;
	}
	else {
		// 否则，计算 beta 的值
		// 公式: beta = 1 / (2 * mean(color difference^2))
		// 其中 mean(color difference^2) = totalDiff / (8 * number_of_pixels - 6 * (width + height) + 4)
		beta = 1.0 / (2 * totalDiff / (8 * _img.cols * _img.rows - 6 * _img.cols - 6 * _img.rows + 4));
	}

	return beta;
}

// 计算一个方向的权重
double calculateWeight(const Vec3d& color1, const Vec3d& color2, double beta, double gamma) {
	Vec3d diff = color1 - color2;
	return gamma * exp(-beta * diff.dot(diff));
}

// 计算相邻像素的权重差，添加新的矩阵来存储额外方向的权重
static void calculateNeighborWeights(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, Mat& _r, Mat& _dr, Mat& _d, Mat& _dl, double _beta, double _gamma) {
	const double gammaDiv = _gamma / std::sqrt(2.0);

	// 创建输出矩阵，大小与输入图像相同，类型为双精度浮点数
	_l.create(_img.size(), CV_64FC1);
	_ul.create(_img.size(), CV_64FC1);
	_u.create(_img.size(), CV_64FC1);
	_ur.create(_img.size(), CV_64FC1);
	_r.create(_img.size(), CV_64FC1);
	_dr.create(_img.size(), CV_64FC1);
	_d.create(_img.size(), CV_64FC1);
	_dl.create(_img.size(), CV_64FC1);

	// 并行化遍历每个像素，计算八个方向的权重差
	cv::parallel_for_(cv::Range(0, _img.rows * _img.cols), [&](const cv::Range& range) {
		for (int idx = range.start; idx < range.end; idx++) {
			int y = idx / _img.cols;
			int x = idx % _img.cols;
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x);

			// 左方向
			if (x - 1 >= 0) {
				_l.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y, x - 1), _beta, _gamma);
			}
			else {
				_l.at<double>(y, x) = 0;
			}

			// 左上方向
			if (x - 1 >= 0 && y - 1 >= 0) {
				_ul.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y - 1, x - 1), _beta, gammaDiv);
			}
			else {
				_ul.at<double>(y, x) = 0;
			}

			// 上方向
			if (y - 1 >= 0) {
				_u.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y - 1, x), _beta, _gamma);
			}
			else {
				_u.at<double>(y, x) = 0;
			}

			// 右上方向
			if (x + 1 < _img.cols && y - 1 >= 0) {
				_ur.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y - 1, x + 1), _beta, gammaDiv);
			}
			else {
				_ur.at<double>(y, x) = 0;
			}

			// 右方向
			if (x + 1 < _img.cols) {
				_r.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y, x + 1), _beta, _gamma);
			}
			else {
				_r.at<double>(y, x) = 0;
			}

			// 右下方向
			if (x + 1 < _img.cols && y + 1 < _img.rows) {
				_dr.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y + 1, x + 1), _beta, gammaDiv);
			}
			else {
				_dr.at<double>(y, x) = 0;
			}

			// 下方向
			if (y + 1 < _img.rows) {
				_d.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y + 1, x), _beta, _gamma);
			}
			else {
				_d.at<double>(y, x) = 0;
			}

			// 左下方向
			if (x - 1 >= 0 && y + 1 < _img.rows) {
				_dl.at<double>(y, x) = calculateWeight(color, (Vec3d)_img.at<Vec3b>(y + 1, x - 1), _beta, gammaDiv);
			}
			else {
				_dl.at<double>(y, x) = 0;
			}
		}
		});
}

// 计算相邻像素的权重差，包括八个方向。
static void calculateNeighborWeights(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma) {
	Mat _r, _dr, _d, _dl;
	calculateNeighborWeights(_img, _l, _ul, _u, _ur, _r, _dr, _d, _dl, _beta, _gamma);
}

// 根据输入矩阵设置 mask，矩阵外肯定是背景，矩阵内可能是前景。-hd
static void initMaskWithRect(Mat& _mask, Size _imgSize, Rect _rect) {
	// 创建一个与图像大小相同的掩码矩阵，初始值设置为确定背景 (MUST_BGD)
	_mask.create(_imgSize, CV_8UC1);
	_mask.setTo(MUST_BGD);

	// 确保矩形框的起始点在图像范围内
	if (_rect.x < 0) {
		_rect.x = 0;
	}
	if (_rect.y < 0) {
		_rect.y = 0;
	}

	// 确保矩形框的宽度和高度在图像范围内
	if (_rect.x + _rect.width > _imgSize.width) {
		_rect.width = _imgSize.width - _rect.x;
	}
	if (_rect.y + _rect.height > _imgSize.height) {
		_rect.height = _imgSize.height - _rect.y;
	}

	// 将矩形框内的区域设置为可能前景 (MAYBE_FGD)
	_mask(_rect).setTo(Scalar(MAYBE_FGD));
}

// 利用 kmeans 方法初始化 GMM 模型-hd
static void initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM) {
	const int kmeansItCount = 5; // k-means 的迭代次数设为5
	Mat bgdLabels, fgdLabels; // 用于存储背景和前景样本的标签矩阵
	vector<Vec3f> bgdSamples, fgdSamples; // 用于存储背景和前景的像素样本
	Point p;

	// 遍历图像的每个像素
	for (p.y = 0; p.y < img.rows; p.y++) {
		for (p.x = 0; p.x < img.cols; p.x++) {
			// 如果掩膜 mask 中的对应像素值为背景（MUST_BGD 或 MAYBE_BGD）
			if (mask.at<uchar>(p) == MUST_BGD || mask.at<uchar>(p) == MAYBE_BGD)
				bgdSamples.push_back((Vec3f)img.at<Vec3b>(p)); // 将该像素的颜色值存入 bgdSamples
			else
				fgdSamples.push_back((Vec3f)img.at<Vec3b>(p)); // 否则存入 fgdSamples
		}
	}

	// 将背景样本转换为 Mat 类型用于 k-means
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	// 对背景样本进行 k-means 聚类
	kmeans(_bgdSamples, GMM::K, bgdLabels,
		TermCriteria(TermCriteria::COUNT, kmeansItCount, 0.0), 0, KMEANS_PP_CENTERS);

	// 将前景样本转换为 Mat 类型用于 k-means
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	// 对前景样本进行 k-means 聚类
	kmeans(_fgdSamples, GMM::K, fgdLabels,
		TermCriteria(TermCriteria::COUNT, kmeansItCount, 0.0), 0, KMEANS_PP_CENTERS);

	// 学习背景 GMM 模型
	bgdGMM.startLearning();
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	bgdGMM.finishLearning();

	// 学习前景 GMM 模型
	fgdGMM.startLearning();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.finishLearning();
}


//合并迭代步骤1和步骤2，将每个像素分配到GMM中所属的高斯模型，并计算GMM参数值。
static void assignAndLearnGMMS(const Mat& _img, const Mat& _mask, GMM& _bgdGMM, GMM& _fgdGMM, Mat& _partIndex) {
	_bgdGMM.startLearning();
	_fgdGMM.startLearning();

	Point p;
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(p);
			uchar t = _mask.at<uchar>(p);
			int componentIndex;

			// 将像素 p 属于背景GMM 模型中的哪个高斯成分的索引存储在 _partIndex 矩阵中
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


//根据得到的结果构造图，使用助教给的现成的库 Done
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
				wSource = -log(_bgdGMM.totalWeight(color)); //计算像素属于背景的概率。
				wSink = -log(_fgdGMM.totalWeight(color));  //计算像素属于前景的概率。
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
// 进行分割-hd
static void estimateSegmentation(GraphCut& _graph, Mat& _mask) {
	// 计算最大流
	_graph.maxFlow();

	// 遍历掩码矩阵中的每个像素
	Point p;
	for (p.y = 0; p.y < _mask.rows; p.y++) {
		for (p.x = 0; p.x < _mask.cols; p.x++) {
			// 如果掩码中的当前像素是 MAYBE_BGD 或 MAYBE_FGD
			if (_mask.at<uchar>(p) == MAYBE_BGD || _mask.at<uchar>(p) == MAYBE_FGD) {
				// 检查该像素是否属于源段
				if (_graph.isSourceSegment(p.y * _mask.cols + p.x)) {
					// 如果是，则将掩码中的像素值设为 MAYBE_FGD
					_mask.at<uchar>(p) = MAYBE_FGD;
				}
				else {
					// 如果不是，则将掩码中的像素值设为 MAYBE_BGD
					_mask.at<uchar>(p) = MAYBE_BGD;
				}
			}
		}
	}
}

GrabCut2D::~GrabCut2D(void) {}
//GrabCut函数-hd
void GrabCut2D::GrabCut(InputArray _img, InputOutputArray _mask, Rect rect,
	InputOutputArray _bgdModel, InputOutputArray _fgdModel,
	int iterCount, int mode) {

	// 加载输入图像
	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();

	// 1. 缩小图像和掩码以加快处理速度
	Mat smallImg, smallMask;
	float scaleFactor = 1.1; // 根据需要调整缩放因子
	resize(img, smallImg, Size(img.cols / scaleFactor, img.rows / scaleFactor));
	resize(mask, smallMask, Size(mask.cols / scaleFactor, mask.rows / scaleFactor), 0, 0, INTER_NEAREST);

	// 初始化掩码，若模式为 GC_WITH_RECT，则使用矩形框
	if (mode == GC_WITH_RECT) {
		Rect smallRect(rect.x / scaleFactor, rect.y / scaleFactor, rect.width / scaleFactor, rect.height / scaleFactor);
		initMaskWithRect(smallMask, smallImg.size(), smallRect);
	}

	// 2. 初始化 GMM 模型
	GMM bgdGMM(bgdModel), fgdGMM(fgdModel);

	// 使用 k-means 初始化 GMM
	if (mode == GC_WITH_RECT || mode == GC_WITH_MASK) {
		initGMMs(smallImg, smallMask, bgdGMM, fgdGMM);
	}

	if (iterCount <= 0) return;

	// 计算图的边权重
	const double gamma = 50;
	const double beta = calculateBeta(smallImg);
	Mat leftW, upleftW, upW, uprightW;
	calculateNeighborWeights(smallImg, leftW, upleftW, upW, uprightW, beta, gamma);

	// 3. 迭代提取掩码mask
	Mat compIdxs(smallImg.size(), CV_32SC1);//单通道32位
	const double lambda = 100;
	for (int i = 0; i < iterCount; i++) {
		GraphCut graph;
		assignAndLearnGMMS(smallImg, smallMask, bgdGMM, fgdGMM, compIdxs);
		//learnGMMs(smallImg, smallMask, bgdGMM, fgdGMM, compIdxs);
		getGraph(smallImg, smallMask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
		estimateSegmentation(graph, smallMask);
	}

	// 4. 将处理后的掩码放大回原始大小
	resize(smallMask, mask, mask.size(), 0, 0, INTER_NEAREST);
}


#endif

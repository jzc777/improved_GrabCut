#ifndef GraphCut_H_
#define GraphCut_H_
#include "graph.h"

class GraphCut {
private:
    Graph<double, double, double>* graphPtr; // 使用指针指向Graph对象

public:
    GraphCut(); 
    GraphCut(int, int); // 带参构造函数
    int addVertex(); // 添加一个节点
    double maxFlow(); // 最大流
    void addVertexWeights(int, double, double); // 添加节点权重
    void addEdges(int, int, double); // 加边
    bool isSourceSegment(int); // 是否为源段
};


GraphCut::GraphCut() {}

GraphCut::GraphCut(int vCount, int eCount) {
    graphPtr = new Graph<double, double, double>(vCount, eCount);
}

int GraphCut::addVertex() {
    return graphPtr->add_node();
}

double GraphCut::maxFlow() {
    return graphPtr->maxflow();
}

void GraphCut::addVertexWeights(int vNum, double sourceWeight, double sinkWeight) {
    graphPtr->add_tweights(vNum, sourceWeight, sinkWeight);
}

void GraphCut::addEdges(int vNum1, int vNum2, double weight) {
    graphPtr->add_edge(vNum1, vNum2, weight, weight);
}

bool GraphCut::isSourceSegment(int _vNum) {
    return graphPtr->what_segment(_vNum) == Graph<double, double, double>::SOURCE;
}

#endif 


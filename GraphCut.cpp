#ifndef GraphCut_H_
#define GraphCut_H_
#include "graph.h"

class GraphCut {
private:
    Graph<double, double, double>* graphPtr; // ʹ��ָ��ָ��Graph����

public:
    GraphCut(); 
    GraphCut(int, int); // ���ι��캯��
    int addVertex(); // ���һ���ڵ�
    double maxFlow(); // �����
    void addVertexWeights(int, double, double); // ��ӽڵ�Ȩ��
    void addEdges(int, int, double); // �ӱ�
    bool isSourceSegment(int); // �Ƿ�ΪԴ��
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


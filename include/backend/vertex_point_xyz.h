#ifndef MYSLAM_BACKEND_POINTVERTEX_H
#define MYSLAM_BACKEND_POINTVERTEX_H

#include "vertex.h"

namespace myslam {
namespace backend {

/**
 * @brief Vertex parameterized in xyz form
 */
class VertexPointXYZ : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPointXYZ() : Vertex(3) {}

    std::string TypeInfo() const { return "VertexPointXYZ"; }
};

}
}

#endif

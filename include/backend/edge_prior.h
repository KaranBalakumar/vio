//
// Created by heyijia on 19-1-30.
//

#ifndef SLAM_COURSE_EDGE_PRIOR_H
#define SLAM_COURSE_EDGE_PRIOR_H

#include <memory>
#include <string>

#include <Eigen/Dense>

#include "eigen_types.h"
#include "edge.h"


namespace myslam {
namespace backend {

/**
* EdgeSE3Prior, this edge is a unary edge, connected to vertex: Ti
*/
class EdgeSE3Prior : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSE3Prior(const Vec3 &p, const Qd &q) :
            Edge(6, 1, std::vector<std::string>{"VertexPose"}),
            Pp_(p), Qp_(q) {}

    /// Return edge type information
    virtual std::string TypeInfo() const override { return "EdgeSE3Prior"; }

    /// Compute residual
    virtual void ComputeResidual() override;

    /// Compute Jacobian
    virtual void ComputeJacobians() override;


private:
    Vec3 Pp_;   // pose prior
    Qd   Qp_;   // Rotation prior
};

}
}


#endif //SLAM_COURSE_EDGE_PRIOR_H

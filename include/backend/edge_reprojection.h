#ifndef MYSLAM_BACKEND_VISUALEDGE_H
#define MYSLAM_BACKEND_VISUALEDGE_H

#include <memory>
#include <string>

#include <Eigen/Dense>

#include "eigen_types.h"
#include "edge.h"

namespace myslam {
namespace backend {

/**
 * This edge is visual reprojection error, this edge is a ternary edge, connected to vertices:
 * InverseDepth of landmark point, source Camera pose T_World_From_Body1 that first observed the landmark point,
 * and measurement Camera pose T_World_From_Body2 that observed the landmark point.
 * Note: vertices_ vertex order must be InverseDepth, T_World_From_Body1, T_World_From_Body2.
 */
class EdgeReprojection : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeReprojection(const Vec3 &pts_i, const Vec3 &pts_j)
        : Edge(2, 4, std::vector<std::string>{"VertexInverseDepth", "VertexPose", "VertexPose", "VertexPose"}) {
        pts_i_ = pts_i;
        pts_j_ = pts_j;
    }

    /// Return edge type information
    virtual std::string TypeInfo() const override { return "EdgeReprojection"; }

    /// Compute residual
    virtual void ComputeResidual() override;

    /// Compute Jacobian
    virtual void ComputeJacobians() override;

//    void SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_);

private:
    //Translation imu from camera
//    Qd qic;
//    Vec3 tic;

    //measurements
    Vec3 pts_i_, pts_j_;
};

/**
* This edge is visual reprojection error, this edge is a binary edge, connected to vertices:
* World coordinate system XYZ of landmark point, Camera pose T_World_From_Body1 that observed the landmark point
* Note: vertices_ vertex order must be XYZ, T_World_From_Body1.
*/
class EdgeReprojectionXYZ : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeReprojectionXYZ(const Vec3 &pts_i)
        : Edge(2, 2, std::vector<std::string>{"VertexXYZ", "VertexPose"}) {
        obs_ = pts_i;
    }

    /// Return edge type information
    virtual std::string TypeInfo() const override { return "EdgeReprojectionXYZ"; }

    /// Compute residual
    virtual void ComputeResidual() override;

    /// Compute Jacobian
    virtual void ComputeJacobians() override;

    void SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_);

private:
    //Translation imu from camera
    Qd qic;
    Vec3 tic;

    //measurements
    Vec3 obs_;
};

/**
 * Example of only computing reprojection pose
 */
class EdgeReprojectionPoseOnly : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeReprojectionPoseOnly(const Vec3 &landmark_world, const Mat33 &K) :
        Edge(2, 1, std::vector<std::string>{"VertexPose"}),
        landmark_world_(landmark_world), K_(K) {}

    /// Return edge type information
    virtual std::string TypeInfo() const override { return "EdgeReprojectionPoseOnly"; }

    /// Compute residual
    virtual void ComputeResidual() override;

    /// Compute Jacobian
    virtual void ComputeJacobians() override;

private:
    Vec3 landmark_world_;
    Mat33 K_;
};

}
}

#endif

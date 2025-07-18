#ifndef MYSLAM_BACKEND_EDGE_H
#define MYSLAM_BACKEND_EDGE_H

#include <memory>
#include <string>
#include "eigen_types.h"
#include <eigen3/Eigen/Dense>
#include "loss_function.h"

namespace myslam {
namespace backend {

class Vertex;

/**
 * Edge is responsible for computing residuals, residual is prediction - observation, dimension is defined in constructor
 * Cost function is residual*information*residual, which is a scalar value, minimized after summing by backend
 */
class Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * Constructor, automatically allocates space for Jacobian
     * @param residual_dimension residual dimension
     * @param num_verticies number of vertices
     * @param verticies_types vertex type names, can be omitted, if omitted check will not verify
     */
    explicit Edge(int residual_dimension, int num_verticies,
                  const std::vector<std::string> &verticies_types = std::vector<std::string>());

    virtual ~Edge();

    /// Return id
    unsigned long Id() const { return id_; }

    /**
     * Set a vertex
     * @param vertex corresponding vertex object
     */
    bool AddVertex(std::shared_ptr<Vertex> vertex) {
        verticies_.emplace_back(vertex);
        return true;
    }

    /**
     * Set vertices
     * @param vertices vertices, arranged in reference order
     * @return
     */
    bool SetVertex(const std::vector<std::shared_ptr<Vertex>> &vertices) {
        verticies_ = vertices;
        return true;
    }

    /// Return the i-th vertex
    std::shared_ptr<Vertex> GetVertex(int i) {
        return verticies_[i];
    }

    /// Return all vertices
    std::vector<std::shared_ptr<Vertex>> Verticies() const {
        return verticies_;
    }

    /// Return number of associated vertices
    size_t NumVertices() const { return verticies_.size(); }

    /// Return edge type information, implemented in subclasses
    virtual std::string TypeInfo() const = 0;

    /// Compute residual, implemented in subclasses
    virtual void ComputeResidual() = 0;

    /// Compute Jacobian, implemented in subclasses
    /// This backend does not support automatic differentiation, need to implement Jacobian computation method for each subclass
    virtual void ComputeJacobians() = 0;

//    ///Compute the effect of this edge on the Hessian matrix, implemented in subclasses
//    virtual void ComputeHessionFactor() = 0;

    /// Compute squared error, multiplied by information matrix
    double Chi2() const;
    double RobustChi2() const;

    /// Return residual
    VecX Residual() const { return residual_; }

    /// Return Jacobian
    std::vector<MatXX> Jacobians() const { return jacobians_; }

    /// Set information matrix
    void SetInformation(const MatXX &information) {
        information_ = information;
        // sqrt information
        sqrt_information_ = Eigen::LLT<MatXX>(information_).matrixL().transpose();
    }

    /// Return information matrix
    MatXX Information() const {
        return information_;
    }

    MatXX SqrtInformation() const {
        return sqrt_information_;
    }

    void SetLossFunction(LossFunction* ptr){ lossfunction_ = ptr; }
    LossFunction* GetLossFunction(){ return lossfunction_;}
    void RobustInfo(double& drho, MatXX& info) const;

    /// Set observation information
    void SetObservation(const VecX &observation) {
        observation_ = observation;
    }

    /// Return observation information
    VecX Observation() const { return observation_; }

    /// Check if all edge information is set
    bool CheckValid();

    int OrderingId() const { return ordering_id_; }

    void SetOrderingId(int id) { ordering_id_ = id; };

protected:
    unsigned long id_;  // edge id
    int ordering_id_;   //edge id in problem
    std::vector<std::string> verticies_types_;  // Vertex type information for each vertex, used for debug
    std::vector<std::shared_ptr<Vertex>> verticies_; // Vertices corresponding to this edge
    VecX residual_;                 // Residual
    std::vector<MatXX> jacobians_;  // Jacobians, each Jacobian dimension is residual x vertex[i]
    MatXX information_;             // Information matrix
    MatXX sqrt_information_;
    VecX observation_;              // Observation information

    LossFunction *lossfunction_;
};

}
}

#endif

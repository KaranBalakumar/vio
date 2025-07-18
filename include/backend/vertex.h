#ifndef MYSLAM_BACKEND_VERTEX_H
#define MYSLAM_BACKEND_VERTEX_H

#include "eigen_types.h"

namespace myslam {
namespace backend {
extern unsigned long global_vertex_id;
/**
 * @brief Vertex, corresponding to a parameter block
 * Variable values are stored as VecX, dimension must be specified during construction
 */
class Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * Constructor
     * @param num_dimension vertex dimension
     * @param local_dimension local parameterization dimension, if -1 then same as vertex dimension
     */
    explicit Vertex(int num_dimension, int local_dimension = -1);

    virtual ~Vertex();

    /// Return variable dimension
    int Dimension() const;

    /// Return variable local dimension
    int LocalDimension() const;

    /// The vertex id
    unsigned long Id() const { return id_; }

    /// Return parameter values
    VecX Parameters() const { return parameters_; }

    /// Return reference to parameter values
    VecX &Parameters() { return parameters_; }

    /// Set parameter values
    void SetParameters(const VecX &params) { parameters_ = params; }

    // Backup and rollback parameters, used to discard bad estimates during iteration
    void BackUpParameters() { parameters_backup_ = parameters_; }
    void RollBackParameters() { parameters_ = parameters_backup_; }

    /// Addition operation, can be redefined
    /// Default is vector addition
    virtual void Plus(const VecX &delta);

    /// Return vertex type name, implemented in subclasses
    virtual std::string TypeInfo() const = 0;

    int OrderingId() const { return ordering_id_; }

    void SetOrderingId(unsigned long id) { ordering_id_ = id; };

    /// Fix the estimated value of this vertex
    void SetFixed(bool fixed = true) {
        fixed_ = fixed;
    }

    /// Test if this vertex is fixed
    bool IsFixed() const { return fixed_; }

protected:
    VecX parameters_;   // Actual stored variable values
    VecX parameters_backup_; // Backup parameters for each iteration, used for rollback
    int local_dimension_;   // Local parameterization dimension
    unsigned long id_;  // Vertex id, automatically generated

    /// Ordering id is the id after sorting in the problem, used to find corresponding Jacobian blocks
    /// Ordering id contains dimension information, e.g., ordering_id=6 corresponds to the 6th column in Hessian
    /// Starting from zero
    unsigned long ordering_id_ = 0;

    bool fixed_ = false;    // Whether it is fixed
};

}
}

#endif

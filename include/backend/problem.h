#ifndef MYSLAM_BACKEND_PROBLEM_H
#define MYSLAM_BACKEND_PROBLEM_H

#include <unordered_map>
#include <map>
#include <memory>

#include "eigen_types.h"
#include "edge.h"
#include "vertex.h"

typedef unsigned long ulong;

namespace myslam {
namespace backend {

typedef unsigned long ulong;
//    typedef std::unordered_map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

class Problem {
public:

    /**
     * Problem type
     * SLAM problem or general problem
     *
     * If it's a SLAM problem, then pose and landmark are separated, Hessian is stored in sparse form
     * SLAM problem only accepts certain specific Vertex and Edge types
     * If it's a general problem, then hessian is dense, unless user sets certain vertices as marginalized
     */
    enum class ProblemType {
        SLAM_PROBLEM,
        GENERIC_PROBLEM
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Problem(ProblemType problemType);

    ~Problem();

    bool AddVertex(std::shared_ptr<Vertex> vertex);

    /**
     * remove a vertex
     * @param vertex_to_remove
     */
    bool RemoveVertex(std::shared_ptr<Vertex> vertex);

    bool AddEdge(std::shared_ptr<Edge> edge);

    bool RemoveEdge(std::shared_ptr<Edge> edge);

    /**
     * Get edges that are judged as outliers during optimization, convenient for frontend to remove outliers
     * @param outlier_edges
     */
    void GetOutlierEdges(std::vector<std::shared_ptr<Edge>> &outlier_edges);

    /**
     * Solve this problem
     * @param iterations
     * @return
     */
    bool Solve(int iterations = 10);

    /// Marginalize a frame and landmarks hosted by it
    bool Marginalize(std::shared_ptr<Vertex> frameVertex,
                     const std::vector<std::shared_ptr<Vertex>> &landmarkVerticies);

    bool Marginalize(const std::shared_ptr<Vertex> frameVertex);
    bool Marginalize(const std::vector<std::shared_ptr<Vertex> > frameVertex,int pose_dim);

    MatXX GetHessianPrior(){ return H_prior_;}
    VecX GetbPrior(){ return b_prior_;}
    VecX GetErrPrior(){ return err_prior_;}
    MatXX GetJtPrior(){ return Jt_prior_inv_;}

    void SetHessianPrior(const MatXX& H){H_prior_ = H;}
    void SetbPrior(const VecX& b){b_prior_ = b;}
    void SetErrPrior(const VecX& b){err_prior_ = b;}
    void SetJtPrior(const MatXX& J){Jt_prior_inv_ = J;}

    void ExtendHessiansPriorSize(int dim);

    //test compute prior
    void TestComputePrior();

private:

    /// Implementation of Solve, solve general problem
    bool SolveGenericProblem(int iterations);

    /// Implementation of Solve, solve SLAM problem
    bool SolveSLAMProblem(int iterations);

    /// Set ordering_index for each vertex
    void SetOrdering();

    /// set ordering for new vertex in slam problem
    void AddOrderingSLAM(std::shared_ptr<Vertex> v);

    /// Construct big H matrix
    void MakeHessian();

    /// schur solve SBA
    void SchurSBA();

    /// Solve linear system
    void SolveLinearSystem();

    /// Update state variables
    void UpdateStates();

    void RollbackStates(); // Sometimes after update residual becomes larger, need to rollback and retry

    /// Compute and update Prior part
    void ComputePrior();

    /// Determine if a vertex is a Pose vertex
    bool IsPoseVertex(std::shared_ptr<Vertex> v);

    /// Determine if a vertex is a landmark vertex
    bool IsLandmarkVertex(std::shared_ptr<Vertex> v);

    /// After adding new vertex, need to adjust the size of several hessians
    void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

    /// Check if ordering is correct
    bool CheckOrdering();

    void LogoutVectorSize();

    /// Get edges connected to a certain vertex
    std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);

    /// Levenberg
    /// Compute initial Lambda for LM algorithm
    void ComputeLambdaInitLM();

    /// Add or subtract Lambda to Hessian diagonal
    void AddLambdatoHessianLM();

    void RemoveLambdaHessianLM();

    /// Used in LM algorithm to determine if Lambda worked in last iteration, and how to scale Lambda
    bool IsGoodStepInLM();

    /// PCG iterative linear solver
    VecX PCGSolver(const MatXX &A, const VecX &b, int maxIter);

    double currentLambda_;
    double currentChi_;
    double stopThresholdLM_;    // LM iteration exit threshold condition
    double ni_;                 // Control Lambda scaling size

    ProblemType problemType_;

    /// Entire information matrix
    MatXX Hessian_;
    VecX b_;
    VecX delta_x_;

    /// Prior part information
    MatXX H_prior_;
    VecX b_prior_;
    VecX b_prior_backup_;
    VecX err_prior_backup_;

    MatXX Jt_prior_inv_;
    VecX err_prior_;

    /// SBA Pose part
    MatXX H_pp_schur_;
    VecX b_pp_schur_;
    // Hessian Landmark and pose parts
    MatXX H_pp_;
    VecX b_pp_;
    MatXX H_ll_;
    VecX b_ll_;

    /// all vertices
    HashVertex verticies_;

    /// all edges
    HashEdge edges_;

    /// Query edge by vertex id
    HashVertexIdToEdge vertexToEdge_;

    /// Ordering related
    ulong ordering_poses_ = 0;
    ulong ordering_landmarks_ = 0;
    ulong ordering_generic_ = 0;
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_;        // Pose vertices sorted by ordering
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_landmark_vertices_;    // Landmark vertices sorted by ordering

    // verticies need to marg. <Ordering_id_, Vertex>
    HashVertex verticies_marg_;

    bool bDebug = false;
    double t_hessian_cost_ = 0.0;
    double t_PCGsovle_cost_ = 0.0;
};

}
}

#endif

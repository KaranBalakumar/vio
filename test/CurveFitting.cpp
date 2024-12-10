#include <iostream>
#include <random>
#include "backend/problem.h"

using namespace myslam::backend;
using namespace std;

// Vertex for the curve fitting model, template parameters: optimization variable dimension and data type
class CurveFittingVertex: public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingVertex(): Vertex(3) {}  // abc: Three parameters, Vertex is 3D
    virtual std::string TypeInfo() const { return "abc"; }
};

// Error model, template parameters: observation dimension, type, and connected vertex type
class CurveFittingEdge: public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge( double x, double y ): Edge(1,1, std::vector<std::string>{"abc"}) {
        x_ = x;
        y_ = y;
    }
    // Compute the residual for the curve model
    virtual void ComputeResidual() override
    {
        Vec3 abc = verticies_[0]->Parameters();  // Estimated parameters
        residual_(0) = std::exp( abc(0)*x_*x_ + abc(1)*x_ + abc(2) ) - y_;  // Construct the residual
    }

    // Compute the Jacobian of the residual with respect to the variables
    virtual void ComputeJacobians() override
    {
        Vec3 abc = verticies_[0]->Parameters();
        double exp_y = std::exp( abc(0)*x_*x_ + abc(1)*x_ + abc(2) );

        Eigen::Matrix<double, 1, 3> jaco_abc;  // Residual is 1D, state variables are 3, so the Jacobian matrix is 1x3
        jaco_abc << x_ * x_ * exp_y, x_ * exp_y , 1 * exp_y;
        jacobians_[0] = jaco_abc;
    }
    /// Return the type information of the edge
    virtual std::string TypeInfo() const override { return "CurveFittingEdge"; }
public:
    double x_,y_;  // x value, y value is the _measurement
};

int main()
{
    double a=1.0, b=2.0, c=1.0;         // True parameter values
    int N = 100;                          // Number of data points
    double w_sigma= 1.;                 // Noise Sigma value

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.,w_sigma);

    // Construct the problem
    Problem problem(Problem::ProblemType::GENERIC_PROBLEM);
    shared_ptr< CurveFittingVertex > vertex(new CurveFittingVertex());

    // Set initial values for the parameters a, b, c
    vertex->SetParameters(Eigen::Vector3d (0.,0.,0.));
    // Add the parameters to the least squares problem
    problem.AddVertex(vertex);

    // Construct N observations
    for (int i = 0; i < N; ++i) {

        double x = i/100.;
        double n = noise(generator);
        // Observation y
        double y = std::exp( a*x*x + b*x + c ) + n;
//        double y = std::exp( a*x*x + b*x + c );

        // Residual function for each observation
        shared_ptr< CurveFittingEdge > edge(new CurveFittingEdge(x,y));
        std::vector<std::shared_ptr<Vertex>> edge_vertex;
        edge_vertex.push_back(vertex);
        edge->SetVertex(edge_vertex);

        // Add this residual to the least squares problem
        problem.AddEdge(edge);
    }

    std::cout<<"\nTest CurveFitting start..."<<std::endl;
    /// Use LM to solve
    problem.Solve(30);

    std::cout << "-------After optimization, we got these parameters :" << std::endl;
    std::cout << vertex->Parameters().transpose() << std::endl;
    std::cout << "-------ground truth: " << std::endl;
    std::cout << "1.0,  2.0,  1.0" << std::endl;

    // std
    return 0;
}

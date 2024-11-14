#include "initial/initial_sfm.h"

GlobalSFM::GlobalSFM() {}

/*****************************************************************************************************************
Triangulation: Given normalized matched feature points {p1, p2} and their respective camera projection matrices {P1, P2} of size [3,4],
estimate the 3D point X3D.
p1 = P1 * X3D
p2 = P2 * X3D
We use the Direct Linear Transformation (DLT) method (transform the equation to the form A*X = 0, and then solve the linear system using SVD):
For p1 = P1 * X3D, cross-multiply both sides with p1 to make the equation equal to zero:
p1 × P1 * X3D = 0
The cross-product matrix is:
|0  -1  y|
|1   0 -x| 
|-y  x  0|

This can be written as:
|0  -1  y|  |P1.row(0)|  
|1   0 -x| *|P1.row(1)| * X3D = 0
|-y  x  0|  |P1.row(2)|  

For the first row |0  -1  y|, multiply it with the three rows of P1 to get four values, which are multiplied with the homogeneous 3D point coordinates to yield zero:
(y * P1.row(2) - P1.row(1)) * X3D = 0
For the second row |1   0 -x|, the equation becomes:
(x * P1.row(2) - P1.row(0)) * X3D = 0
This gives two constraints. Similarly, for point p2 = P2 * X3D, two more equations are derived:
(y' * P2.row(2) - P2.row(1)) * X3D = 0
(x' * P2.row(2) - P2.row(0)) * X3D = 0
Writing this in the form A * X = 0 gives a 4x4 matrix A:
A =
|y * P1.row(2) - P1.row(1)|
|x * P1.row(2) - P1.row(0)|
|y' * P2.row(2) - P2.row(1)|
|x' * P2.row(2) - P2.row(0)|

Solve for X using Singular Value Decomposition (SVD):
cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
x3D = vt.row(3).t();  // The last column of vt contains the solution for X
x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);  // Normalize to get non-homogeneous coordinates
*****************************************************************************************************************/

// Triangulate a 3D point from two views
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                                 Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
    // Set up the design matrix for the triangulation
    Matrix4d design_matrix = Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);

    // Solve the linear system using SVD
    Vector4d triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    // Normalize the result to get the 3D point in non-homogeneous coordinates
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

// Solve for camera pose using PnP (Perspective-n-Point) algorithm
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f)
{
    vector<cv::Point2f> pts_2_vector;
    vector<cv::Point3f> pts_3_vector;

    // Collect 2D and 3D feature points from the SFM feature list
    for (int j = 0; j < feature_num; j++)
    {
        if (sfm_f[j].state != true) continue;
        Vector2d point2d;
        for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
        {
            if (sfm_f[j].observation[k].first == i)
            {
                Vector2d img_pts = sfm_f[j].observation[k].second;
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
                cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
                pts_3_vector.push_back(pts_3);
                break;
            }
        }
    }

    // If there are too few points, the pose estimation is unstable
    if (int(pts_2_vector.size()) < 15)
    {
        printf("unstable feature tracking, please move your device slowly!\n");
        if (int(pts_2_vector.size()) < 10)
            return false;
    }

    // Solve the PnP problem using OpenCV
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
    
    if (!pnp_succ)
    {
        return false;
    }

    // Convert the rotation vector to matrix form
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);

    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // Update initial camera pose
    R_initial = R_pnp;
    P_initial = T_pnp;
    return true;
}

// Triangulate points between two frames
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                                      int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                                      vector<SFMFeature> &sfm_f)
{
    assert(frame0 != frame1);

    for (int j = 0; j < feature_num; j++)
    {
        if (sfm_f[j].state == true) continue;

        bool has_0 = false, has_1 = false;
        Vector2d point0;
        Vector2d point1;

        // Find the 2D points in both frames
        for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
        {
            if (sfm_f[j].observation[k].first == frame0)
            {
                point0 = sfm_f[j].observation[k].second;
                has_0 = true;
            }
            if (sfm_f[j].observation[k].first == frame1)
            {
                point1 = sfm_f[j].observation[k].second;
                has_1 = true;
            }
        }

        // If points are available in both frames, triangulate the 3D point
        if (has_0 && has_1)
        {
            Vector3d point_3d;
            triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
        }
    }
}

//  q w_R_cam t w_R_cam
//  c_rotation cam_R_w
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool GlobalSFM::construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
                          const Matrix3d relative_R, const Vector3d relative_T,
                          vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
    feature_num = sfm_f.size();
    // Initialize frame l as the origin, and compute the pose of other frames
    // Assume frame l as the origin, and compute the pose of the current frame relative to frame l using relative_R and relative_T
    q[l].w() = 1;
    q[l].x() = 0;
    q[l].y() = 0;
    q[l].z() = 0;
    T[l].setZero();
    q[frame_num - 1] = q[l] * Quaterniond(relative_R);  // Set the relative rotation for the last frame
    T[frame_num - 1] = relative_T;  // Set the relative translation for the last frame

    // Pose represents the transformation matrix from frame l to each other frame
    Matrix3d c_Rotation[frame_num];
    Vector3d c_Translation[frame_num];
    Quaterniond c_Quat[frame_num];
    double c_rotation[frame_num][4];
    double c_translation[frame_num][3];
    Eigen::Matrix<double, 3, 4> Pose[frame_num];

    // Compute the rotation and translation for frame l
    c_Quat[l] = q[l].inverse();  // Inverse the quaternion for frame l
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    // Compute the rotation and translation for the last frame
    c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
    c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
    c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
    Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
    Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

    // 1. Triangulate 3D points for frame l (reference frame) and frame frame_num-1 (current frame)
    // 2. Use PnP to solve for transformation matrices (R_initial, P_initial) from frame l+1 to each frame, 
    //    and then triangulate the corresponding points.
    // 3. Triangulate the points for frames l to frame_num-2.
    // 4. Solve PnP for frames l-1 to 0 and triangulate those points.

    for (int i = l; i < frame_num - 1; i++)
    {
        // Solve for PnP for frames after l
        if (i > l)
        {
            Matrix3d R_initial = c_Rotation[i - 1];
            Vector3d P_initial = c_Translation[i - 1];
            if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))  // Solve PnP for frame i
                return false;
            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quat[i] = c_Rotation[i];
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }

        // Triangulate points based on the PnP result
        triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
    }

    // Triangulate points between frames l, l+1, ..., frame_num-2
    for (int i = l + 1; i < frame_num - 1; i++)
        triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);

    // Solve PnP for frames l-1 to 0 and triangulate the points
    for (int i = l - 1; i >= 0; i--)
    {
        Matrix3d R_initial = c_Rotation[i + 1];
        Vector3d P_initial = c_Translation[i + 1];
        if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))  // Solve PnP for frame i
            return false;
        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quat[i] = c_Rotation[i];
        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);  // Triangulate between frame i and frame l
    }

    // 5. Triangulate all other points that have been observed in multiple frames.
    // By now, we have the pose for all frames and 3D coordinates for the feature points.
    for (int j = 0; j < feature_num; j++)
    {
        if (sfm_f[j].state == true)
            continue;
        if ((int)sfm_f[j].observation.size() >= 2)  // Only triangulate points observed in at least two frames
        {
            Vector2d point0, point1;
            int frame_0 = sfm_f[j].observation[0].first;
            point0 = sfm_f[j].observation[0].second;
            int frame_1 = sfm_f[j].observation.back().first;
            point1 = sfm_f[j].observation.back().second;
            Vector3d point_3d;
            triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
        }
    }

    // 6. Perform global Bundle Adjustment (BA) using Ceres Solver
    ceres::Problem problem;
    ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();
    // Add parameters for each frame's rotation and translation
    for (int i = 0; i < frame_num; i++)
    {   
        // Convert rotation and translation for Ceres
        c_translation[i][0] = c_Translation[i].x();
        c_translation[i][1] = c_Translation[i].y();
        c_translation[i][2] = c_Translation[i].z();
        c_rotation[i][0] = c_Quat[i].w();
        c_rotation[i][1] = c_Quat[i].x();
        c_rotation[i][2] = c_Quat[i].y();
        c_rotation[i][3] = c_Quat[i].z();
        problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
        problem.AddParameterBlock(c_translation[i], 3);
        if (i == l)
        {
            problem.SetParameterBlockConstant(c_rotation[i]);
        }
        if (i == l || i == frame_num - 1)
        {
            problem.SetParameterBlockConstant(c_translation[i]);
        }
    }

    // Add residual blocks for all observed 3D feature points
    for (int i = 0; i < feature_num; i++)
    {
        if (sfm_f[i].state != true)
            continue;
        for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
        {
            int l = sfm_f[i].observation[j].first;
            ceres::CostFunction *cost_function = ReprojectionError3D::Create(
                sfm_f[i].observation[j].second.x(),
                sfm_f[i].observation[j].second.y());

            problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l],
                                     sfm_f[i].position);
        }
    }

    // Solver options for Ceres BA
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Check if BA converged
    if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
    {
        // BA converged
    }
    else
    {
        // BA did not converge
        return false;
    }

    // 7. After BA, convert the rotation and translation from the local frame to global coordinates
    for (int i = 0; i < frame_num; i++)
    {
        q[i].w() = c_rotation[i][0];
        q[i].x() = c_rotation[i][1];
        q[i].y() = c_rotation[i][2];
        q[i].z() = c_rotation[i][3];
        q[i] = q[i].inverse();
    }
    for (int i = 0; i < frame_num; i++)
    {
        T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
    }
    
    // Store tracked points in the map
    for (int i = 0; i < (int)sfm_f.size(); i++)
    {
        if (sfm_f[i].state)
            sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
    }
    return true;
}


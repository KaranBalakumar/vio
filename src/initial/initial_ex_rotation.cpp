#include "initial/initial_ex_rotation.h"

// Constructor for InitialEXRotation class
InitialEXRotation::InitialEXRotation() {
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());  // Identity rotation matrix for camera
    Rc_g.push_back(Matrix3d::Identity()); // Identity rotation matrix for camera-IMU relation
    Rimu.push_back(Matrix3d::Identity()); // Identity rotation matrix for IMU
    ric = Matrix3d::Identity();  // Identity rotation matrix for camera-IMU extrinsic calibration
}

/***********************************************************************************************************************
CalibrationExRotation() - Calibrate the extrinsic rotation matrix
1. solveRelativeR(corres) calculates the relative rotation matrix between consecutive camera frames based on epipolar geometry. The corres input is the normalized coordinates of corresponding feature points between the current frame and the previous frame.
    1.1. Extract the corresponding 2D coordinates from corres and store them in ll and rr.
    1.2. Use epipolar geometry to compute the essential matrix between the two frames.
    1.3. Decompose the essential matrix using SVD to obtain four possible R and t solutions.
    1.4. Use triangulation to verify each Rt solution, selecting the one with positive depth.
    1.5. R here is the transformation from the previous frame to the current frame, and its transpose gives the current frame's pose relative to the previous frame.
2. Get the relative rotation matrices between camera and IMU for each frame and store them in vectors.
    In the code, L is the left multiplication matrix for the camera rotation quaternion, and R is the right multiplication matrix for the IMU rotation quaternion. This results in the following equation:
    qbc * q(bkbk+1) = qbc * q(ckck+1)
3. Solve for the extrinsic rotation matrix from the camera to the IMU using the above relationship.
4. After iterating at least WINDOW_SIZE times and ensuring that the singular values of R are greater than 0.25, the calibration is considered successful.
***********************************************************************************************************************/
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result) {
    frame_count++;
    Rc.push_back(solveRelativeR(corres));               // Calculate camera rotation matrix between frames using epipolar geometry
    Rimu.push_back(delta_q_imu.toRotationMatrix());     // Calculate IMU rotation matrix between frames from IMU preintegration
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);  // Calculate IMU rotation relative to the starting frame IMU

    // Construct the system matrix A based on the quaternion rotations
    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++) {
        Quaterniond r1(Rc[i]);
        Quaterniond r2(Rc_g[i]);
        
        double angular_distance = 180 / M_PI * r1.angularDistance(r2);  // Calculate angular distance between two rotations

        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;  // Apply Huber weighting function for robustness

        ++sum_ok;
        Matrix4d L, R;
        
        double w = Quaterniond(Rc[i]).w();
        Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);
    }

    // Perform SVD decomposition of matrix A
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Matrix<double, 4, 1> x = svd.matrixV().col(3);  // Get the last column of V, which gives the quaternion solution
    Quaterniond estimated_R(x);  // Convert the result to a quaternion
    ric = estimated_R.toRotationMatrix().inverse();  // Calculate the camera to IMU extrinsic rotation

    // Get the covariance of the estimated extrinsic rotation
    Vector3d ric_cov = svd.singularValues().tail<3>();
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25) {
        calib_ric_result = ric;  // Store the calibrated extrinsic rotation matrix
        return true;
    } else {
        return false;  // Calibration failed
    }
}

// Function to solve the relative rotation matrix between consecutive frames
Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres) {
    if (corres.size() >= 9) {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++) {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }

        // Compute the essential matrix using OpenCV
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;
        decomposeE(E, R1, R2, t1, t2);  // Decompose the essential matrix into R and t

        // If the determinant of R1 is close to zero, reverse the sign of E and recompute
        if (determinant(R1) + 1.0 < 1e-09) {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }

        // Test the triangulation to verify the R and t solutions
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;  // Choose the R matrix with better triangulation ratio

        // Convert the chosen rotation matrix to Eigen format
        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j);
        return ans_R_eigen;  // Return the relative rotation matrix
    }
    return Matrix3d::Identity();  // Return identity if not enough correspondences are found
}

// Function to test triangulation for verifying the correct R and t solutions
double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                             const vector<cv::Point2f> &r,
                                             cv::Mat_<double> R, cv::Mat_<double> t) {
    cv::Mat pointcloud;
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                 0, 1, 0, 0,
                                 0, 0, 1, 0);
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                  R(1, 0), R(1, 1), R(1, 2), t(1),
                                  R(2, 0), R(2, 1), R(2, 2), t(2));

    // Perform triangulation to compute 3D points
    cv::triangulatePoints(P, P1, l, r, pointcloud);

    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++) {
        double normal_factor = pointcloud.col(i).at<float>(3);
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)  // Check if the point is in front of both cameras
            front_count++;
    }
    return 1.0 * front_count / pointcloud.cols;  // Return the ratio of valid triangulated points
}

// Function to decompose the essential matrix into rotation and translation
void InitialEXRotation::decomposeE(cv::Mat E,
                                   cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                   cv::Mat_<double> &t1, cv::Mat_<double> &t2) {
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);

    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}


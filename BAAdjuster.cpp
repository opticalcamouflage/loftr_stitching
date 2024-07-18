#include "BAAdjuster.h"
#include <iostream>

void AngleAxis2RotationMatrix(const double* angle_axis, Eigen::Matrix3d & R )
{
    double kOne = double(1.0);
    double theta = std::hypot(angle_axis[0], angle_axis[1], angle_axis[2]);
    double wx = angle_axis[0] / theta;
    double wy = angle_axis[1] / theta;
    double wz = angle_axis[2] / theta;
    double costheta = cos(theta);
    double sintheta = sin(theta);
    R(0, 0) =     costheta   + wx*wx*(kOne -    costheta);
    R(1, 0) =  wz*sintheta   + wx*wy*(kOne -    costheta);
    R(2, 0) = -wy*sintheta   + wx*wz*(kOne -    costheta);
    R(0, 1) =  wx*wy*(kOne - costheta)     - wz*sintheta;
    R(1, 1) =     costheta   + wy*wy*(kOne -    costheta);
    R(2, 1) =  wx*sintheta   + wy*wz*(kOne -    costheta);
    R(0, 2) =  wy*sintheta   + wx*wz*(kOne -    costheta);
    R(1, 2) = -wx*sintheta   + wy*wz*(kOne -    costheta);
    R(2, 2) =     costheta   + wz*wz*(kOne -    costheta);
}

void RotationMatrix2AngleAxis(const Eigen::Matrix3d& R, double* angle_axis)
{
    double quaternion[4];
    double trace = R(0, 0) + R(1, 1) + R(2, 2);
    if (trace >= 0.0) {
        double t = sqrt(trace + double(1.0));
        quaternion[0] = double(0.5) * t;
        t = double(0.5) / t;
        quaternion[1] = (R(2, 1) - R(1, 2)) * t;
        quaternion[2] = (R(0, 2) - R(2, 0)) * t;
        quaternion[3] = (R(1, 0) - R(0, 1)) * t;
    } 
    else {
        int i = 0;
        if (R(1, 1) > R(0, 0)) {
            i = 1;
        }
        if (R(2, 2) > R(i, i)) {
        i = 2;
        }
        const int j = (i + 1) % 3;
        const int k = (j + 1) % 3;
        double t = sqrt(R(i, i) - R(j, j) - R(k, k) + double(1.0));
        quaternion[i + 1] = double(0.5) * t;
        t = double(0.5) / t;
        quaternion[0] = (R(k, j) - R(j, k)) * t;
        quaternion[j + 1] = (R(j, i) + R(i, j)) * t;
        quaternion[k + 1] = (R(k, i) + R(i, k)) * t;
    }

    const double& q1 = quaternion[1];
    const double& q2 = quaternion[2];
    const double& q3 = quaternion[3];
    const double sin_theta = std::hypot(q1, q2, q3);

    if (std::fpclassify(sin_theta) != FP_ZERO) {
        const double& cos_theta = quaternion[0];
        const double two_theta =
            double(2.0) * ((cos_theta < double(0.0)) ? atan2(-sin_theta, -cos_theta)
                                        : atan2(sin_theta, cos_theta));
        const double k = two_theta / sin_theta;
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    } else {
        const double k(2.0);
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    }    
}

BAAdjuster::BAAdjuster(const cv::Size& imageSize,const std::vector<std::vector<cv::KeyPoint>>& keypoints)
{
    Eigen::Matrix3d K = estimateCameraMatrix(imageSize, 1.0);
    
    if (keypoints.size() == 2 && keypoints[0].size() == keypoints[1].size()) 
        keypointSize = keypoints[0].size();
    else 
        std::cerr << "Error: Keypoints size mismatch or does not contain exactly two elements." << std::endl;
    
    for (int i=0; i < keypointSize; i++) 
    {
        cam_points1.push_back(pixel2cam(keypoints[0][i].pt, K));//相机坐标系
        cam_points2.push_back(pixel2cam(keypoints[1][i].pt, K));
    }
    cv::Mat P1,P2;
    P1 = (cv::Mat_<double>(3, 4) << 1.0, 0.0, 0.0, 0.0,
                                    0.0, -1.0, 0.0, 0.0,
                                    0.0, 0.0, -1.0, 1.0);
    TriangulatePoints(P1, P2, cam_points1, cam_points2, points3d, FIXED);    
}

void BAAdjuster::adjust(cv::Mat& H)
{
    // std::vector<cv::Mat> R, t, n;//旋转矩阵R、平移矩阵t、关键点所在平面的法向量n
    //TODO: K to cv
    // int solutions = cv::decomposeHomographyMat(H, K, R, t, n);
    // std::vector<int>  sol;
    // std::vector<uchar> pointMask;
    // for (int i = 0; i < keypointSize; i++) {
	// 	pointMask.push_back(1);
	// }
    // cv::filterHomographyDecompByVisibleRefpoints(R, n, cam_points1, cam_points2, sol, pointMask);
    // std::cout << "solution: " << solutions << std::endl;    
    // for (int i = 0; i < solutions; ++i) {
	// 	std::cout << "======== " << i << " ========" << std::endl;
	// 	std::cout << "rotation" << i << " = " << std::endl;
	// 	std::cout << R.at(i) << std::endl;
	// 	std::cout << "translation" << i << " = " << std::endl;
	// 	std::cout << t.at(i) << std::endl;
	// 	std::cout << "n" << i << " = " << std::endl;
	// 	std::cout << n.at(i) << std::endl;		
	// }
    // std::cout << "the real solution num is : " << sol.size()<<std::endl;
	// std::cout << "the real solution is : " << sol[0] << std::endl << sol[1] << std::endl;



    //TODO R to Eigen
    Eigen::Matrix3d Runit;
    Runit <<    1.0, 0.0, 0.0,
                0.0, -1.0, 0.0,
                0.0, 0.0, -1.0;
    RotationMatrix2AngleAxis(Runit, camera_para2);
    camera_para2[3] = 0.0;
    camera_para2[4] = 0.0;
    camera_para2[5] = 1.0;
    std::vector<Eigen::Vector3d> points3D;
    for (const auto& pt : points3d)
    {
        points3D.push_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
        // std::cout << pt <<std::endl;
    }

    for (size_t i = 0; i < keypointSize ; ++i) {    
        ceres::CostFunction* cost_function = ReprojectionError::Create(cam_points2[i].x, cam_points2[i].y, points3D[i]);
        problem.AddResidualBlock(cost_function, nullptr, camera_para2);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;


    Eigen::Matrix3d ResultR2;
    AngleAxis2RotationMatrix(camera_para2, ResultR2);
    std::cout << ResultR2 <<std::endl;
    std::cout << camera_para2[3] <<std::endl;
    std::cout << camera_para2[4] <<std::endl;
    std::cout << camera_para2[5] <<std::endl;

    // Eigen::Matrix3d RH = ResultR2.transpose() * Runit;
    // Eigen::Vector3d tH = ResultR2.transpose() * (Eigen::Vector3d(0.0, 0.0, 1.0) - Eigen::Vector3d(camera_para2[3], camera_para2[4], camera_para2[5]));
    Eigen::Matrix3d RH = Runit.transpose() * ResultR2;
    Eigen::Vector3d tH = Runit.transpose() * (Eigen::Vector3d(camera_para2[3], camera_para2[4], camera_para2[5]) - Eigen::Vector3d(0.0, 0.0, 1.0));
    Eigen::Matrix3d H0 = K * (RH - tH * Eigen::Vector3d(0.0, 0.0, 1.0).transpose()) * K.inverse();

    std::cout << "calc H: " << H0 << std::endl;
    cv::eigen2cv(H0, H);
}

Eigen::Matrix3d BAAdjuster::estimateCameraMatrix(const cv::Size& imageSize, double alpha) {
    double cx = imageSize.width / 2.0;
    double cy = imageSize.height / 2.0;
    // double f = alpha * std::max(imageSize.width, imageSize.height);
    double f = 2888.0;
    
    K << f, 0, cx,
         0, f, cy,
         0, 0, 1;
    return K;
}

cv::Point2d BAAdjuster::pixel2cam(const cv::Point2d& p, const Eigen::Matrix3d& K )
{
    return cv::Point2d
           (
               ( p.x - K( 0,2 ) ) / K( 0,0 ),
               ( p.y - K( 1,2 ) ) / K( 1,1 )
           );
}

void BAAdjuster::TriangulatePoints(const cv::Mat& P1, const cv::Mat& P2, const Point2fVector& points1, const Point2fVector& points2,
                             Point3fVector& points3D, TriangulationMethod method) {
   switch (method) {
        case SVD:
            TriangulateSVD(P1, P2, points1, points2, points3D);
            break;
        case LINEAR_AVERAGE:
            TriangulateLinearAverage(P1, P2, points1, points2, points3D);
            break;
        case FIXED:
            TriangulateFixed(P1, P2, points1, points2, points3D);
            break;
        case SVD_FIXED:
            TriangulateSVDFixed(P1, P2, points1, points2, points3D);
            break;
    }
}

void BAAdjuster::TriangulateSVD(const cv::Mat& P1, const cv::Mat& P2, const Point2fVector& points1, const Point2fVector& points2, Point3fVector& points3D)
{
    points3D.clear();
    int N = points1.size();
    cv::Mat A( 4, 4, CV_64F);
    cv::Mat U, W, Vt;
    for (size_t i = 0; i < N; ++i) {
        double x1 = points1[i].x;
        double y1 = points1[i].y;
        double x2 = points2[i].x;
        double y2 = points2[i].y;

        A.at<double>( 0, 0) = x1 * P1.at<double>(2, 0) - P1.at<double>(0, 0);
        A.at<double>( 0, 1) = x1 * P1.at<double>(2, 1) - P1.at<double>(0, 1);
        A.at<double>( 0, 2) = x1 * P1.at<double>(2, 3) - P1.at<double>(0, 3);

        A.at<double>( 1, 0) = y1 * P1.at<double>(2, 0) - P1.at<double>(1, 0);
        A.at<double>( 1, 1) = y1 * P1.at<double>(2, 1) - P1.at<double>(1, 1);
        A.at<double>( 1, 2) = y1 * P1.at<double>(2, 3) - P1.at<double>(1, 3);

        A.at<double>( 2, 0) = x2 * P2.at<double>(2, 0) - P2.at<double>(0, 0);
        A.at<double>( 2, 1) = x2 * P2.at<double>(2, 1) - P2.at<double>(0, 1);
        A.at<double>( 2, 2) = x2 * P2.at<double>(2, 3) - P2.at<double>(0, 3);

        A.at<double>( 3, 0) = y2 * P2.at<double>(2, 0) - P2.at<double>(1, 0);
        A.at<double>( 3, 1) = y2 * P2.at<double>(2, 1) - P2.at<double>(1, 1);
        A.at<double>( 3, 2) = y2 * P2.at<double>(2, 3) - P2.at<double>(1, 3);

        cv::SVD::compute(A, W, U, Vt);
        cv::Mat X = Vt.row(2).t();        
        points3D.emplace_back(cv::Point3f(X.at<double>(0) / X.at<double>(2),
                                          X.at<double>(1) / X.at<double>(2),
                                          0)); // Z = 0
    }   
}

void BAAdjuster::TriangulateLinearAverage(const cv::Mat& P1, const cv::Mat& P2, const Point2fVector& points1, const Point2fVector& points2, Point3fVector& points3D)
{
    points3D.clear();
    int N = points1.size();
    for (size_t i = 0; i < N; ++i) {
        double x1 = points1[i].x;
        double y1 = points1[i].y;
        double x2 = points2[i].x;
        double y2 = points2[i].y;

        cv::Mat A1(2, 2, CV_64F);
        cv::Mat B1(2, 1, CV_64F);
        cv::Mat A2(2, 2, CV_64F);
        cv::Mat B2(2, 1, CV_64F);

        // Fill in A1 and B1 for the first camera
        A1.at<double>(0, 0) = x1 * P1.at<double>(2, 0) - P1.at<double>(0, 0);
        A1.at<double>(0, 1) = x1 * P1.at<double>(2, 1) - P1.at<double>(0, 1);
        A1.at<double>(1, 0) = y1 * P1.at<double>(2, 0) - P1.at<double>(1, 0);
        A1.at<double>(1, 1) = y1 * P1.at<double>(2, 1) - P1.at<double>(1, 1);

        B1.at<double>(0, 0) = -(x1 * P1.at<double>(2, 3) - P1.at<double>(0, 3));
        B1.at<double>(1, 0) = -(y1 * P1.at<double>(2, 3) - P1.at<double>(1, 3));

        // Fill in A2 and B2 for the second camera
        A2.at<double>(0, 0) = x2 * P2.at<double>(2, 0) - P2.at<double>(0, 0);
        A2.at<double>(0, 1) = x2 * P2.at<double>(2, 1) - P2.at<double>(0, 1);
        A2.at<double>(1, 0) = y2 * P2.at<double>(2, 0) - P2.at<double>(1, 0);
        A2.at<double>(1, 1) = y2 * P2.at<double>(2, 1) - P2.at<double>(1, 1);

        B2.at<double>(0, 0) = -(x2 * P2.at<double>(2, 3) - P2.at<double>(0, 3));
        B2.at<double>(1, 0) = -(y2 * P2.at<double>(2, 3) - P2.at<double>(1, 3));

        cv::Mat XY1, XY2;
        cv::solve(A1, B1, XY1, cv::DECOMP_SVD);
        cv::solve(A2, B2, XY2, cv::DECOMP_SVD);

        double X = (XY1.at<double>(0, 0) + XY2.at<double>(0, 0)) / 2.0;
        double Y = (XY1.at<double>(1, 0) + XY2.at<double>(1, 0)) / 2.0;

        points3D.emplace_back(cv::Point3f(X, Y, 0));
    }
}

void BAAdjuster::TriangulateFixed(const cv::Mat& P1, const cv::Mat& P2, const Point2fVector& points1, const Point2fVector& points2, Point3fVector& points3D)
{
    points3D.clear();
    int N = points1.size();
    for (size_t i = 0; i < N; ++i) {
        double x1 = points1[i].x;
        double y1 = points1[i].y;

        cv::Mat A1(2, 2, CV_64F);
        cv::Mat B1(2, 1, CV_64F);   

        // Fill in A1 and B1 for the first camera
        A1.at<double>(0, 0) = x1 * P1.at<double>(2, 0) - P1.at<double>(0, 0);
        A1.at<double>(0, 1) = x1 * P1.at<double>(2, 1) - P1.at<double>(0, 1);
        A1.at<double>(1, 0) = y1 * P1.at<double>(2, 0) - P1.at<double>(1, 0);
        A1.at<double>(1, 1) = y1 * P1.at<double>(2, 1) - P1.at<double>(1, 1);

        B1.at<double>(0, 0) = -(x1 * P1.at<double>(2, 3) - P1.at<double>(0, 3));
        B1.at<double>(1, 0) = -(y1 * P1.at<double>(2, 3) - P1.at<double>(1, 3));

        cv::Mat XY1;
        cv::solve(A1, B1, XY1, cv::DECOMP_SVD);

        double X = XY1.at<double>(0, 0);
        double Y = XY1.at<double>(1, 0);

        points3D.emplace_back(cv::Point3f(X, Y, 0));
        // if (i < 50)
        //     std::cout << cv::Point3f(X, Y, 0) << std::endl;
    }
}

void BAAdjuster::TriangulateSVDFixed(const cv::Mat& P1, const cv::Mat& P2, const Point2fVector& points1, const Point2fVector& points2, Point3fVector& points3D)
{
    //not finished yet
    points3D.clear();
    int N = points1.size();
    cv::Mat A( 2, 4, CV_64F);
    cv::Mat U, W, Vt;
    for (size_t i = 0; i < N; ++i) {
        double x1 = points1[i].x;
        double y1 = points1[i].y;

        A.at<double>( 0, 0) = x1 * P1.at<double>(2, 0) - P1.at<double>(0, 0);
        A.at<double>( 0, 1) = x1 * P1.at<double>(2, 1) - P1.at<double>(0, 1);
        A.at<double>( 0, 2) = x1 * P1.at<double>(2, 2) - P1.at<double>(0, 2);
        A.at<double>( 0, 3) = x1 * P1.at<double>(2, 3) - P1.at<double>(0, 3);

        A.at<double>( 1, 0) = y1 * P1.at<double>(2, 0) - P1.at<double>(1, 0);
        A.at<double>( 1, 1) = y1 * P1.at<double>(2, 1) - P1.at<double>(1, 1);
        A.at<double>( 1, 2) = y1 * P1.at<double>(2, 2) - P1.at<double>(1, 2);
        A.at<double>( 1, 3) = y1 * P1.at<double>(2, 3) - P1.at<double>(1, 3);


        cv::SVD::compute(A, W, U, Vt);
        cv::Mat X = Vt.row(3).t();        
        points3D.emplace_back(cv::Point3f(X.at<double>(0) / X.at<double>(2),
                                          X.at<double>(1) / X.at<double>(2),
                                          0)); // Z = 0
    }   
}
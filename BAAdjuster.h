#include <vector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y,Eigen::Vector3d point)
        : observed_x(observed_x), observed_y(observed_y),point(point) {}

    template <typename T>
    bool operator()(const T* const camera, T* residuals) const {
        T p[3];
        // ceres::AngleAxisRotatePoint(camera, point, p);
        Eigen::Matrix<T, 3, 3> R_mat;
        ceres::AngleAxisToRotationMatrix(camera, R_mat.data()); 

        p[0] = R_mat(0,0) * point[0] + R_mat(0,1) * point[1] + R_mat(0,2) * point[2];
        p[1] = R_mat(1,0) * point[0] + R_mat(1,1) * point[1] + R_mat(1,2) * point[2];
        p[2] = R_mat(2,0) * point[0] + R_mat(2,1) * point[1] + R_mat(2,2) * point[2];

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        residuals[0] = xp - T(observed_x);
        residuals[1] = yp - T(observed_y);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y,const Eigen::Vector3d point) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(
            new ReprojectionError(observed_x, observed_y,point)));
    }

    double observed_x;
    double observed_y;
    Eigen::Vector3d point;

};

class BAAdjuster
{
public:
    BAAdjuster(const cv::Size& imageSize,const std::vector<std::vector<cv::KeyPoint>>& keypoints);// output: camera parameters/translation pixels
    void adjust(cv::Mat& H);

public:
    typedef std::vector<cv::Point2f> Point2fVector;
    typedef std::vector<cv::Point3f> Point3fVector;

    enum TriangulationMethod {
        SVD,
        LINEAR_AVERAGE,
        FIXED,
        SVD_FIXED
    };

private:
    Eigen::Matrix3d estimateCameraMatrix(const cv::Size& imageSize, double alpha);
    cv::Point2d pixel2cam (const cv::Point2d& p, const Eigen::Matrix3d& K );
    void TriangulatePoints(const cv::Mat& P1, const cv::Mat& P2, const Point2fVector& points1, const Point2fVector& points2,
                             Point3fVector& points3D, TriangulationMethod method);
    void TriangulateSVD(const cv::Mat& P1, const cv::Mat& P2, const Point2fVector& points1, const Point2fVector& points2, Point3fVector& points3D);
    void TriangulateLinearAverage(const cv::Mat& P1, const cv::Mat& P2, const Point2fVector& points1, const Point2fVector& points2, Point3fVector& points3D);
    void TriangulateFixed(const cv::Mat& P1, const cv::Mat& P2, const Point2fVector& points1, const Point2fVector& points2, Point3fVector& points3D);
    void TriangulateSVDFixed(const cv::Mat& P1, const cv::Mat& P2, const Point2fVector& points1, const Point2fVector& points2, Point3fVector& points3D);

    int keypointSize = 0;
    Eigen::Matrix3d K;
    Point2fVector cam_points1, cam_points2; //x/z y/z
    Point3fVector points3d;
    ceres::Problem problem;
    double camera_para1[6];
    double camera_para2[6];
};
/**
 * @file    LoFTRApp.cpp
 *
 * @author  btran
 *
 */

#include "LoFTR.hpp"
#include "Utility.hpp"
#include "BAAdjuster.h"
#include <utility>
#include <opencv2/opencv.hpp>


#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/util.hpp"



#include <iostream>
#include <memory>
#include <windows.h>

static constexpr float CONFIDENCE_THRESHOLD = 0.1;

namespace
{
std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>>
processOneImagePair(const Ort::LoFTR& loftrOsh, const cv::Mat& queryImg, const cv::Mat& refImg, float* queryData,
                    float* refData, float confidenceThresh = CONFIDENCE_THRESHOLD);
}  // namespace

int main(int argc, char* argv[])
{
    // if (argc != 4) {
    //     std::cerr << "Usage: [apps] [path/to/onnx/loftr] [path/to/image1] [path/to/image2]" << std::endl;
    //     return EXIT_FAILURE;
    // }

    // const std::string ONNX_MODEL_PATH = argv[1];
    // const std::vector<std::string> IMAGE_PATHS = {argv[2], argv[3]};
    const std::string ONNX_MODEL_PATH = "C:/opensource projects/loftr_stitching/assets/loftr.onnx";
    const std::vector<std::string> IMAGE_PATHS = {"C:/opensource projects/loftr_stitching/assets/1.jpg",
                                                     "C:/opensource projects/loftr_stitching/assets/2.jpg"};

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> grays;
    std::transform(IMAGE_PATHS.begin(), IMAGE_PATHS.end(), std::back_inserter(images),
                   [](const auto& imagePath) { return cv::imread(imagePath); });
    for (int i = 0; i < 2; ++i) {
        if (images[i].empty()) {
            throw std::runtime_error("failed to open " + IMAGE_PATHS[i]);
        }
    }

    std::transform(IMAGE_PATHS.begin(), IMAGE_PATHS.end(), std::back_inserter(grays),
                   [](const auto& imagePath) { return cv::imread(imagePath, 0); });

    std::vector<float> queryData(Ort::LoFTR::IMG_CHANNEL * Ort::LoFTR::IMG_H * Ort::LoFTR::IMG_W);
    std::vector<float> refData(Ort::LoFTR::IMG_CHANNEL * Ort::LoFTR::IMG_H * Ort::LoFTR::IMG_W);


    DWORD loads = GetTickCount();
    Ort::LoFTR osh(
        ONNX_MODEL_PATH, 0,
        std::vector<std::vector<int64_t>>{{1, Ort::LoFTR::IMG_CHANNEL, Ort::LoFTR::IMG_H, Ort::LoFTR::IMG_W},
                                          {1, Ort::LoFTR::IMG_CHANNEL, Ort::LoFTR::IMG_H, Ort::LoFTR::IMG_W}});
    std::cout << "load model time: " << GetTickCount() - loads << "ms" << std::endl;


    DWORD infers = GetTickCount();
    auto matchedKpts = processOneImagePair(osh, grays[0], grays[1], queryData.data(), refData.data());
    std::cout << "inference time: " << GetTickCount() - infers << "ms" << std::endl;



    const std::vector<cv::KeyPoint>& queryKpts = matchedKpts.first;
    const std::vector<cv::KeyPoint>& refKpts = matchedKpts.second;
    std::vector<cv::DMatch> matches;
    for (int i = 0; i < queryKpts.size(); ++i) {
        cv::DMatch match;
        match.imgIdx = 0;
        match.queryIdx = i;
        match.trainIdx = i;
        matches.emplace_back(std::move(match));
    }
    





    // std::vector<cv::detail::ImageFeatures> features;
    // cv::detail::ImageFeatures features_query, features_ref;
    // features_query.img_idx = 0; // 图像索引
    // features_query.keypoints = queryKpts; // 特征点
    // features_query.descriptors = cv::UMat(); // 空描述符
    // features_ref.img_idx = 1; // 图像索引
    // features_ref.keypoints = refKpts; // 特征点
    // features_ref.descriptors = cv::UMat(); // 空描述符
    // features.push_back(features_query);
    // features.push_back(features_ref);
    // cv::detail::MatchesInfo matches_info;
    // matches_info.src_img_idx = 0; // 源图像索引
    // matches_info.dst_img_idx = 1; // 目标图像索引
    // matches_info.matches = matches; // 复制匹配信息
    // matches_info.num_inliers = matches.size(); // 假设所有匹配都是内点
    // matches_info.confidence = 1.0; // 设置一个默认的高信任度
    // for (const auto& match : matches) {
    //     matches_info.inliers_mask.push_back(1); // 假设所有匹配都是内点
    // }
    // std::vector<cv::detail::MatchesInfo> pairwise_matches;
    // pairwise_matches.push_back(matches_info);
    // pairwise_matches.push_back(cv::detail::MatchesInfo());
    // std::vector<cv::detail::CameraParams> cameras;
    // std::unique_ptr<cv::detail::HomographyBasedEstimator> estimator;
    // estimator = std::make_unique<cv::detail::HomographyBasedEstimator>();
    // estimator->operator()(features, pairwise_matches, cameras);
    // for (size_t i = 0; i < cameras.size(); ++i)
    // {
    //     cv::Mat R;
    //     cameras[i].R.convertTo(R, CV_32F);
    //     cameras[i].R = R;        
    // }
    // std::cout << cameras.size() << std::endl;











    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    for( size_t i = 0; i < matches.size(); i++ )
    {
        obj.push_back( queryKpts[ matches[i].queryIdx ].pt );
        scene.push_back( refKpts[ matches[i].trainIdx ].pt );
    }
    cv::Mat H = cv::findHomography( obj, scene, 0 );

	std::vector<std::vector<cv::KeyPoint>> keypoints = {queryKpts, refKpts};
    BAAdjuster adjuster(images[0].size(), keypoints);
    adjuster.adjust(H);

    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cv::Point2f(0, 0);
    obj_corners[1] = cv::Point2f( (float)images[0].cols, 0 );
    obj_corners[2] = cv::Point2f( (float)images[0].cols, (float)images[0].rows );
    obj_corners[3] = cv::Point2f( 0, (float)images[0].rows );
    std::vector<cv::Point2f> scene_corners(4);
    cv::perspectiveTransform( obj_corners, scene_corners, H);

    // std::vector<cv::Mat> R, t, n;
    // cv::Mat K = (cv::Mat_<double>(3, 3) << 2888.0, 0.0, 2000.0,
    //                                         0.0, 2888.0, 1500.0,
    //                                         0.0, 0.0, 1.0);
    // int solutions = cv::decomposeHomographyMat(H, K, R, t, n);
    // cv::Mat R1 = R[0];
    // cv::Mat t1 = t[0];
    // cv::Mat Rt(3, 4, R1.type());
    // R1.copyTo(Rt(cv::Rect(0, 0, 3, 3)));
    // t1.copyTo(Rt(cv::Rect(3, 0, 1, 3)));
    // cv::Mat pt = (cv::Mat_<double>(3, 1) << obj_corners[1].x, obj_corners[1].y, 1.0);
    // cv::Mat transformed_pt = K * Rt * pt;
    // double x = transformed_pt.at<double>(0, 0) / transformed_pt.at<double>(2, 0);
    // double y = transformed_pt.at<double>(1, 0) / transformed_pt.at<double>(2, 0);
    // std::cout << cv::Point2f(x, y) << std::endl;








    for (auto it = scene_corners.begin(); it !=scene_corners.end(); it++)
    {
        std::cout<< *it <<std ::endl;
    }
     std::vector<cv::Point2f> all_corners = {
        cv::Point2f(0, 0),
        cv::Point2f((float)images[1].cols, 0),
        cv::Point2f((float)images[1].cols, (float)images[1].rows),
        cv::Point2f(0, (float)images[1].rows)
    };
    all_corners.insert(all_corners.end(), scene_corners.begin(), scene_corners.end());
    float min_x = all_corners[0].x;
    float max_x = all_corners[0].x;
    float min_y = all_corners[0].y;
    float max_y = all_corners[0].y;
    for (const auto& pt : all_corners) {
        if (pt.x < min_x) min_x = pt.x;
        if (pt.x > max_x) max_x = pt.x;
        if (pt.y < min_y) min_y = pt.y;
        if (pt.y > max_y) max_y = pt.y;
    }
    cv::Size stitched_image_size((int)std::ceil(max_x - min_x), (int)std::ceil(max_y - min_y));
    cv::Mat stitched_image = cv::Mat::zeros(stitched_image_size, images[1].type());
    cv::Mat roi(stitched_image, cv::Rect(-min_x, -min_y, images[1].cols, images[1].rows));
    images[1].copyTo(roi);
    warpPerspective(images[0], stitched_image, H, stitched_image_size,cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);


    cv::Mat matchesImage;
    cv::drawMatches(images[0], queryKpts, images[1], refKpts, matches, matchesImage, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


    cv::imwrite("loftr.jpg", stitched_image);
    cv::imshow("loftr", stitched_image );
    cv::waitKey();

    return EXIT_SUCCESS;


}

namespace
{
std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>>
processOneImagePair(const Ort::LoFTR& loftrOsh, const cv::Mat& queryImg, const cv::Mat& refImg, float* queryData,
                    float* refData, float confidenceThresh)
{
    int origQueryW = queryImg.cols, origQueryH = queryImg.rows;
    int origRefW = refImg.cols, origRefH = refImg.rows;

    cv::Mat scaledQueryImg, scaledRefImg;
    cv::resize(queryImg, scaledQueryImg, cv::Size(Ort::LoFTR::IMG_W, Ort::LoFTR::IMG_H), 0, 0, cv::INTER_CUBIC);
    cv::resize(refImg, scaledRefImg, cv::Size(Ort::LoFTR::IMG_W, Ort::LoFTR::IMG_H), 0, 0, cv::INTER_CUBIC);

    loftrOsh.preprocess(queryData, scaledQueryImg.data, Ort::LoFTR::IMG_W, Ort::LoFTR::IMG_H, Ort::LoFTR::IMG_CHANNEL);
    loftrOsh.preprocess(refData, scaledRefImg.data, Ort::LoFTR::IMG_W, Ort::LoFTR::IMG_H, Ort::LoFTR::IMG_CHANNEL);
    auto inferenceOutput = loftrOsh({queryData, refData});

    // inferenceOutput[0].second: keypoints0 of shape [num kpt x 2]
    // inferenceOutput[1].second: keypoints1 of shape [num kpt x 2]
    // inferenceOutput[2].second: confidences of shape [num kpt]

    int numKeyPoints = inferenceOutput[2].second[0];
    std::vector<cv::KeyPoint> queryKpts, refKpts;
    queryKpts.reserve(numKeyPoints);
    refKpts.reserve(numKeyPoints);

    for (int i = 0; i < numKeyPoints; ++i) {
        float confidence = inferenceOutput[2].first[i];
        if (confidence < confidenceThresh) {
            continue;
        }
        float queryX = inferenceOutput[0].first[i * 2 + 0];
        float queryY = inferenceOutput[0].first[i * 2 + 1];
        float refX = inferenceOutput[1].first[i * 2 + 0];
        float refY = inferenceOutput[1].first[i * 2 + 1];
        cv::KeyPoint queryKpt, refKpt;
        queryKpt.pt.x = queryX * origQueryW / Ort::LoFTR::IMG_W;
        queryKpt.pt.y = queryY * origQueryH / Ort::LoFTR::IMG_H;

        refKpt.pt.x = refX * origRefW / Ort::LoFTR::IMG_W;
        refKpt.pt.y = refY * origRefH / Ort::LoFTR::IMG_H;

        queryKpts.emplace_back(std::move(queryKpt));
        refKpts.emplace_back(std::move(refKpt));
    }

    return std::make_pair(queryKpts, refKpts);
}
}  // namespace

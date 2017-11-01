#include "feature_solver.h"

FeatureSolver::FeatureSolver(string str1, string str2)
  :imgpath_1(str1),imgpath_2(str2)
{

}
void FeatureSolver::run()
{
  Mat img1=imread(imgpath_1,CV_LOAD_IMAGE_COLOR);
  Mat img2=imread(imgpath_2,CV_LOAD_IMAGE_COLOR);
  
  keypoints_1.clear();
  keypoints_2.clear();
  Mat descriptors_1,descriptors_2;
  Ptr<ORB> orb=ORB::create(500,1.2F,8,31,0,2,ORB::HARRIS_SCORE,31,20);
  
  orb->detect(img1,keypoints_1);
  orb->detect(img2,keypoints_2);
  
  orb->compute(img1,keypoints_1,descriptors_1);
  orb->compute(img2,keypoints_2,descriptors_2);
  
  Mat outimg1;
  drawKeypoints(img1,keypoints_1,outimg1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
  imwrite("../result/feature_points.jpg",outimg1);
  
  vector<DMatch> matches;
  BFMatcher matcher(NORM_HAMMING);
  
  matcher.match(descriptors_1,descriptors_2,matches);
  
  double min_dist=10000,max_dist=0;
  
  for(int i=0;i<descriptors_1.rows;i++)
  {
    double dist=matches[i].distance;
    if(dist<min_dist)min_dist=dist;
    if(dist>max_dist)max_dist=dist;
  }
  
  good_matches.clear();
  for(int i=0;i<descriptors_1.rows;i++)
  {
    if(matches[i].distance<=max(2*min_dist,30.0))
      good_matches.push_back(matches[i]);
  }
  
  Mat img_match;
  Mat img_goodmatch;
  drawMatches(img1,keypoints_1,img2,keypoints_2,matches,img_match);
  drawMatches(img1,keypoints_1,img2,keypoints_2,good_matches,img_goodmatch);
  
  imwrite("../result/img_allmatch.jpg",img_match);
  imwrite("../result/img_goodmatch.jpg",img_goodmatch);
  
}


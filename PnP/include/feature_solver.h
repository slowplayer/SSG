#ifndef FEATURE_SOLVER_H 
#define FEATURE_SOLVER_H

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class FeatureSolver
{
public:
   FeatureSolver(string str1,string str2);
   ~FeatureSolver(){};
   
   void run();
   
   inline vector<KeyPoint> get_keypoints_1(){return keypoints_1;}
   inline vector<KeyPoint> get_keypoints_2(){return keypoints_2;}
   inline vector<DMatch> get_good_matches(){return good_matches;}
private:
   string imgpath_1,imgpath_2;
   vector<KeyPoint> keypoints_1,keypoints_2;
   vector<DMatch> good_matches;
};

#endif
#include "feature_solver.h"
#include "epnp.h"

#include <iostream>
#include <Eigen/Geometry>

using namespace std;
using namespace cv;

Eigen::Vector3f backProject(KeyPoint kp,float invfx,float invfy,float cx,float cy,float z)
{
  float x,y;
  Eigen::Vector3f point3d;
  
  x=(kp.pt.x-cx)*z*invfx;
  y=(kp.pt.y-cy)*z*invfy;
  
  point3d<<x,y,z;
  
  return point3d;
}

int main()
{
    string rgb1="../data/rgb1.jpg";
    string rgb2="../data/rgb2.jpg";
    string depth1="../data/depth1.jpg";
 // string depth2="../data/depth2.jpg";
    
    float fx=520.9;
    float fy=521.0;
    float cx=325.1;
    float cy=249.7;
    
    Mat d1=imread(depth1,CV_LOAD_IMAGE_UNCHANGED);
    
    FeatureSolver featureSolver(rgb1,rgb2);
    
    featureSolver.run();
    
    vector<KeyPoint> keypoint1=featureSolver.get_keypoints_1();
    vector<KeyPoint> keypoint2=featureSolver.get_keypoints_2();
    vector<DMatch> matches=featureSolver.get_good_matches();
    
    epnp PnP;
    PnP.set_internal_parameters(cx,cy,fx,fy);
    PnP.set_maximum_number_of_correspondences(matches.size());
    
    for(DMatch m:matches)
    {
      ushort ud1=d1.ptr<unsigned short>(int(keypoint1[m.queryIdx].pt.y))[
					 int(keypoint1[m.queryIdx].pt.x)];
      if(ud1==0)
	continue;
      
      float fd1=ud1/1000.0;
      Eigen::Vector3f p1;
      p1=backProject(keypoint1[m.queryIdx],1.0/fx,1.0/fy,cx,cy,fd1);
      
      PnP.add_correspondence(p1[0],p1[1],p1[2],keypoint2[m.trainIdx].pt.x,keypoint2[m.trainIdx].pt.y);
     }
     
     double R_est[3][3],t_est[3];
     double err2=PnP.compute_pose(R_est,t_est);
     
     cout<<"Reprojection error: "<<err2<<endl;
     cout<<"Found pose:"<<endl;
     PnP.print_pose(R_est,t_est);
     
     return 0;
}
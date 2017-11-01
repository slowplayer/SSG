#ifndef ICP_SOLVER_H
#define ICP_SOLVER_H

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Sparse>

class IcpSolver
{
public:
  IcpSolver();
  ~IcpSolver(){};
  
  
  inline float getAccumulatedWeight()const {return accumulated_weight_;}
  inline unsigned int getNoOfSamples(){return no_of_samples_;}
  
  void reset();
  void add(const Eigen::Vector3f& point,const Eigen::Vector3f& corresponding_point,
		  float weight=1.0);
  Eigen::Affine3f getTransformation();
  
protected:
  unsigned int no_of_samples_;
  float accumulated_weight_;
  Eigen::Vector3f mean1_,mean2_;
  Eigen::Matrix<float,3,3> covariance_;
};
#endif
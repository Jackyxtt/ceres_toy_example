#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include<Eigen/Core>
#include<Eigen/Geometry>
#include <math.h>

struct ReprojectionError3D
{
    ReprojectionError3D(double observe_u, double observe_v):
    observe_u(observe_u), observe_v(observe_v){}

    template <typename T>
    bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residual) const { //为什么point不用设定指针指向为常量
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p);
        p[0] += camera_T[0];
        p[1] += camera_T[1];
        p[2] += camera_T[2];
        
        residual[0] = T(p[0]/p[2]) - T(observe_u);
        residual[1] = T(p[1]/p[2]) - T(observe_v);
        return true;
    }

    static ceres::CostFunction* Create(const double observe_x, const double observe_y){
        return new ceres::AutoDiffCostFunction<ReprojectionError3D, 2, 4, 3, 3>(new ReprojectionError3D(observe_x, observe_y));
    }

    double observe_u;
    double observe_v;
};

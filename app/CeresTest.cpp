#include <iostream>
#include<glog/logging.h>

#include <random>

#include "backend/ceres_reprojection.h"
#include "unordered_map"
#include "vector"

using namespace std;
using namespace Eigen;

struct Frame
{
    Frame(Matrix3d R, Vector3d t): Rwc(R), twc(t), qwc(R) {}
    Matrix3d Rwc;
    Quaterniond qwc;
    Vector3d twc;

    unordered_map<int, Vector3d> featurePerId;
};

struct Point{
    Point(Vector3d pose):gt_pos(pose), pos(pose){}
    Vector3d gt_pos;
    Vector3d pos;
};

void GenSimDataInWorldFrame(vector<Frame>& cameraPoses, vector<Point>& points){
    int featureNum = 20;
    int poseNum = 3;

    int radius = 8;
    for (int i = 0; i < poseNum; i++){
        double theta = i * 2 * M_PI / (4 * poseNum);
        Matrix3d R;
        R = AngleAxisd(theta, Vector3d::UnitZ());
        Vector3d t = Vector3d(radius * cos(theta) - radius, radius * sin(theta), sin(2 * theta));
        cameraPoses.push_back(Frame(R, t));
    }

    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0. ,1. / 1000.);
    for(int i = 0; i < featureNum; i++){
        uniform_real_distribution<double> xy_rand(-4, 4.0);
        uniform_real_distribution<double> z_rand(4. ,8. );
        Vector3d Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
        points.push_back(Point(Pw));



        for (int j = 0; j < poseNum; j++){

            Matrix3d Rcw = cameraPoses[j].Rwc.inverse();
            Vector3d tcw = -Rcw * cameraPoses[j].twc;

            Vector3d Pc = Rcw * Pw + tcw;
            Pc[0] = Pc[0] / Pc[2] + noise_pdf(generator);
            Pc[1] = Pc[1] / Pc[2] + noise_pdf(generator);
            Pc[2] = 1;
            cameraPoses[j].featurePerId.insert({int(i), Pc});
        }
    }

}


int main(){
    FLAGS_log_dir = "/home/ros/dev_workspace/hw_course5_new/log";
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging("TestingCeres");

    vector<Frame> cameraPoses;
    vector<Point> points;
    GenSimDataInWorldFrame(cameraPoses, points);

    int NumCameraPoses = cameraPoses.size();

    Matrix3d c_Rotation[NumCameraPoses];
    Vector3d c_Translation[NumCameraPoses];
    Quaterniond c_Quat[NumCameraPoses];
    double c_translation[NumCameraPoses][3];
    double c_rotation[NumCameraPoses][4];
    double pointsLoc[points.size()][3];

    ceres::Problem problem; 
    ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();

    for (int i = 0; i < cameraPoses.size(); i++){

        // 从第1帧到第i帧
        c_Quat[i] = Quaterniond(cameraPoses[i].Rwc.inverse());
        c_Rotation[i] = c_Quat[i].toRotationMatrix();
        c_Translation[i] = -c_Rotation[i] * cameraPoses[i].twc;

        c_translation[i][0] = c_Translation[i][0];
        c_translation[i][1] = c_Translation[i][1];
        c_translation[i][2] = c_Translation[i][2];
        c_rotation[i][0] = c_Quat[i].w();
        c_rotation[i][1] = c_Quat[i].x();
        c_rotation[i][2] = c_Quat[i].y();
        c_rotation[i][3] = c_Quat[i].z();

        problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
        problem.AddParameterBlock(c_translation[i], 3);

        if (i == 0)
        {
            problem.SetParameterBlockConstant(c_rotation[i]);
            problem.SetParameterBlockConstant(c_translation[i]);
        }
        if (i == 2) problem.SetParameterBlockConstant(c_translation[i]);
    }

    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0. ,0.2);

    for (int i = 0; i < points.size(); i++){

        //为3d路标点添加噪声
        points[i].pos[0] += noise_pdf(generator); 
        points[i].pos[1] += noise_pdf(generator); 
        points[i].pos[2] += noise_pdf(generator); 
        pointsLoc[i][0] = points[i].pos[0];
        pointsLoc[i][1] = points[i].pos[1];
        pointsLoc[i][2] = points[i].pos[2];
        

        for (int j = 0; j < NumCameraPoses; j++){

            Vector2d p_obs(cameraPoses[j].featurePerId[i][0], cameraPoses[j].featurePerId[i][1]);
            ceres::CostFunction *cost_function = ReprojectionError3D::Create(p_obs[0], p_obs[1]);
            problem.AddResidualBlock(cost_function, NULL, c_rotation[j], c_translation[j], pointsLoc[i]);
            LOG(INFO) << "point" << i << " being added to ceres problem is \n" << pointsLoc[i][0] << " " << pointsLoc[i][1]  << " " <<  pointsLoc[i][2] << endl; 
            LOG(INFO) << "cam" << j << " inputting t is \n" << c_translation[j][0] << " " << c_translation[j][1]  << " " <<  c_translation[j][2] << endl;
             
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << summary.FullReport() << "\n";

    if ((summary.termination_type == ceres::CONVERGENCE) && summary.final_cost < 5e-3 ){
        LOG(INFO) << "ceres solve converge" << endl;
    }else{
        LOG(INFO) << "ceres solve not converge" << endl;
    }

    for (int i = 0; i < points.size(); i++){

        // points[i].pos[0] = pointsLoc[i][0];
        // points[i].pos[1] = pointsLoc[i][1];
        // points[i].pos[2] = pointsLoc[i][2];

        LOG(INFO) << "point" << i << " gt coodinate under world frame is " << points[i].gt_pos[0] << ", " << points[i].gt_pos[1] << ", " << points[i].gt_pos[2] << endl; 
        LOG(INFO) << "point" << i << " noisy coodinate under world frame is " << points[i].pos[0] << ", " << points[i].pos[1] << ", " << points[i].pos[2] << endl;
        LOG(INFO) << "point" << i << " opt coodinate under world frame is " << pointsLoc[i][0] << ", " << pointsLoc[i][1] << ", " << pointsLoc[i][2] << endl; 
    }
    for (int i = 0; i < cameraPoses.size(); i++){
        Vector3d t;
        t[0] = c_translation[i][0];
        t[1] = c_translation[i][1];
        t[2] = c_translation[i][2];

        Quaterniond q;
        q.w() = c_rotation[i][0];
        q.x() = c_rotation[i][1];
        q.y() = c_rotation[i][2];
        q.z() = c_rotation[i][3];

        Matrix3d R;
        R = q.inverse().toRotationMatrix();
        t = -R * t;

        LOG(INFO) << "cam" << i << " gt R under world frame is \n" << cameraPoses[i].Rwc << endl; 
        LOG(INFO) << "cam" << i << " opt R under world frame is \n" << R << endl; 

        LOG(INFO) << "cam" << i << " gt t under world frame is \n" << cameraPoses[i].twc << endl;
        LOG(INFO) << "cam" << i << " opt t under world frame is \n" << t << endl;
    }
    
    google::ShutdownGoogleLogging();
    return 0;
}
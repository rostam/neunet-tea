#include <iostream>
#include <vector>
#include <fstream>
#include "Eigen/Dense"
#include <cmath>
#include <sstream>
#include <boost/algorithm/string.hpp>

using namespace std;

int main() {
    cout << "Hello World!" << endl;
    std::ifstream file("car_features.csv");
    std::string str;
    //km, fuel, age, price
    //https://matrices.io/deep-neural-network-from-scratch/
    /**
     * Number of kilometers: quantitative, number between 0 and 350k.
     * Type of fuel: binary data diesel/gasoline.
     * Age: quantitative, number between 0 and 40.
     * Price: quantitative, number between 0 and 40k.
     */
    Eigen::MatrixXd mat(9061, 4);
     int numOfLines = 0;
    std::string line;
    int cnt = 0;
    while (file >> line)
    {
        std::replace( line.begin(), line.end(), ',', ' ');
        stringstream ss(line);
        int km, age, price;
        string fuel;
        ss >> km >> fuel >> age >> price;
        int fuel_i = (fuel == "Diesel") ? -1 : 1;
        if(price > 1000) {
            if(fuel == "Diesel" || fuel == "Essence") {
//                cerr << cnt;
            mat.row(cnt) = Eigen::Vector4d(km, fuel_i, age, price).transpose();
                cnt++;
            }
        }
    }

    //normalize number of kilometers (km)
    double meanValue = mat.col(0).mean();
    mat.col(0) -= meanValue*Eigen::VectorXd::Ones(mat.rows());
    double standard_deviation = sqrt((mat.col(0).transpose()*mat.col(0)).sum()/(mat.rows()-1));
    mat.col(0) /= standard_deviation;
    cerr << meanValue << "  " << standard_deviation << endl;

    //normalize age
    meanValue = mat.col(2).mean();
    mat.col(2) -= meanValue*Eigen::VectorXd::Ones(mat.rows());
    standard_deviation = sqrt((mat.col(2).transpose()*mat.col(2)).sum()/(mat.rows()-1));
    mat.col(2) /= standard_deviation;

    //normalize price
    double min_price = mat.col(3).minCoeff();
    double max_price = mat.col(3).maxCoeff();
    cerr << min_price << " " << max_price << endl;
    mat.col(3) -= min_price * Eigen::VectorXd::Ones(mat.rows());
    double d = max_price - min_price;
    mat.col(3) /= d;
    cerr << meanValue << " " << standard_deviation;
    return 0;

    Eigen::MatrixXd X(5, 4);
    X << 1.4, -1, 0.4, 1,
         0.4, -1, 0.1, 1,
         5.4, -1, 4,   1,
         1.5, -1, 1,   1,
         1.8,  1, 1,   1;//last column for bias

    Eigen::MatrixXd W1(4,3);
    W1 << 0.01, 0.05, 0.07,
          0.20, 0.041, 0.11,
          0.04, 0.56, 0.13,
          0.1,  0.1,  0.1;//bias

    Eigen::MatrixXd Z2(5,3);
    Z2 = X*W1;
    Eigen::MatrixXd a2 = Z2.unaryExpr([&](double d) {return tanh(d);});
    a2.conservativeResize(Eigen::NoChange, a2.cols()+1);
    a2(0,3) = 1;
    a2(1,3) = 1;
    a2(2,3) = 1;
    a2(3,3) = 1;
    a2(4,3) = 1;

    Eigen::MatrixXd W2(4, 2);
    W2 << 0.04, 0.78,
          0.40, 0.45,
          0.65, 0.23,
          0.1, 0.1;

    Eigen::MatrixXd Z3  = a2*W2;
    Eigen::MatrixXd a3 = Z3.unaryExpr([&](double d) {return tanh(d);});
    a3.conservativeResize(Eigen::NoChange, a3.cols()+1);
    a3(0,2) = 1;
    a3(1,2) = 1;
    a3(2,2) = 1;
    a3(3,2) = 1;
    a3(4,2) = 1;

    Eigen::MatrixXd W3(3, 1);
    W3 << 0.04,
          0.41,
          0.1;
//
    Eigen::MatrixXd Z4  = a3*W3;
    Eigen::MatrixXd a4 = Z4.unaryExpr([&](double d) {return tanh(d);});


    double error = 0;
//    for(int i=0;i < X.rows();i++)
//        error += (X.row(i) - a4).squaredNorm();
//    error = (X.row(0).transpose() - a4).squaredNorm();
//    cout << X.row(0).transpose() << endl << a4 << endl;
    cout << error;
    return 0;

}

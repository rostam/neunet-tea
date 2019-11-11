#include <iostream>
#include <vector>
#include <fstream>
#include "Eigen/Dense"
#include <cmath>

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
//    std::getline(file, str);
//    while (std::getline(file, str)) {
//        cout << str << endl;
//    }

    Eigen::Vector3d X(1.4, -1, 0.4);

    Eigen::Matrix3d W1;
    W1 << 0.01, 0.05, 0.07,
          0.20, 0.041, 0.11,
          0.04, 0.56, 0.13;

//    cout << X.transpose().size() << "\n" << W1.size();
    Eigen::Vector3d Z2 = X.transpose()*W1;
    Eigen::Vector3d a = Z2.unaryExpr([&](double d) {return tanh(d);});
    cout << a;
    Eigen::MatrixXd W2(3, 2);
    W2 << 0.04, 0.78,
          0.40, 0.45,
          0.65, 0.23;
//    Z3  = a
    return 0;

}

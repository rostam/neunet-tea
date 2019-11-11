#ifndef NET_H
#define NET_H

#include <array>
#include <cmath>

class net {
  std::array<double,6> initial_weights = {0.15, 0.20, 0.25, 0.35, 0.40, 0.45};
  std::array<double,2> initial_inputs = {0.05, 0.10};
  std::array<double,2> biases = {0.35, 0.60};
  std::array<double,2> outputs = {0.01, 0.99};

  float activation(float activation){
    return 1/(1+exp(-activation));
  }
};

#endif // NET_H

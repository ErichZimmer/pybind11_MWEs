#ifndef KERNELS_H
#define KERNELS_H

#include <vector>
#include <functional>

#include "lib/constants.h"

namespace kernels
{
   std::vector<float> gaussian(int, float);
   std::vector<float> box(int, float);
   
   std::function<std::vector<float>(int, float)> get_kernel_type(int);
   
   float apply_conv_kernel(
      const float*, 
      const std::vector<float>&,
      int, int, int, 
      int
   );
}

#endif
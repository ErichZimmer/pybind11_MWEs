
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

//#include "kernels.h"
namespace constants
{
   constexpr float PI = 22/7;
}

namespace kernels
{
   std::vector<float> gaussian(int kernel_size, float sigma)
   {
       std::vector<float> kernel(kernel_size*kernel_size,0);
       
       if (sigma <=0 ){
           sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8;
       }
       float s = 2.0 * sigma * sigma;
   
       // sum is for normalization
       float sum = 0.0;
   
       // generating nxn kernel
       int i,j;
       float mean = kernel_size/2;
       for (i=0 ; i<kernel_size ; i++) {
           for (j=0 ; j<kernel_size ; j++) {
               kernel[(i*kernel_size)+j] =exp( -0.5 * (pow((i-mean)/sigma, 2.0) + pow((j-mean)/sigma,2.0)) ) / (2 * constants::PI * sigma * sigma);
               sum += kernel[(i*kernel_size)+j];
           }
       }
   
       // normalising the Kernel
       for (int i = 0; i < kernel.size(); ++i){
           kernel[i] /= sum;
       }
   
       return kernel;
   }
}

float apply_kernel(
   const float* input,
   const std::vector<float>& kernel,
   int row, int col, int img_rows, 
   int kernel_size
){
   int k_ind{0};
   float sum{0};
   for (int i{-kernel_size / 2}; i <= (kernel_size / 2); ++i)
   {
      for (int j{-kernel_size / 2}; j <= (kernel_size / 2); ++j)
      {
         // The operation should be done on images with range [0,1]
         sum += kernel[k_ind] * (input[(row + i) * img_rows + (col + j)]);
         ++k_ind;
      }
   }
   return sum;
}

void apply_filter(
   float* output,
   float* input,
   std::vector<float>& kernel,
   int img_cols, int img_rows, 
   int kernel_size
){
   for (int row{kernel_size / 2}; row < (img_cols - kernel_size / 2); ++row)
   {
      for (int col{kernel_size / 2}; col < (img_rows - kernel_size / 2); ++col)
      {
         output[img_rows * row + col] = apply_kernel(input, kernel, row, col, img_rows, kernel_size);
      }
   }
}

// Interface

namespace py = pybind11;

// wrap C++ function with NumPy array IO
py::object wrapper(
   py::array_t<float> output,
   py::array_t<float> input,
   int kernel_size = 3,
   float sigma = 1
){
   // check input dimensions
   if ( output.ndim() != 2 )
      throw std::runtime_error("Ouput should be 2-D NumPy array");
   if ( input.ndim() != output.ndim() )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = output.request();
   auto buf2 = input.request();
   if (buf1.size != buf2.size) 
      throw std::runtime_error("sizes do not match!");

   int N = input.shape()[0], M = input.shape()[1];

   float* ptr_out = (float*) buf1.ptr;
   float* ptr_in = (float*) buf2.ptr;

   auto GKernel = kernels::gaussian(kernel_size, sigma);
   
   // call pure C++ function
   apply_filter(
      ptr_out,
      ptr_in,
      GKernel,
      N, M, 
      kernel_size
   );
   return py::cast<py::none>(Py_None);
}

PYBIND11_MODULE(example_filters,m) {
  m.doc() = "Bindings for convolution filters written in c++.";
  m.def("gaussian_filter", &wrapper, "Apply a gaussian filter to a 2D array");
}

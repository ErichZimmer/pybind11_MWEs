#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "lib/kernels.h"
#include "lib/filters.h"

// Interface
namespace py = pybind11;

// wrap C++ function with NumPy array IO

py::array_t<float> intensity_cap_wrapper(intensity_binarize_wrapper
   py::array_t<float> input,
   float std_mult = 2.f
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();
   
   int N = input.shape()[0], M = input.shape()[1];

   py::array_t<float> result = py::array_t<float>(buf1.size);
   auto buf2 = result.request();
   
   float* ptr_in  = (float*) buf1.ptr;
   float* ptr_out = (float*) buf2.ptr;
   
   // call pure C++ function
   intensity_cap_filter(
      ptr_out,
      ptr_in,
      N*M, 
      std_mult
   );
   
   result.resize({N,M});
   
   return result;
}

py::array_t<float> intensity_binarize_wrapper(
   py::array_t<float> input,
   float threshold = 0.5
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();
   
   int N = input.shape()[0], M = input.shape()[1];

   py::array_t<float> result = py::array_t<float>(buf1.size);
   auto buf2 = result.request();
   
   float* ptr_in  = (float*) buf1.ptr;
   float* ptr_out = (float*) buf2.ptr;
   
   // call pure C++ function
   binarize_filter(
      ptr_out,
      ptr_in,
      N*M, 
      threshold
   );
   
   result.resize({N,M});
   
   return result;
}

py::array_t<float> low_pass_filter_wrapper(
   py::array_t<float> input,
   int kernel_size = 3,
   float sigma = 1
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();
   
   int N = input.shape()[0], M = input.shape()[1];

   py::array_t<float> result = py::array_t<float>(buf1.size);
   auto buf2 = result.request();
   
   float* ptr_in  = (float*) buf1.ptr;
   float* ptr_out = (float*) buf2.ptr;
   
   int kernel_type = 0; // gaussian kernel
   auto GKernel = kernels::get_kernel_type(kernel_type)(kernel_size, sigma);

   // call pure C++ function
   apply_kernel_lowpass(
      ptr_out,
      ptr_in,
      GKernel,
      N, M, 
      kernel_size
   );
   
   result.resize({N,M});
   
   return result;
}

py::array_t<float> high_pass_filter_wrapper(
   py::array_t<float> input,
   int kernel_size = 3,
   float sigma = 1,
   py::bool_ clip_at_zero = false
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();
   
   int N = input.shape()[0], M = input.shape()[1];

   py::array_t<float> result = py::array_t<float>(buf1.size);
   auto buf2 = result.request();
   
   float* ptr_in  = (float*) buf1.ptr;
   float* ptr_out = (float*) buf2.ptr;
   
   int kernel_type = 0; // gaussian kernel
   auto GKernel = kernels::get_kernel_type(kernel_type)(kernel_size, sigma);

   // call pure C++ function
   apply_kernel_highpass(
      ptr_out,
      ptr_in,
      GKernel,
      N, M, 
      kernel_size,
      clip_at_zero
   );
   
   result.resize({N,M});
   
   return result;
}

py::array_t<float> local_variance_norm_wrapper(
   py::array_t<float> input,
   int kernel_size = 3,
   float sigma1 = 2,
   float sigma2 = 2,
   py::bool_ clip_at_zero = false
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();

   int N = input.shape()[0], M = input.shape()[1];
   
   py::array_t<float> result = py::array_t<float>(buf1.size);
   auto buf2 = result.request();
   
   py::array_t<float> temp_buffer = py::array_t<float>(buf1.size);
   auto buf3 = temp_buffer.request();
   
   float* ptr_out = (float*) buf2.ptr;
   float* ptr_in  = (float*) buf1.ptr;
   float* ptr_buf = (float*) buf3.ptr;
   
   // call pure C++ function
   local_variance_norm(
      ptr_out,
      ptr_in,
      ptr_buf,
      N, M, 
      kernel_size,
      sigma1,
      sigma2,
      clip_at_zero
   );
   
   result.resize({N,M});
   
   return result;
}

void mult_scal(
   float* output,
   float* input,
   const int constant,
   int N, int M
){
   int step = M;
   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < M; ++j)
      {
         output[step * i + j] = input[step * i + j] * constant;
      }
   }
//   float test_mean = std::accumulate(std::begin(input), std::end(output), 0.0)/(N*M);
}

py::array_t<float> test_wrapper(
   py::array_t<float> input,
   int testConst
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();
   
   int N = input.shape()[0], M = input.shape()[1];

   py::array_t<float> result = py::array_t<float>(buf1.size);
   auto buf2 = result.request();
   
   float* ptr_in  = (float*) buf1.ptr;
   float* ptr_out = (float*) buf2.ptr;

   // call pure C++ function
   mult_scal(
      ptr_out,
      ptr_in,
      N, M, 
      testConst
   );
   
   result.resize({N,M});
   
   return result;
}


PYBIND11_MODULE(piv_filters_core,m) {
   m.doc() = "Python interface for filters written in c++.";
   
   m.def("_test_wrapper",
      &test_wrapper, 
      "Test wrapper by multiplying a scalar to an array."
   );
   m.def("_intensity_cap", 
      &intensity_cap_wrapper,
      "Apply an intensity cap filter to a 2D array"
//       py::arg("std_mult") = 2.f
   );
   m.def("_threshold_binarization", 
      &intensity_binarize_wrapper,
      "Apply an binarization filter to a 2D array"
//       py::arg("threshold") = 2.f
   );
   m.def("_gaussian_lowpass_filter", 
      &low_pass_filter_wrapper,
      "Apply a gaussian low pass filter to a 2D array"
//       py::arg("kernel_size") = 3, 
//       py::arg("sigma") = 1
   );
   m.def("_gaussian_highpass_filter", 
      &high_pass_filter_wrapper,
      "Apply a gaussian high pass filter to a 2D array"
//      py::arg("kernel_size") = 3, 
//      py::arg("sigma") = 1,
//      py::arg("clip_at_zero") = false
   );
   m.def("_local_variance_normalization", 
      &local_variance_norm_wrapper, 
      "Apply a local variance normalization filter to a 2D array"
//      py::arg("kernel_size") = 3, 
//      py::arg("sigma1") = 2,
//      py::arg("sigma2") = 2,
//      py::arg("clip_at_zero") = false
   );
}

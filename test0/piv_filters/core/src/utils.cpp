#include <algorithm>

#include "lib/utils.h" 

imgDtype buffer_find_min(
   imgDtype* in,
   std::size_t N
){
   std::size_t i{};
   imgDtype buff_min{ 2.f };
   
   for (i = 0; i < N; ++i)
      buff_min = (in[i] < buff_min) ? in[i] : buff_min;
   
   return buff_min;
}

imgDtype buffer_find_max(
   imgDtype* in,
   std::size_t N
){
   std::size_t i{};
   imgDtype buff_max{ -2.f };
   
   for (i = 0; i < N; ++i)
      buff_max = (in[i] > buff_max) ? in[i] : buff_max;
   
   return buff_max;
}

void buffer_divide_scalar(
   imgDtype* in,
   imgDtype scalar, // Implicit conversion 
   std::size_t N
){
   std::size_t i{};
   
   for (i = 0; i < N; ++i)
      in[i] /= scalar;
}

void buffer_p_norm(
   imgDtype* in, 
   std::size_t N
){
   imgDtype buff_max{ buffer_find_max(in, N) };
   buffer_divide_scalar(in, buff_max, N);
}

void buffer_clip(
   imgDtype* in, 
   imgDtype lower, // implicit conversion
   imgDtype upper, // implicit conversion
   std::size_t N
){
   std::size_t i{};
   
   for (i = 0; i < N; ++i)
   {
      if (in[i] > upper)
         in[i] = upper;
      
      if (in[i] < lower)
         in[i] = lower;
   }
}

std::vector<imgDtype> buffer_mean_std(
   imgDtype* in, 
   std::size_t N_M
){
   imgDtype sum{}, mean{}, std_{};
   std::size_t i{};
   
   for (i = 0; i < N_M; ++i)
   {
      sum += in[i];
      std_ += in[i]*in[i]; // temp
   }
   mean = sum / N_M;
   std_ = sqrt( (std_ / N_M) + (mean*mean) - (2*mean*mean) );
   
   std::vector<imgDtype> out(2);
   out[0] = mean; out[1] = std_;
   
   return out;
}

std::int32_t sub2Dind(
   std::int32_t x, 
   std::int32_t y,
   std::int32_t yStep
){
   return (y * yStep) + x;
}

std::int32_t sub3Dind(
   std::int32_t x, 
   std::int32_t y, 
   std::int32_t z, 
   std::int32_t yStep, 
   std::int32_t zStep
){
   return (z*yStep*zStep) + sub2Dind(x, y, yStep);
}

/*
imgDtype vector_median(
   imgDtype& kernel,
   std::size_t N,
   std::size_t offset = 0 // for nan values
){
   std::nth_element(
      std::begin(kernel),
      std::begin(kernel) + N / 2,
      std::end(kernel)
   );
      
   return (imgDtype)kernel[(N / 2) + offset];
   
}
*/
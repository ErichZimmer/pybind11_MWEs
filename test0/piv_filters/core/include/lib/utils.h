#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <vector> 

using imgDtype = float;

imgDtype buffer_find_min(
   imgDtype*,
   std::size_t
);

imgDtype buffer_find_max(
   imgDtype*,
   std::size_t
);

void buffer_divide_scalar(
   imgDtype*,
   imgDtype, // Implicit conversion 
   std::size_t
);

void buffer_p_norm(
   imgDtype*, 
   std::size_t
);

void buffer_clip(
   imgDtype*, 
   imgDtype, // implicit conversion
   imgDtype, // implicit conversion
   std::size_t
);

std::vector<imgDtype> buffer_mean_std(
   imgDtype*, 
   std::size_t
);

std::int32_t sub2Dind(
   std::int32_t, 
   std::int32_t,
   std::int32_t
);

std::int32_t sub3Dind(
   std::int32_t, 
   std::int32_t, 
   std::int32_t, 
   std::int32_t, 
   std::int32_t
);
/*
imgDtype vector_median(
   imgDtype&,
   std::size_t,
   std::size_t
)
*/
#endif
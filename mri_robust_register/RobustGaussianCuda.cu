/**
 * @file RobustGaussianCuda.cu
 * @brief A class to esimate a robust Gaussian (using median and mad) on the GPU
 *
 */

/*
 * Original Author: Martin Reuter
 * CVS Revision Info:
 *    $Author: mreuter $
 *    $Date: 2012/09/21 23:05:16 $
 *    $Revision: 1.17 $
 *
 * Copyright © 2016 The General Hospital Corporation (Boston, MA) "MGH"
 *
 * Terms and conditions for use, reproduction, distribution and contribution
 * are found in the 'FreeSurfer Software License Agreement' contained
 * in the file 'LICENSE' found in the FreeSurfer distribution, and here:
 *
 * https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense
 *
 * Reporting: freesurfer@nmr.mgh.harvard.edu
 *
 */
#include "RobustGaussianCuda.h"

#include <thrust/version.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

// fabsdiff<T> computes the abs of the difference x - (const) a
template <typename T>
struct fabsdiff
{
    const T a;

    fabsdiff(T _a) : a(_a) {}
    
    __host__ __device__
        T operator()(const T& x) const { 
            return fabs(x - a);
        }
};


template<class T> 
T RobustGaussianCuda<T>::kth_smallest(thrust::device_vector<T>& v, int k)
// sorts v (and returns sorted v via reference)
// returns kth smallest (k = 1,...,N)
{
  thrust::sort(v.begin(), v.end());    
  return v[k-1];
}   

template<class T>
T RobustGaussianCuda<T>::kth_smallest(T a[], int n, int k)
{
  // transfer to device
  thrust::device_vector<T> d_a(a, a + n);
  return kth_smallest(d_a,k);
}   

template<class T>
std::pair<T, int> RobustGaussianCuda<T>::kth_smallestI(thrust::device_vector<T>& d_v, int k)
// sorts v (and returns sorted v via reference)
// returns kth smallest (k = 1,...,N)
{

  thrust::device_vector<int> d_idx(d_v.size());
  thrust::sequence(d_idx.begin(), d_idx.end());
  thrust::sort_by_key(d_v.begin(), d_v.end(), d_idx.begin());

  return std::pair< T, int>(d_v[k-1],d_idx[k-1]);
}   

template<class T>
std::pair<T, int> RobustGaussianCuda<T>::kth_smallestI(T a[], int n, int k)
{
  // transfer to device
  thrust::device_vector<T> d_a(a, a + n);
  return kth_smallestI(d_a,k);
}


template<typename T>
T RobustGaussianCuda<T>::median(thrust::device_vector<T>& v)
{
  thrust::sort(v.begin(), v.end()); 
  unsigned int n = v.size();
  if (n % 2 == 1) //odd
    return v[ (n - 1) / 2];
  else
    return 0.5 * (v[ n / 2] + v[ n / 2 - 1]);
}   

template<class T>
T RobustGaussianCuda<T>::median(T a[], int n)
{
  // transfer to device
  thrust::device_vector<T> d_a(a, a + n);
  
  return median(d_a);
}   

template<typename T>
T RobustGaussianCuda<T>::mad(thrust::device_vector<T>& v, float d)
//default d = 1.4826
// v gets modified 
{
  // compute median
  T medi = median(v);
  
  // compute v[i] = fabs(v[i] - medi) using unary functor
  thrust::transform(v.begin(), v.end(), v.begin(), fabsdiff<T>(medi));
  
  // compute median on that
  medi = median(v);

  return medi * d;
}   

template<class T>
T RobustGaussianCuda<T>::mad(T a[], int n, float d)
//default d = 1.4826
{
  // transfer to device
  thrust::device_vector<T> d_a(a, a + n);

  return mad(d_a,d);
}   


template<class T>
T RobustGaussianCuda<T>::mad(const T a[], int n, float d)
//default d = 1.4826
{
  //std::cout << " call to CUDA MAD ... " << std::endl;
  // transfer to device
  thrust::device_vector<T> d_a(a, a + n);

  return mad(d_a,d);
}   

template class RobustGaussianCuda<float>;
template class RobustGaussianCuda<double>;

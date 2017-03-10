/**
 * @file RobustGaussian.h
 * @brief A class to esimate a robust Gaussian (using median and mad) on the GPU
 *
 */

/*
 * Original Author: Martin Reuter
 * CVS Revision Info:
 *    $Author: mreuter $
 *    $Date: 2012/09/21 23:05:16 $
 *    $Revision: 1.10 $
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
#ifndef RobustGaussianCuda_H
#define RobustGaussianCuda_H

#include <utility> 

// forward declaration of thrust vectors to avoid including anything here
namespace thrust
{
  template<typename T> class device_malloc_allocator;
  template<typename T, typename Alloc> class host_vector;
  template<typename T, typename Alloc> class device_vector;
}


/** \class RobustGaussianCuda
 * \brief A static class with routines for robust compuations (median, MAD, ...) on the GPU using thust::sort
 */
template<class T>
class RobustGaussianCuda
{
public:
  
  //! Find k-th smallest of array a with length n
  static T kth_smallest(T a[], int n, int k);
  //! Find k-th smallest of array a with length n
  static T kth_smallest(thrust::device_vector<T,thrust::device_malloc_allocator<T> >& v, int k);  
  //! Find k-th smallest of array a with length n 
  static T quick_select(T a[], int n, int k)
  { return kth_smallest(a,n,k); }
  
  //! Find k-th smallest of array a with length n and the index
  static std::pair<T, int> kth_smallestI(T a[], int n, int k);
  //! Find k-th smallest of array a with length n and the index
  static std::pair<T, int> kth_smallestI(thrust::device_vector<T,thrust::device_malloc_allocator<T> >& v, int k);
  //! Find k-th smallest of array a with length n and the index 
  static std::pair<T, int> quick_selectI(T a[], int n, int k)
  { return kth_smallestI(a,n,k); }

  //! Find median of array a with length n
  static T median(thrust::device_vector<T,thrust::device_malloc_allocator<T> >& v);
  //! Find median of array a with length n
  static T median(T a[], int n);
  
  //! Find median absolute deviation
  static T mad(thrust::device_vector<T,thrust::device_malloc_allocator<T> >& v, float d = 1.4826);
  //! Find median absolute deviation
  static T mad(T a[], int n, float d = 1.4826);
  //! Find median absolute deviation
  static T mad(const T a[], int n, float d = 1.4826);
  
};

//#include "RobustGaussianCuda.cu"
//instead instantiate in the .cu file, to allow compilation with nvcc

#endif

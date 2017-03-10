/**
 * @file RegressionCuda.h
 * @brief A class to solve overconstrained system A X = b 
 *
 *   it uses either least squares (standard regression)
 *   or a robust estimator (Tukey's Biweight with iterative reweighted least
 *   squares) on GPU
 *
 */

/*
 * Original Author: Martin Reuter
 * CVS Revision Info:
 *    $Author: mreuter $
 *    $Date: 2016/03/10 15:14:00 $
 *    $Revision: 1.18 $
 *
 * Copyright © 2011 The General Hospital Corporation (Boston, MA) "MGH"
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
//
//
// written by Martin Reuter
// March 16th ,2016
//
#ifndef RegressionCuda_H
#define RegressionCuda_H

#define SATr 4.685  // this is suggested for gaussian noise
#include <utility>
#include <string>
#include <cassert>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>

// forward declaration of thrust vectors to avoid including anything here
namespace thrust
{
  template<typename T> class device_malloc_allocator;
  template<typename T, typename Alloc> class host_vector;
  template<typename T, typename Alloc> class device_vector;
}

struct cublasContext;
typedef struct cublasContext *cublasHandle_t;


/** \class RegressionCuda
 * \brief Templated class for iteratively reweighted least squares
 */
template<class T>
class RegressionCuda
{
public:

  //! Constructor initializing A and b
  RegressionCuda(vnl_matrix<T> & Ap, vnl_vector<T> & bp);

  //! Constructor initializing b (for simple case where x is single variable and A is (...1...)^T
  //RegressionCuda(vnl_vector<T> & bp);
  
  //! Destuctor (delete GPU mem and destroy blas handle)
  ~RegressionCuda();

  //! Robust solver
//  vnl_vector<T> getRobustEst(double sat = SATr, double sig=1.4826);
  //! Robust solver (returning also the sqrtweights)
  vnl_vector<T> getRobustEstW(vnl_vector<T>&w, double sat=SATr, double sig=1.4826);

  //! Least Squares
//  vnl_vector<T> getLSEst();
  //! Weighted least squares
//  vnl_vector<T> getWeightedLSEst(const vnl_vector<T> & sqrtweights);
  //! Weighted least squares in float (only for the T=double version)
  //vnl_vector<T> getWeightedLSEstFloat(const vnl_vector<T> & sqrtweights);
  void getWeightedLSEstFloat(const thrust::device_vector<T,thrust::device_malloc_allocator<T> >& d_w,
         thrust::device_vector<T,thrust::device_malloc_allocator<T> >& d_p);

  double getLastError()
  {
    return lasterror;
  }

  double getLastWeightPercent()
  {
    return lastweight;
  }

  double getLastZeroWeightPercent()
  {
    return lastzero;
  }

  //! Set verbose level
  void setVerbose(int v)
  {
    verbose = v;
    if (v < 0)
      verbose = 0;
    if (v > 2)
      verbose = 2;
  }
  
  //! Specify if SVD is float (also in double case)
  void setFloatSvd(bool b)
  {
    floatsvd = b;
  }

//  void plotPartialSat(const std::string& fname);

protected:

  vnl_vector<T> getRobustEstWAB(vnl_vector<T>&w, double sat = SATr, double sig = 1.4826);
//  double getRobustEstWB(vnl_vector<T>&w, double sat = SATr, double sig = 1.4826);

//  T getSigmaMAD(const vnl_vector<T>& r, T d = 1.4826);
//  T VectorMedian(const vnl_vector<T>& v);


  void getSqrtTukeyDiaWeights(const thrust::device_vector<T,thrust::device_malloc_allocator<T> >& d_r,
    thrust::device_vector<T,thrust::device_malloc_allocator<T> >& d_w, double sigma, double sat = SATr);

//  void getTukeyBiweight(const vnl_vector<T>& r, vnl_vector<T> &w, double sat = SATr);
//  double getTukeyPartialSat(const vnl_vector<T>& r, double sat = SATr);


private:

  vnl_matrix<T> * A;
  unsigned int arows;
  unsigned int acols;
  vnl_vector<T> * b;
  double lasterror, lastweight, lastzero;
  int verbose;
  bool floatsvd;
  
  cublasHandle_t handle;
  
  thrust::device_vector<T,thrust::device_malloc_allocator<T> > *d_A;
  thrust::device_vector<T,thrust::device_malloc_allocator<T> > *d_b;
  thrust::device_vector<T,thrust::device_malloc_allocator<T> > *d_Atmp;
  thrust::device_vector<T,thrust::device_malloc_allocator<T> > *d_btmp;
  thrust::device_vector<T,thrust::device_malloc_allocator<T> > *d_wAtwA;
  thrust::device_vector<T,thrust::device_malloc_allocator<T> > *d_wAtwb;
  
  //! cublas wrapper that computes d_r := d_r - d_A * d_p
  void getResiduals(cublasHandle_t &handle, const T *d_A, const T *d_p, T *d_r,
                    const int arows, const int acols);
  
  //! cublas wrapper for out of place transpose  d_At := d_A^T                         
  void transpose(cublasHandle_t &handle, const T *d_A, T *d_At,
                 const int arows, const int acols);
  
};



#endif

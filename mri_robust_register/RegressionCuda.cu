/**
 * @file RegressionCuda.cu
 * @brief Templated class for iteratively reweighted least squares on GPU
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

#include "RegressionCuda.h"
#include "RobustGaussianCuda.h"

#include <iostream>
#include <iomanip>
#include <cassert>
#include <limits>
#include <fstream>
#include <timer.h>

#include <cublas_v2.h>
#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>

#include <vnl/algo/vnl_cholesky.h>
#include <vnl/algo/vnl_qr.h>
#include <vnl/vnl_matlab_print.h>
#include <vcl_iostream.h>



template<class T>
void saveMatrix(const std::string& fname, const vnl_matrix<T> * M)
{
  int m = M->rows();
  int n = M->cols();
  std::ofstream myfile (fname.c_str(), std::ios::out | std::ios::binary);
  myfile.write(reinterpret_cast<char *>(&m),sizeof(int));
  myfile.write(reinterpret_cast<char *>(&n),sizeof(int));  
  myfile.write(reinterpret_cast<const char *>(M->data_block()), sizeof(T)*m*n);
  myfile.close();
}


template<class T>
void saveVector(const std::string& fname, const vnl_vector<T> * v)
{
  std::ofstream myfile (fname.c_str(), std::ios::out | std::ios::binary);
  int m = v->size();
  int n = 1;
  myfile.write(reinterpret_cast<char *>(&m),sizeof(int));
  myfile.write(reinterpret_cast<char *>(&n),sizeof(int));  
  myfile.write(reinterpret_cast<const char *>(v->data_block()), sizeof(T)*m);
  myfile.close();
}

template<class T>
void saveVector(const std::string& fname, const thrust::device_vector<T>& d_v)
{
  // copy off GPU:
  vnl_vector <T> v (d_v.size());
  thrust::copy(d_v.begin(), d_v.end(), v.data_block());
  std::ofstream myfile (fname.c_str(), std::ios::out | std::ios::binary);
  int m = v.size();
  int n = 1;
  myfile.write(reinterpret_cast<char *>(&m),sizeof(int));
  myfile.write(reinterpret_cast<char *>(&n),sizeof(int));  
  myfile.write(reinterpret_cast<const char *>(v.data_block()), sizeof(T)*m);
  myfile.close();
}

template<class T>
void loadMatrix(const std::string& fname, int & m, int & n, vnl_matrix<T> & M)
{
  int im, in;
  std::streampos begin,end,size;
  std::ifstream file (fname.c_str(), std::ios::in|std::ios::binary);
  char * memblock;
  if (file.is_open())
  {
    file.read(reinterpret_cast<char *>(&im),sizeof(int));
    file.read(reinterpret_cast<char *>(&in),sizeof(int));
    begin = file.tellg();
    file.seekg (0, std::ios::end);
    end = file.tellg();
    size = end-begin;
    std::cout << " m: " << im << " n: " << in << "  size: " << size <<  "  element_size: " << size/(im*in) << "  type size: " << sizeof(T)<< std::endl;
    assert ((int)size == (int)( sizeof(T)*in)*im);

    m = im;
    n = in;
        
    //memblock = new char [size];
    M.resize(m,n);
    memblock = reinterpret_cast<char *>(M.data_block());
    file.seekg (begin, std::ios::beg);
    file.read (memblock, size);
    file.close();

   /* // still transposed (if both m, n > 1)
    T * M = reinterpret_cast<float *>(memblock);
    
    if (m==1 || n==1) return M;
    
    // here we need to transpose:
    float* Mt;
    TESTING_MALLOC_CPU( Mt,    float, lda*n     );    
    magma_int_t i, j;
    for( j=0; j < n; ++j ) {
        for( i=0; i < m; ++i ) {
            Mt[i + j * lda] = M [ i * in + j]; 
        }
    }
//   magmablas_stranspose(n,m,M,n,Mt,m);   // did not work
    cout << " element 100 4 : " << Mt [ 100 + 4 * lda ] << endl;
    cout << " element 0 0 : " << Mt [ 0 ] << endl;
    cout << " element 0 1 : " << Mt [ 0 + 1 *lda] << endl;
    cout << " element 1 0 : " << Mt [ 1 + 0 *lda] << endl;
//    cout << " element 100 4 : " << M [ 100 *in + 4  ] << endl;
    delete M;
    
    return Mt;
*/
  }
  //return 0;
}


template<class T>
void loadMatrix( const std::string& fname, int & m, vnl_vector<T> & v)
{
  int im, in;
  std::streampos begin,end,size;
  std::ifstream file (fname.c_str(), std::ios::in|std::ios::binary);
  char * memblock;
  if (file.is_open())
  {
    file.read(reinterpret_cast<char *>(&im),sizeof(int));
    file.read(reinterpret_cast<char *>(&in),sizeof(int));
    begin = file.tellg();
    file.seekg (0, std::ios::end);
    end = file.tellg();
    size = end-begin;
    std::cout << " m: " << im << " n: " << in << "  size: " << size <<  "  element_size: " << size/(im*in) << "  type size: " << sizeof(T)<< std::endl;
    assert ((int)size == (int)( sizeof(T)*in)*im);

    m = im;
    
    assert(in==1);
        
    //memblock = new char [size];
    v.resize(m);
    memblock = reinterpret_cast<char *>(v.data_block());
    file.seekg (begin, std::ios::beg);
    file.read (memblock, size);
    file.close();
  }
}    


/**********************/
/* cuBLAS ERROR CHECK */
/**********************/
#ifndef cublasSafeCall
#define cublasSafeCall(err)     __cublasSafeCall(err, __FILE__, __LINE__)
#endif

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
  if( CUBLAS_STATUS_SUCCESS != err)
  {
    std::cerr << "CUBLAS error in file '" << __FILE__ <<"', line "<<__LINE__<<"\n \nerror "<<err<<" \nterminating!\n"; 
    cudaDeviceReset(); assert(0); 
  }
}



/**********************/
/* Thrust helper      */
/**********************/

// fabsdiff<T> computes weights (1- x^2 / (sigma*sat)^2) if fabs(x) < (sigma*sat), else 0
template <typename T>
struct getweights
{
    const T sigmasat;

    getweights(T _sigmasat) : sigmasat(_sigmasat) {}
    
    __host__ __device__
        T operator()(const T& x) const { 
          if (fabs(x) >= sigmasat)
            return 0.0;
          else
          {
            T t = (x/sigmasat);
            return 1.0 - t * t; // this is equal to sqrt of weights (see paper)
          }
        }
};

// square<T> unary function computing the square x*x
template <typename T>
struct square : public thrust::unary_function<T,T>
{
  __host__ __device__
  float operator()(T x) const
  {
    return x * x;
  }
};

// prodsquare<T> binary function computing the product of squares x*x*y*y
template <typename T>
struct prodsquare : public thrust::binary_function<T,T,T>
{
  __host__ __device__
  float operator()(T x, T y) const
  {
    return x * x * y * y;
  }
};




/**********************/
/* Member definitions */
/**********************/

//! Constructor initializing A and b
template<class T>
RegressionCuda<T>::RegressionCuda(vnl_matrix<T> & Ap, vnl_vector<T> & bp) :
      A(&Ap), arows(Ap.rows()), acols(Ap.columns()), b(&bp), lasterror(-1), lastweight(-1), lastzero(-1), verbose(1), floatsvd(false)
{
  //struct timeb start;
  //TimerStart(&start);
  int m = arows;
  int n = acols;
  int N = n*m;
  assert(bp.size() == m);

  // allocate gpu space
  d_Atmp = new thrust::device_vector<T>(Ap.data_block(), Ap.data_block() + N); 
  d_A    = new thrust::device_vector<T> (N);

  // init cublas
  cublasSafeCall(cublasCreate(&handle)); 

  // transpose A  (could be dropped in future, when we work with A^T directly, but not worth the time savings)
  transpose(handle,thrust::raw_pointer_cast(d_Atmp->data()),thrust::raw_pointer_cast(d_A->data()), arows, acols);
  
  // allocate more gpu space and set d_b
  d_b     = new thrust::device_vector<T>(bp.data_block(), bp.data_block() + bp.size());
  d_btmp  = new thrust::device_vector<T>(arows);
  d_wAtwA = new thrust::device_vector<T>(acols*acols);
  d_wAtwb = new thrust::device_vector<T>(acols);
  //std::cout << std::endl << "time construct: " << ((float) TimerStop(&start) / 1000.0f) << std::endl;
}



//! Constructor initializing b (for simple case where x is single variable and A is (...1...)^T
//template<class T>
//RegressionCuda<T>::RegressionCuda(vnl_vector<T> & bp) :
//      A(NULL), b(&bp), lasterror(-1), lastweight(-1), lastzero(-1), verbose(1), floatsvd(false)
//{
//  //d_b.assign(bp.data_block(), bp.data_block() + bp.size());
//  d_b     = new thrust::device_vector<T>(bp.data_block(), bp.data_block() + bp.size());
//}


template<class T>
RegressionCuda<T>::~RegressionCuda()
{
  delete(d_A);
  delete(d_Atmp);
  delete(d_b);
  delete(d_btmp);
  delete(d_wAtwA);
  delete(d_wAtwb);
  cublasSafeCall(cublasDestroy(handle));
}


template<class T>
vnl_vector<T> RegressionCuda<T>::getRobustEstW(vnl_vector<T>& w, double sat, double sig)
{
  if (A)
    return getRobustEstWAB(w, sat, sig);
  else
  {
    //return vnl_vector<T>(1, getRobustEstWB(w, sat, sig));
    std::cerr << "RegressionCuda<T>::getRobustEstWB not implemented yet!" << std::endl;
    exit(1);
  }
    
}


/** Solves overconstrained system A p = b using
 M estimators (Beaton and Tukey's biweigt function).
 Returns vectors p and w (w by reference as it is large)
 where p is the robust solution and w the used weights.
 Method: iterative reweighted least squares
 */
template<class T>
vnl_vector<T> RegressionCuda<T>::getRobustEstWAB(vnl_vector<T>& wfinal, double sat, double sig)
{
  if (verbose > 1)
  {
    std::cout << "  RegressionCuda<T>::getRobustEstWAB( "<<sat<<" , "<<sig<<" ) " ;
    if (floatsvd) std::cout << "  FLOAT version " ;
    else std::cout << "  DOUBLE version " ;
    std::cout << std::endl;
  }
//saveMatrix("A.bin", A);
//saveVector("b.bin", b);

  //  struct timeb start;
  //  int msec;
  //  TimerStart(&start);
  
  // constants
  const int MAXIT = 20;
  //double EPS = 2e-16;
  const double EPS = 2e-12;

  // INIT CUBLAS!! // done in constructor
  //cublasHandle_t handle;
  //cublasCreate(&handle);
  
  // variables
  std::vector<T> err(MAXIT + 1);
  //thrust::device_vector<T> d_err(MAXIT+1);
  err[0] = std::numeric_limits<T>::infinity();
  err[1] = 1e20;
  double sigma;

//  const int arows = A->rows(); // large (voxels)
//  const int acols = A->cols(); // small (parameters)

  //pre-alocate vectors
  // init residuals (based on zero p, so r := b )
  thrust::device_vector<T> d_r(*d_b);
  thrust::device_vector<T> d_p1(acols);
  thrust::device_vector<T> d_w1(arows);

  thrust::device_vector<T> d_p2(acols);
  thrust::device_vector<T> d_w2(arows);

  //vnl_vector<T> *vtmp = NULL;
  thrust::device_vector<T> *d_vtmp  = NULL;
  thrust::device_vector<T> *d_p     = &d_p1;
  thrust::device_vector<T> *d_lastp = &d_p2;
  thrust::device_vector<T> *d_w     = &d_w1;
  thrust::device_vector<T> *d_lastw = &d_w2;
   
  int count = 0;
  int incr = 0;
  // iteration until we increase the error, we reach maxit or we have no error
  do
  {
    //struct timeb startloop;
    //TimerStart(&startloop);
    
    count++; //first = 1

    if (count > 1)
    {
      // store lastp (not necessary in first run)
      // instead of copy, we swap pointers:
      d_vtmp = d_lastp;
      d_lastp = d_p;
      d_p = d_vtmp;

      // store last weights
      // instead of copy, we swap pointers:
      d_vtmp = d_lastw;
      d_lastw = d_w;
      d_w = d_vtmp;
    }
    
    // normalize r and compute weights (or rather w = sqrt of weights)
    //sigma = getSigmaMAD(*r);
    thrust::device_vector<T> d_rtmp(d_r); // copy as mad will sort it
    sigma = RobustGaussianCuda<T>::mad(d_rtmp,sig);
    //std::cout << "Sigma: " << sigma << std::endl;
    if (sigma < EPS) // e.g. if images are identical
    {
      std::cout << "  Sigma too small: " << sigma << " (identical images?)" << std::endl;
      thrust::fill(d_w->begin(), d_w->end(),1.0);
    }
    else
    {
       //*r *= (1.0 / sigma); // now divide by sigma inside getSqrtTukeyDiaWeights (need to probably adjust other calling functions)
      // here we get sqrt of weights into w
      getSqrtTukeyDiaWeights(d_r, *d_w, sigma, sat);
    }
    // free residuals (to reduce max memory load) seems to have no effect on speed
    //r->clear();

    // compute weighted least squares
    //struct timeb start2;
    //int msec2;
    //TimerStart(&start2);
//    if (floatsvd)
      getWeightedLSEstFloat(*d_w, *d_p);

//    else
//      *p = getWeightedLSEst(*w);
    //msec2 = TimerStop(&start2);
    //cout << endl << "time2: " << ((float) msec2 / 1000.0f) << endl;
    //vnl_matlab_print(vcl_cerr,*p,"p",vnl_matlab_print_format_long);std::cerr << std::endl;
    //for (int i = 0;i<acols;i++)
    //  std::cout << d_p->operator[](i) << " ";
    //std::cout << std::endl;
   
    // compute new residuals
    // *r = *b - (*A * *p);
    // perform one of the matrix-matrix operations   C := alpha*op( A )*op( B ) + beta*C,
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    thrust::copy(d_b->begin(), d_b->end(), d_r.begin());
    getResiduals(handle,  thrust::raw_pointer_cast(&d_A->operator[](0)),thrust::raw_pointer_cast(&(d_p->operator[](0))),thrust::raw_pointer_cast(&d_r[0]),arows, acols);


    // and total errors (using new r)
    // err = sum (w r^2) / sum (w)
    T sw  = thrust::reduce(thrust::make_transform_iterator(d_w->begin(), square<T>()),
                           thrust::make_transform_iterator(d_w->end(),   square<T>()));
    T swr = thrust::inner_product(d_w->begin(),d_w->end(),d_r.begin(),0.0,thrust::plus<T>(),prodsquare<T>());

//    for (unsigned int rr = 0; rr < r->size(); rr++)
//    {
//      T t1 = w->operator[](rr);
//      T t2 = r->operator[](rr);
//      t1 *= t1; // remember w is the sqrt of the weights
//      t2 *= t2;
//      sw += t1;
//      swr += t1 * t2;
//    }
    err[count] = swr / sw;
    //std::cout << "err [ " << count << " ] = " << err[count] << std::endl<< std::endl;
    if (err[count - 1] <= err[count])
      incr++;

    //std::cout << std::endl << "time loop: " << ((float) TimerStop(&startloop) / 1000.0f) << std::endl;
    
  } while (incr < 1 && count < MAXIT && err[count] > EPS);

//  delete (r); // won't be needed below

  // copy back from GPU (depending on error increase in last iteration)
  vnl_vector<T> pfinal(acols);
  wfinal.set_size(arows);
  if (err[count] > err[count - 1])
  {
    // take previous values (since actual values made the error increase)
    // std::cout << " last step was no improvement, taking values : "<<  count-1 << std::endl;
    //pfinal = *lastp;
    //wfinal = *lastw;
    d_vtmp = d_lastw;
    thrust::copy(d_lastp->begin(), d_lastp->end(), pfinal.data_block());
    thrust::copy(d_lastw->begin(), d_lastw->end(), wfinal.data_block());
    if (verbose > 1)
      std::cout << "     Step: " << count - 2 << " ERR: " << err[count - 1] << std::endl;
    lasterror = err[count - 1];
  }
  else
  {
    //pfinal = *p;
    //wfinal = *w;
    d_vtmp = d_w;
    thrust::copy(d_p->begin(), d_p->end(), pfinal.data_block());
    thrust::copy(d_w->begin(), d_w->end(), wfinal.data_block());
    if (verbose > 1)
      std::cout << "     Step: " << count - 1 << " ERR: " << err[count] << std::endl;
    lasterror = err[count];
  }
//  delete (p);
//  delete (w);
//  delete (lastw);
//  delete (lastp);
      
  //  msec = TimerStop(&start);
  //  std::cout << std::endl << "time: " << ((float) msec / 1000.0f) << std::endl;
   
  // compute statistics on weights:
  // dd = average of w_i with abs(b_i) > 0.00001
  // ddcount = number of fabs(b_i) > 0.00001 cases
  // zcount  = number of w_i < 0.1 with abs(b_i) > 0.00001
  // lastzero = zcount / ddcount;
  //double d = 0.0;
  double dd = 0.0;
  double ddcount = 0;
  int zcount = 0;
  T val;
  for (unsigned int i = 0; i < wfinal.size(); i++)
  {
    val = wfinal[i];
    //d += val;
    if (fabs(b->operator[](i)) > 0.00001)
    {
      dd += val;
      ddcount++;
      if (val < 0.1)
        zcount++;
    }
  }
  //d /= wfinal.size();
  dd /= ddcount;
  if (verbose > 1)
    std::cout << "          weights average: " << dd << "  zero: "
        << (double) zcount / ddcount << std::endl;
  //"  on significant b vals ( " << ddcount << " ): " << dd <<endl;
  lastweight = dd;
  lastzero = (double) zcount / ddcount;
//saveVector("p.bin", &pfinal);
//exit(1);
  return pfinal;
}


// wrapper functions around cu blas calls (float and double instantiations)
 
template<>
void RegressionCuda<float>::getResiduals(cublasHandle_t &handle, const float *d_A, const float *d_p, float *d_r,
                                         const int arows, const int acols)
{
  int lda=arows,ldb=acols,ldc=arows;
  const float alf = -1.0;
  const float bet = 1.0;
  const float *alpha = &alf;
  const float *beta = &bet;
  int nrhs = 1;
 
  // Do the actual multiplication
  cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, arows, nrhs, acols, alpha, d_A, lda, d_p, ldb, beta, d_r, ldc));
}

template<>
void RegressionCuda<double>::getResiduals(cublasHandle_t &handle, const double *d_A, const double *d_p, double *d_r,
                                          const int arows, const int acols)
{
  int lda=arows,ldb=acols,ldc=arows;
  const double alf = -1.0;
  const double bet = 1.0;
  const double *alpha = &alf;
  const double *beta = &bet;
  int nrhs = 1;
 
  // Do the actual multiplication
  cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, arows, nrhs, acols, alpha, d_A, lda, d_p, ldb, beta, d_r, ldc));
}


template<>
void RegressionCuda<float>::transpose(cublasHandle_t &handle, const float *d_A, float *d_At,
                                         const int arows, const int acols)
{
  //std::cout << "Transpose array - cuBLAS" << std::endl;
  float alpha = 1.0;
  float beta  = 0.0;
  cublasSafeCall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, arows, acols, &alpha, d_A, acols, &beta, d_A, acols, d_At, arows));
}

template<>
void RegressionCuda<double>::transpose(cublasHandle_t &handle, const double *d_A, double *d_At,
                                         const int arows, const int acols)
{
  //std::cout << "Transpose array - cuBLAS" << std::endl;
  double alpha = 1.0;
  double beta  = 0.0;
  cublasSafeCall(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, arows, acols, &alpha, d_A, acols, &beta, d_A, acols, d_At, arows));
}



/*
// slow (copying large matrices and transposing) for testing only 
template<>
void RegressionCuda<float>::getWeightedLSEstFloat(const thrust::device_vector<float>& d_w,
         thrust::device_vector<float>& d_p)
{         
  float alpha = 1.0;
  float beta  = 0.0;
  assert(d_w.size() == arows);

  //saveMatrix("A.cu.bin", A);
  //saveVector("b.cu.bin", b);
  //saveVector("w.cu.bin", d_w);

  // with w a diagonal matrix
  // first Atemp := diag(w) * A
  cublasSafeCall(cublasSdgmm(handle, CUBLAS_SIDE_LEFT,arows,acols,
                          thrust::raw_pointer_cast(&d_A->operator[](0)), arows,
                          thrust::raw_pointer_cast(&d_w[0]), 1,
                          thrust::raw_pointer_cast(&d_Atmp->operator[](0)), arows));
                           
  // first btemp := diag(w) * b
  cublasSafeCall(cublasSdgmm(handle, CUBLAS_SIDE_LEFT,arows,1,
                          thrust::raw_pointer_cast(&d_b->operator[](0)), arows,
                          thrust::raw_pointer_cast(&d_w[0]), 1,
                          thrust::raw_pointer_cast(&d_btmp->operator[](0)), arows));

  
  // get them from GPU and decompose on CPU
  // (to avoid inclusion of GPU LAPACK libraries)
  vnl_matrix<float> c_wA(acols, arows);
  vnl_vector<float> c_wb(arows);
  thrust::copy(d_Atmp->begin(), d_Atmp->end(), c_wA.data_block());
  thrust::copy(d_btmp->begin(), d_btmp->end(), c_wb.data_block());
 // vnl_matlab_print(vcl_cout,c_wAtwb,"wAtwb",vnl_matlab_print_format_long); std::cout << std::endl;
  

  vnl_qr<float>* QR = new vnl_qr<float>(c_wA.transpose());
  vnl_vector<float> p = QR->solve(c_wb);
  delete (QR);

  // copy p back to GPU:
  thrust::copy(p.data_block(), p.data_block()+acols, d_p.begin());

}*/


template<>
void RegressionCuda<float>::getWeightedLSEstFloat(const thrust::device_vector<float>& d_w,
         thrust::device_vector<float>& d_p)
{         
  float alpha = 1.0;
  float beta  = 0.0;
  assert(d_w.size() == arows);

//saveMatrix("A.cu.bin", A);
//saveVector("b.cu.bin", b);
//saveVector("w.cu.bin", d_w);

  // compute symmetric wAtwA := (w A)^T (w A) = A^T w^T w A
  // with w a diagonal matrix
  // first Atemp := diag(w) * A
  cublasSafeCall(cublasSdgmm(handle, CUBLAS_SIDE_LEFT,arows,acols,
                          thrust::raw_pointer_cast(&d_A->operator[](0)), arows,
                          thrust::raw_pointer_cast(&d_w[0]), 1,
                          thrust::raw_pointer_cast(&d_Atmp->operator[](0)), arows));
                         
  // now small square symetric matrix d_wAtwA := Atemp^T Atemp
  cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, acols, acols, arows, &alpha,
                          thrust::raw_pointer_cast(&d_Atmp->operator[](0)), arows,
                          thrust::raw_pointer_cast(&d_Atmp->operator[](0)), arows,
                          &beta, thrust::raw_pointer_cast(&d_wAtwA->operator[](0)), acols));
  
  // compute vector wAtwb := (w A)^T (w b)
  // first btemp := diag(w) * b
  cublasSafeCall(cublasSdgmm(handle, CUBLAS_SIDE_LEFT,arows,1,
                          thrust::raw_pointer_cast(&d_b->operator[](0)), arows,
                          thrust::raw_pointer_cast(&d_w[0]), 1,
                          thrust::raw_pointer_cast(&d_btmp->operator[](0)), arows));

  // now compute vector d_wAtwb := Atemp^T btemp
  cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, acols, 1, arows, &alpha,
                          thrust::raw_pointer_cast(&d_Atmp->operator[](0)), arows,
                          thrust::raw_pointer_cast(&d_btmp->operator[](0)), arows,
                          &beta, thrust::raw_pointer_cast(&d_wAtwb->operator[](0)), acols));
  
//saveVector("wAtwb.cu.bin", *d_wAtwb);
//saveVector("wAtwA.cu.bin", *d_wAtwA);
//exit(1);
  
  // now that matrices are small, get them from GPU and decompose on CPU
  // (to avoid inclusion of GPU LAPACK libraries)
  vnl_matrix<float> c_wAtwA(acols, acols);
  vnl_vector<float> c_wAtwb(acols);
  thrust::copy(d_wAtwA->begin(), d_wAtwA->end(), c_wAtwA.data_block());
  thrust::copy(d_wAtwb->begin(), d_wAtwb->end(), c_wAtwb.data_block());
  //vnl_matlab_print(vcl_cout,c_wAtwb,"wAtwb",vnl_matlab_print_format_long); std::cout << std::endl;

//  works only for double, not templated 
//  // compute Cholesky  R^t R = wAtwa
//  vnl_cholesky* CHOL = new vnl_cholesky(c_wAtwA);
//  // solve (see also dposl linpack) wAtWA p  = wAtwb
//  // internally it solves R' y = wAtwb  then solve R p = y 
//  vnl_vector<T> p = CHOL->solve(c_wAtwb);
//  delete (CHOL);

  vnl_qr<float>* QR = new vnl_qr<float>(c_wAtwA);
  vnl_vector<float> p = QR->solve(c_wAtwb);
  delete (QR);

  // copy p back to GPU:
  thrust::copy(p.data_block(), p.data_block()+acols, d_p.begin());

}

template<>
void RegressionCuda<double>::getWeightedLSEstFloat(const thrust::device_vector<double>& d_w,
         thrust::device_vector<double>& d_p)
{         
  double alpha = 1.0;
  double beta  = 0.0;
  assert(d_w.size() == arows);

  // compute symmetric wAtwA := (w A)^T (w A) = A^T w^T w A
  // with w a diagonal matrix
  // first Atemp := diag(w) * A
  cublasSafeCall(cublasDdgmm(handle, CUBLAS_SIDE_LEFT,arows,acols,
                          thrust::raw_pointer_cast(&d_A->operator[](0)), arows,
                          thrust::raw_pointer_cast(&d_w[0]), 1,
                          thrust::raw_pointer_cast(&d_Atmp->operator[](0)), arows));
                          
  // now small square symetric matrix d_wAtwA := Atemp^T Atemp
  cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, acols, acols, arows, &alpha,
                          thrust::raw_pointer_cast(&d_Atmp->operator[](0)), arows,
                          thrust::raw_pointer_cast(&d_Atmp->operator[](0)), arows,
                          &beta, thrust::raw_pointer_cast(&d_wAtwA->operator[](0)), acols));
  
  // compute vector wAtwb := (w A)^T (w b)
  // first btemp := diag(w) * b
  cublasSafeCall(cublasDdgmm(handle, CUBLAS_SIDE_LEFT,arows,1,
                          thrust::raw_pointer_cast(&d_b->operator[](0)), arows,
                          thrust::raw_pointer_cast(&d_w[0]), 1,
                          thrust::raw_pointer_cast(&d_btmp->operator[](0)), arows));
                          
  // now compute vector d_wAtwb := Atemp^T btemp
  cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, acols, 1, arows, &alpha,
                          thrust::raw_pointer_cast(&d_Atmp->operator[](0)), arows,
                          thrust::raw_pointer_cast(&d_btmp->operator[](0)), arows,
                          &beta, thrust::raw_pointer_cast(&d_wAtwb->operator[](0)), acols));
  
  // now that matrices are small, get them from GPU and decompose on CPU
  // (to avoid inclusion of GPU LAPACK libraries)
  vnl_matrix<double> c_wAtwA(acols, acols);
  vnl_vector<double> c_wAtwb(acols);
  thrust::copy(d_wAtwA->begin(), d_wAtwA->end(), c_wAtwA.data_block());
  thrust::copy(d_wAtwb->begin(), d_wAtwb->end(), c_wAtwb.data_block());

//  works only for double, not templated 
  // compute Cholesky  R^t R = wAtwa
  vnl_cholesky* CHOL = new vnl_cholesky(c_wAtwA);
  // solve (see also dposl linpack) wAtWA p  = wAtwb
  // internally it solves R' y = wAtwb  then solve R p = y 
  vnl_vector<double> p = CHOL->solve(c_wAtwb);
  delete (CHOL);

//  vnl_qr<double>* QR = new vnl_qr<double>(c_wAtwA);
//  vnl_vector<double> p = QR->solve(c_wAtwb);
//  delete (QR);


  // copy p back to GPU:
  thrust::copy(p.data_block(), p.data_block()+acols, d_p.begin());

}

/** Computes sqrt(weights) for a reweighted least squares approach.
 Returns elements of diag matrix W (as a column vector).
 Input: r (residuals) and sat
 Output: w sqrt of weights
 */
template<class T>
void RegressionCuda<T>::getSqrtTukeyDiaWeights(const thrust::device_vector<T>& d_r,
    thrust::device_vector<T>& d_w, double sigma, double sat)
{
  //cout << " getTukeyDiaWeights  r size: " << r->rows << " , " << r->cols << endl;
  assert(d_r.size() == d_w.size());

  // compute d_w[i] = weights (0 if r >= sigmasat or 1- (r/sigmasat)^2 else)
  thrust::transform(d_r.begin(), d_r.end(), d_w.begin(), getweights<T>(sigma*sat));
  
}


template class RegressionCuda<float>;
template class RegressionCuda<double>;

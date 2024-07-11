#include <complex>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include "cu_pdd.h"

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
  static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
      return __longlong_as_double(old);
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }
  #endif



__device__ cuDoubleComplex global_v[256] = {0,0.0};



__global__ void w_update(
    double * w_u, 
    double * w_d, 
    double * e_u, 
    double * e_d, 
    double * A,
    int use) {
    int idx = threadIdx.x;
    if (idx < use) {
        w_u[ idx ] = 1.0 / (log(2.0) * e_u[ idx ]);
        A[idx] = 0.0;
    }
    else if (idx < 2 * use) {
        w_d[ idx - use ] = 1.0 / (log(2.0)  * e_d[ idx - use ]);
    }
}






__global__ void p_update(
    double* p_u, 
    double* w_u,
    double* w_d,
    cuDoubleComplex* v_u,
    cuDoubleComplex* v_d,
    cuDoubleComplex* H_u,
    cuDoubleComplex* H1, 
    cuDoubleComplex* H3,
    double* A,  //[a1,a1 .. ]
    double I_th,
    int M,
    int use)
    {
    //<<<(4,4);32>>>
    int k = blockIdx.x;
    int s = blockIdx.y;
    int n = threadIdx.x;



    __shared__ cuDoubleComplex H_H[32];
    H_H[n] = cuCmul(cuConj(v_u[s * M + n]) , H_u[k * M + n]);
    
    __syncthreads();

    
    for (size_t i = M/2; i > 0; i >>= 1)
    {
        if(n < i){
            H_H[n] = cuCadd(H_H[n] , H_H[n+i]);
        }
        /* code */
        __syncthreads();
    }
    __syncthreads();

    if(n == 0){
        H_H[0] = cuCmul(make_cuDoubleComplex(w_u[s], 0.0), cuCmul(H_H[0], cuConj(H_H[0])));
        
        cuDoubleComplex hh = cuCmul(v_d[s], H1[k*use+s]);
        hh = cuCmul(make_cuDoubleComplex(w_d[s], 0.0), cuCmul(hh, cuConj(hh)));
        hh = cuCadd(hh, H_H[0]);

        atomicAdd(&A[k], hh.x);
    }
    __syncthreads();

    if(s == 0){
        __shared__ double B[32];

        cuDoubleComplex b=cuCmul(cuConj(v_u[k*M+n]), H_u[k*M+n]);
        B[n] = b.x;
        __syncthreads();

        for (size_t i = M/2; i > 0; i >>= 1)
        {
            if(n < i){
                B[n] = B[n] + B[n + i];
            }
            __syncthreads();
            /* code */
        }
        __syncthreads();
        if(n == 0){
            double bb = B[0]*w_u[k];
            double a = A[k];

            if(bb >= 0){
                double p_use = 1.0;
                double b_a = pow(bb / a,2);
                double I_h3 = I_th / cuCreal(cuCmul(H3[k], cuConj(H3[k])));

                p_u[k] = min(min(p_use, b_a) , I_h3);
            }
            else
                p_u[k] = 0;
        }


    }



}

// __global__ void p_update(
//     double* p_u, 
//     double* w_u,
//     double* w_d,
//     cuDoubleComplex* v_u,
//     cuDoubleComplex* v_d,
//     cuDoubleComplex* H_u,
//     cuDoubleComplex* H1, 
//     cuDoubleComplex* H3,
//     double I_th,
//     int M,
//     int use)
//     {
//     //<<<(4,4);32>>>
//     int k = blockIdx.x;
//     int s = blockIdx.y;
//     int n = threadIdx.x;

//     global_v[n] = {0.0,0.0};


//     __shared__ cuDoubleComplex H_H[32];
//     H_H[n] = cuCmul(cuConj(v_u[s * M + n]) , H_u[k * M + n]);
    

//     __syncthreads();

    
//     for (size_t i = M/2; i > 0; i >>= 1)
//     {
//         if(n < i){
//             H_H[n] = cuCadd(H_H[n] , H_H[n+i]);
//         }
//         /* code */
//         __syncthreads();
//     }
//     __syncthreads();

//     if(n == 0){
//         H_H[0] = cuCmul(make_cuDoubleComplex(w_u[s], 0.0), cuCmul(H_H[0], cuConj(H_H[0])));
        
//         cuDoubleComplex hh = cuCmul(v_d[s], H1[k*use+s]);
//         hh = cuCmul(make_cuDoubleComplex(w_d[s], 0.0), cuCmul(hh, cuConj(hh)));
//         hh = cuCadd(hh, H_H[0]);
//         global_v[k*use+s] = hh;
//     }
//     __syncthreads();
//     __shared__ double B[32];

//     cuDoubleComplex b;
//     b = cuCmul(cuConj(v_u[k*M+n]), H_u[k*M+n]);
//     B[n] = b.x;
//     __syncthreads();

//     for (size_t i = M/2; i > 0; i >>= 1)
//     {
//         if(n < i){
//             B[n] = B[n] + B[n + i];
//         }
//         __syncthreads();
//         /* code */
//     }
//     __syncthreads();


//     double bb = B[0]*w_u[k];


//     if(s == 0 && n == 0){

//         cuDoubleComplex a = {0.0,0.0};
//         for(int i=0;i<use;i++){
//             a = cuCadd(a, global_v[k*use+i]);
//         }

//         cuDoubleComplex b = make_cuDoubleComplex(bb, 0.0);

//         if(b.x >= 0){
//             double p_use = 1.0;
//             double b_a = pow(cuCabs(cuCdiv(b , a)), 2);
//             double I_h3 = I_th / cuCreal(cuCmul(H3[k], cuConj(H3[k])));

//             p_u[k] = min(min(p_use, b_a) , I_h3);
//         }
//         else
//             p_u[k] = 0;

//     }
//     __syncthreads();

// }


//F_update
//按照列存储运算，输出参数为列存储
__global__ void XY_update(
    cuDoubleComplex * X,
    cuDoubleComplex * H,
    cuDoubleComplex * Y, 
    cuDoubleComplex * v_u, 
    cuDoubleComplex * v_d, 
    double * w_u,
    double * w_d,
    cuDoubleComplex * H_d,
    int use){
    
    int k = blockIdx.x;
    int m = threadIdx.x;
    int n = threadIdx.y;
    int M = 32;

    if(k == 0){
        __shared__ cuDoubleComplex w[1024];
        w[n*M+m] = {0.0,0.0};
        cuDoubleComplex ww;
        for(int i=0;i<use;i++){
            ww = cuCmul(v_u[i*M+m],cuConj(v_u[i*M+n]));
            ww = cuCmul(ww, make_cuDoubleComplex(w_u[i],0.0));
            w[n*M+m] = cuCadd(w[n*M+m], ww);
            __syncthreads();
        }
        X[n*M+m] = w[n*M+m];
        if (m == n) {
            X[n*M+m].y = 0.0;
        }
    }
    else if (k == 1) {
        __shared__ cuDoubleComplex h[1024];
        h[n*M+m] = {0.0,0.0};
        cuDoubleComplex hh;
        for(int i=0;i<use;i++){
            hh = cuCmul(cuConj(v_d[i]), v_d[i]);
            hh = cuCmul(cuConj(H_d[m*use+i]), hh);
            hh = cuCmul(hh, H_d[n*use+i]);
            h[n*M+m] = cuCadd(h[n*M+m], cuCmul(make_cuDoubleComplex(w_d[i], 0.0), hh));
            __syncthreads();
        }
        H[n*M+m] = h[n*M+m];
    }
    else {
        if (m < 32 && n < use) {
            cuDoubleComplex yy;
            yy = cuCmul(cuConj(H_d[m*use+n]), cuConj(v_d[n]));
            Y[n*M+m] = cuCmul(make_cuDoubleComplex(w_d[n],0.0), yy);
        }
    }
      
}

__global__ void mat_add_I(cuDoubleComplex * x, double p){
    int m = threadIdx.x;
    int n = threadIdx.y;


    int M = 32;
    if(m == n){
        x[m*M+n] = cuCadd(x[m*M+n], make_cuDoubleComplex(p, 0.0));
    }
}



__global__ void fun6(cuDoubleComplex*A,int M){
    int n = threadIdx.x;
    A[n*M+n] = make_cuDoubleComplex(1.0, 0.0);
}



void inv(cuDoubleComplex * F, cuDoubleComplex * F_inv, int M,cusolverDnHandle_t cusolverH){

    int lwork;
    cuDoubleComplex * work;
    int *devIpiv = nullptr;
    int *devInfo = nullptr;
    cusolverDnZgetrf_bufferSize(cusolverH,M,M,F,M,&lwork);
    
    cudaMalloc((void**)&work,lwork*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&devIpiv, M * sizeof(int));
    cusolverDnZgetrf(cusolverH, M, M, F, M, work, NULL, devInfo);

    fun6<<<1,M>>>(F_inv,M);

    cusolverDnZgetrs(cusolverH, CUBLAS_OP_N, M, M, F, M, NULL, F_inv, M, devInfo);
    
    //cusolverDnDestroy(cusolverH);
    cudaFree(work);
    cudaFree(devIpiv);
    
}






void F_update(
    cuDoubleComplex * F, 
    cuDoubleComplex * F_RF,
    cuDoubleComplex * F_BB,
    cuDoubleComplex * v_u, 
    cuDoubleComplex * v_d, 
    double * w_u,
    double * w_d,
    cuDoubleComplex * H_d,
    cuDoubleComplex * H_SI,
    cuDoubleComplex * H1,
    cuDoubleComplex * lambda,
    double p,
    int M,
    int use,
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH
    ){


    //cublasCreate(&cublasH);

    cuDoubleComplex * X, * Y, * H, *w;
    cudaMalloc((void**)&X, M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&Y, M*use*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&H, M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&w, M*M*sizeof(cuDoubleComplex));


    dim3 block(32,32);
    XY_update<<<3, block>>>(w,H,Y,v_u,v_d,w_u,w_d,H_d,use);
   
    //cuDoubleComplex test[1024];


    //X Y TODO:可以使用stream

    cuDoubleComplex alpha = {1.0,0.0};
    cuDoubleComplex beta1 ={0.0,0.0};
    cuDoubleComplex beta2 ={1.0,0.0};

    cublasZgemm(
        cublasH,CUBLAS_OP_C,CUBLAS_OP_N,
        M,M,M,&alpha,
        H_SI, M,
        w, M,
        &beta1,
        X, M
    );

    
    cublasZgemm(
        cublasH, CUBLAS_OP_N,CUBLAS_OP_N,
        M,M,M, &alpha,
        X,M,
        H_SI,M,
        &beta2,
        H,M
    );

    //alphe = 0.08
    double temp = 0.5/p+0.08;
    mat_add_I<<<1,block>>>(H,temp);


    cuDoubleComplex * F_inv;
    cudaMalloc((void**)&F_inv, M*M*sizeof(cuDoubleComplex));

    // cudaMemcpy(test,H,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<32*32;i++){
    //     printf("H :%d: %f %f\n",i,test[i].x,test[i].y);
    // }
   
    inv(H, F_inv, M,cusolverH);

    // cudaMemcpy(test,F_inv,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<32*32;i++){
    //     printf("inv :%d: %f %f\n",i,test[i].x,test[i].y);
    // }

    double l = 0.5 / p;
    cuDoubleComplex alpha2 = {l,0.0};
    cuDoubleComplex beta3 = {-0.5,0.0};

    int N = 8;

    cublasZgemm(
        cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
        M, use, N, &alpha2,
        F_RF, M,
        F_BB, N,
        &beta2,
        Y,M
    );

    cublasZgeam(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        M,use,
        &alpha,
        Y,M,
        &beta3,
        lambda,M,
        Y,M
    );
    
    // cudaMemcpy(test,Y,M*use*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<32;i++){
    //     printf("Y :%d: %f %f\n",i,test[i].x,test[i].y);
    // }


    cublasZgemm(
        cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
        M,use,M,&alpha,
        F_inv,M,
        Y,M,
        &beta1,
        F,M
    );


    //cuDoubleComplex test[1024];
    // cudaMemcpy(test,F,M*use*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<4;i++){
    //     printf("F :%d: %f %f\n",i,test[i].x,test[i].y);
    // }


}


//F_BB_update

//V' * (S^-1)'
__global__ void v_mul_s(cuDoubleComplex * V ,double * S,cuDoubleComplex * VS,int M,int N){
    int m = threadIdx.x;
    int n = threadIdx.y;
    if(m < N&& n < N){
        VS[n * N + m] = cuCmul(cuConj(V[m*N+n]),make_cuDoubleComplex(1.0/S[n],0.0));
    }else{
        VS[n * N + m] = make_cuDoubleComplex(0.0,0.0);
    }
}

void F_BB_update(
    cuDoubleComplex * F_RF,
    cuDoubleComplex * F,
    cuDoubleComplex * F_BB,
    cuDoubleComplex * lambda,
    double p,
    int M,
    int N,
    int use,
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH){


    double *S;
    cuDoubleComplex * U, * V;
    cudaMalloc((void**)&S,M*sizeof(double));
    cudaMalloc((void**)&U,M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&V,N*N*sizeof(cuDoubleComplex));
    
    
    int lwork;
    cuDoubleComplex * work;
    double * rwork = nullptr;
    int *devInfo = nullptr;

    cusolverDnZgesvd_bufferSize(cusolverH,M,N,&lwork);
    cudaMalloc((void**)&work,lwork*sizeof(cuDoubleComplex));
    
    //SVD
    signed char jobu = 'A';
    signed char jobvt = 'A';

    cuDoubleComplex * F_RF_;
    cudaMalloc((void**)&F_RF_,M*N*sizeof(cuDoubleComplex));
    cudaMemcpy(F_RF_,F_RF,M*N*sizeof(cuDoubleComplex),cudaMemcpyDeviceToDevice);

    cusolverDnZgesvd(
        cusolverH,jobu,jobvt,
        M,N,F_RF_,M,
        S,
        U,M,
        V,N,
        work,lwork,rwork,
        devInfo
    );
    
    cuDoubleComplex * VS;
    cudaMalloc((void**)&VS, N*M*sizeof(cuDoubleComplex));
    v_mul_s<<<1,dim3(N,M)>>>(V,S,VS,M,N);

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0,0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0,0.0);
    cuDoubleComplex beta_p = {p,0.0};

    cuDoubleComplex * F_RF_pinv;
    cudaMalloc((void**)&F_RF_pinv, N*M*sizeof(cuDoubleComplex));

    
    cublasZgemm(
        cublasH,CUBLAS_OP_N,CUBLAS_OP_C,
        N,M,M,
        &alpha,
        VS,N,
        U,M,
        &beta, 
        F_RF_pinv,N
    );
    
    cuDoubleComplex * C;
    cudaMalloc((void**)&C,M*use*sizeof(cuDoubleComplex));
    cublasZgeam(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        M,use,
        &alpha,
        F,M,
        &beta_p,
        lambda,M,
        C,M
    );

    cublasZgemm(
        cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
        N,use,M,
        &alpha,
        F_RF_pinv,N,
        C,M,
        &beta,
        F_BB,N
    );

    // cuDoubleComplex test[1024];
    // cudaMemcpy(test,F_BB,N*use*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<4;i++){
    //     printf("F_BB :%d: %f %f\n",i,test[i].x,test[i].y);
    // }
}

//F_RF_update

__global__ void RF_update(
    cuDoubleComplex * F_RF, 
    cuDoubleComplex * A ,
    cuDoubleComplex * B,
    int M,
    int N){
    //<<<M,M>>>
    int k = blockIdx.x;
    int n = threadIdx.x;

    __shared__ cuDoubleComplex V[32];
    __shared__ cuDoubleComplex x[32];
    if(n < N){
        V[n] = F_RF[n*M+k]; 
    }

    for(int s=0;s<N;s++){
        
        x[n] = make_cuDoubleComplex(0.0, 0.0);
        __syncthreads();
        if(n < N){
            x[n] = cuCmul(make_cuDoubleComplex(-1.0*V[n].x,-1.0*V[n].y),A[s*N+n] );
        }
        else if(n == N){
            x[n] = B[s*M+k];
        }
        else if(n == N+1){
            x[n] = cuCmul(V[s],A[s*N+s]);
        }
        else{
            x[n] = make_cuDoubleComplex(0.0,0.0);
        }
        __syncthreads();

        for(int i = M/2; i > 0 ; i >>= 1){
            if(n < i)
                x[n] = cuCadd(x[n], x[n+i]);
            __syncthreads();
        }
        __syncthreads();
        if(n==0){
            V[s] = cuCdiv(x[0],make_cuDoubleComplex(cuCabs(x[0]),0.0));
        
        }
        __syncthreads();


    }
    if(n < N){
        F_RF[n*M+k] = V[n] ; 
    }

}


void F_RF_update(
    cuDoubleComplex* F_BB,
    cuDoubleComplex* F,
    cuDoubleComplex* F_RF,
    cuDoubleComplex* lambda,
    double p,
    int M,
    int N,
    int use,
    cublasHandle_t cublasH){

    cuDoubleComplex * A, *B, *C;
    cudaMalloc((void**)&A, N*N*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&B, M*N*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&C, M*use*sizeof(cuDoubleComplex));

    cuDoubleComplex alpha = {1.0,0.0};
    cuDoubleComplex beta = {0.0,0.0};
    cuDoubleComplex beta_p = {p,0.0};
    cublasZgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_C,
        N,N,use,
        &alpha,
        F_BB,N,
        F_BB,N,
        &beta,
        A,N
    );
    
    cublasZgeam(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        M,use,
        &alpha,
        F,M,
        &beta_p,
        lambda,M,
        C,M
    );

    cublasZgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_C,
        M,N,use,
        &alpha,
        C,M,
        F_BB,N,
        &beta,
        B,M
    );
    //cuDoubleComplex test[1024];


    RF_update<<<M,M>>>(F_RF, A, B, M, N);

    // cudaMemcpy(test,F_RF,M*N*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<4;i++){
    //     printf("F_RF :%d: %f %f\n",i,test[i].x,test[i].y);
    // }

}

//V_u V_d update




__global__ void A_update(
    cuDoubleComplex * A,
    double N,
    cuDoubleComplex * F,
    double * p_u,
    cuDoubleComplex * H_u,
    cuDoubleComplex * H_SI,
    cuDoubleComplex * I_W2B,
    int use){

    int m = threadIdx.x;
    int n = threadIdx.y;
    //<<<1,(M,M)>>>
    int M = 32;

    __shared__ cuDoubleComplex A_s[1024];
    __shared__ cuDoubleComplex H_v[32];

    A_s[n*M+m] = {0.0,0.0};

    for(int k=0;k<use;k++){
        cuDoubleComplex a;
        a = cuCmul(H_u[k*M+m], cuConj(H_u[k*M+n]));
        a = cuCmul(make_cuDoubleComplex(p_u[k], 0.0), a);

        // if( n==0 ){
        //     printf("a :%d %d %e %e\n",k ,m,a.x,a.y);
        // }
        __syncthreads();
        if(m == 0){
            cuDoubleComplex aa = {0.0,0.0};
            for(int i=0; i<M; i++){
                aa = cuCadd(aa, cuCmul(H_SI[i*M+n], F[k*M+i]));
            }
            H_v[n] = aa;

            // printf("a :%d %d %e %e\n",k ,n,H_v[n].x,H_v[n].y);
        }
        __syncthreads();

        // if(n==0){
        //     printf("A_s_pre %d :%e %e\n",k, A_s[n*M+m].x,A_s[n*M+m].y);
        // }
        a = cuCadd(a, A_s[n*M+m]);
        A_s[n*M+m] = cuCadd(a, cuCmul(H_v[m],cuConj(H_v[n])));
        // if(n==0){
        //     printf("A_s %d :%e %e\n",k, A_s[n*M+m].x,A_s[n*M+m].y);
        // }
        __syncthreads();

    }
    __syncthreads();


    A_s[n*M+m] = cuCadd(A_s[n*M+m], I_W2B[n*M+m]);
    __syncthreads();
    if(m == n){
        A_s[n*M+m] = cuCadd(A_s[n*M+m], make_cuDoubleComplex(N, 0.0));
    }
    __syncthreads();
    A[n*M+m] = A_s[n*M+m];

}

// __global__ void A_update(
//     cuDoubleComplex * A,
//     double N,
//     cuDoubleComplex * F,
//     double * p_u,
//     cuDoubleComplex * H_u,
//     cuDoubleComplex * H_SI,
//     cuDoubleComplex * I_W2B,
//     int use){

//     int k = blockIdx.x;
//     int m = threadIdx.x;
//     int n = threadIdx.y;

//     int M = 32;

//     __shared__ cuDoubleComplex A_s[1024];
//     if (k < use) {
//         cuDoubleComplex a;
//         a = cuCmul(H_u[k*M+m], cuConj(H_u[k*M+n]));
//         A_s[n*M+m] = cuCmul(make_cuDoubleComplex(p_u[k], 0.0), a);
//     }
//     else if (k <= use*2) {
//         __shared__ cuDoubleComplex H_v[32];
//         int k_u = k - use;
//         if(m == 0 && n < M){
//             cuDoubleComplex a = {0.0,0.0};
//             for(int i=0; i<M; i++){
//                 a = cuCadd(a, cuCmul(H_SI[i*M+n], F[k_u*M+i]));
//             }
//             H_v[n] = a;
//         }
//         __syncthreads();

//         A_s[n*M+m] = cuCmul(H_v[m],cuConj(H_v[n]));
//     }

//     atomicAdd(&A[n*M+m].x, A_s[n*M+m].x);
//     atomicAdd(&A[n*M+m].y, A_s[n*M+m].y);

//     if(k == 0){
//         A[n*M+m] = cuCadd(A[n*M+m], I_W2B[n*M+m]);
//         if(m == n){
//             A[n*M+m] = cuCadd(A[n*M+m], make_cuDoubleComplex(N, 0.0));
//         }
//     }
// }



__global__ void a_update(
    cuDoubleComplex * a,
    double N,
    cuDoubleComplex * H_V,
    double * p_u,
    cuDoubleComplex * H1,
    double * I_W2U,
    int use){
    
    int m = threadIdx.x;

    double a_f = 0;
    for(int i=0;i<use;i++){
        a_f += p_u[i] * pow(cuCabs(H1[i*use+m]),2);
    }
    
    cuDoubleComplex aa;
    aa = cuCadd(make_cuDoubleComplex(N, 0.0), H_V[m*use+m]);
    a_f = a_f + I_W2U[m];
    a[m] = cuCadd(aa, make_cuDoubleComplex(a_f, 0.0));

}

__global__ void B_update(
    cuDoubleComplex * B,
    cuDoubleComplex * H_u,
    double * p_u){

    int m = threadIdx.x;
    int n = threadIdx.y;
    int M = 32;

    B[n*M+m] = cuCmul(H_u[n*M+m],make_cuDoubleComplex(sqrt(p_u[n]),0.0));


}

__global__ void VE_update(
    cuDoubleComplex * v_u,
    cuDoubleComplex * v_d,
    cuDoubleComplex * A,
    cuDoubleComplex * A_inv,
    cuDoubleComplex * B,
    cuDoubleComplex * a,
    cuDoubleComplex * b,
    double * e_u,
    double * e_d,
    int use){
    int k = blockIdx.x;
    int m = threadIdx.x;
    int n = threadIdx.y;
    int M = 32;
    __shared__ cuDoubleComplex v_u_l[32];
    if(n == 0)
        v_u_l[m] = {0.0,0.0};
    __syncthreads();
    if(k < use){
        if(m == 0){
            cuDoubleComplex vv = {0.0,0.0};
            for(int i=0;i<M;i++){
                vv = cuCadd(vv, cuCmul(A_inv[i*M+n], B[k*M+i]));
            }
            v_u_l[n] = vv;
            v_u[k*M+n] = vv;
        }
        __syncthreads();

        __shared__ double E_u[1024];
        E_u[n*M+m] = 0.0;
        E_u[n*M+m] = cuCmul(cuConj(v_u_l[m]), cuCmul(A[n*M+m], v_u_l[n])).x;

        __syncthreads();

        if(m == 0){
            E_u[n*M+m] = E_u[n*M+m] - 2 * cuCmul(cuConj(v_u_l[n]), B[k*M+n]).x;
        }
        __syncthreads();

        int tid = m*M+n;
        for(int i=512;i>0;i>>=1){
            if(tid < i){
                E_u[tid] = E_u[tid] + E_u[tid+i];
            }
            __syncthreads();
        }

        __syncthreads();
        if(n==0&&m==0)
            e_u[k] = E_u[0] + 1.0;
    }
    else if(k < use*2){
        if(m == 0 && n == 0){
            int kk = k - use;
            cuDoubleComplex v_dd;
            v_dd = cuCdiv(b[kk*use+kk], a[kk]);
            v_d[kk] = v_dd;
            e_d[kk] = a[kk].x * pow(cuCabs(v_dd),2) - 2 * cuCmul(v_dd,b[kk*use+kk]).x + 1;
        }
    }

}

void V_update(
    double N,
    cuDoubleComplex * F,
    double * p_u,
    cuDoubleComplex * v_u,
    cuDoubleComplex * v_d,
    double * e_u,
    double * e_d,
    cuDoubleComplex * H_u,
    cuDoubleComplex * H_SI,
    cuDoubleComplex * H_d,
    cuDoubleComplex * H1,
    cuDoubleComplex * I_W2B,
    double * I_W2U,
    int use,
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH){
    
    int M =32;

    //test
    //cuDoubleComplex test[1024];
    //double test2[1024];

    cuDoubleComplex *A,*B, *a, *b;
    cudaMalloc((void**)&A, M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&B, M*use*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&a, use*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&b, use*sizeof(cuDoubleComplex));

    A_update<<<1, dim3(M,M)>>>(A, N, F, p_u, H_u, H_SI, I_W2B, use);


    // cudaMemcpy(test,A,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<4;i++){
    //     printf("A:%d :%e %e\n",i,test[i].x,test[i].y);
    // }


    cuDoubleComplex *H_F,*H_F_t;
    cudaMalloc((void**)&H_F, use*use*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&H_F_t, use*use*sizeof(cuDoubleComplex));

    cuDoubleComplex alpha = {1.0,0.0};
    cuDoubleComplex beta = {0.0,0.0};

    cublasZgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        use,use,M,
        &alpha,
        H_d,use,
        F,M,
        &beta,
        H_F,use
    );
    cublasZgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_C,
        use,use,use,
        &alpha,
        H_F,use,
        H_F,use,
        &beta,
        H_F_t,use
    );

    a_update<<<1,use>>>(a,N,H_F_t,p_u,H1,I_W2U,use);

    B_update<<<1,dim3(M,use)>>>(B,H_u,p_u);

    


    cublasZgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        use,use,M,
        &alpha,
        H_d,use,
        F,M,
        &beta,
        b,use
    );



    cuDoubleComplex * A_inv,*AA;
    cudaMalloc((void**)&A_inv, M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&AA, M*M*sizeof(cuDoubleComplex));
    cudaMemcpy(AA,A,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToDevice);

    inv(A, A_inv, M,cusolverH);
    // cudaMemcpy(test,AA,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<32;i++){
    //     printf("A :%d: %f %f\n",i,test[i].x,test[i].y);
    // }

    // cudaMemcpy(test,A_inv,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<32;i++){
    //     printf("Ainv :%d: %f %f\n",i,test[i].x,test[i].y);
    // }
    
    // cuDoubleComplex *e;
    // cudaMalloc((void**)&e, M*M*sizeof(cuDoubleComplex));
    // cublasZgemm(
    //     cublasH,
    //     CUBLAS_OP_N,CUBLAS_OP_N,
    //     M,M,M,
    //     &alpha,
    //     AA,M,
    //     A_inv,M,
    //     &beta,
    //     e,M
    // );
    // cudaMemcpy(test,e,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<32;i++){
    //     printf("E :%d: %f %f\n",i,test[i].x,test[i].y);
    // }
    



    VE_update<<<use*2, dim3(M,M)>>>(v_u, v_d, AA, A_inv, B, a, b, e_u, e_d, use);

    // cuDoubleComplex ff[1024], vd[4];
    // double eu[4],ed[4];
    // cudaMemcpy(ff,v_u,M*use*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // cudaMemcpy(vd,v_d,use*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // cudaMemcpy(eu,e_u,use*sizeof(double),cudaMemcpyDeviceToHost);
    // cudaMemcpy(ed,e_d,use*sizeof(double),cudaMemcpyDeviceToHost);

    // for(int i=0;i<4;i++){
    //     printf("v_u :%d: %f %f\n",i,ff[i].x,ff[i].y);
    // }

    // for(int i=0;i<4;i++){
    //     printf("e_u :%d: %f\n",i,eu[i]);
    // }

    cudaFree(A);

}


//计算f

__global__ void f_cal( 
    double * f,   
    double * w_u,
    double * w_d,
    double * e_u,
    double * e_d,
    double F_norm,
    double p,
    int use,
    int use_norm){
    
    int m = threadIdx.x;
    int n = threadIdx.y;

    extern __shared__ double sh[];

    if(m == 0){
        sh[m*use+n] = w_u[n] * e_u[n];
    }
    else if (m == 1) {
        sh[m*use+n] = -1.0 * __log2f(w_u[n]);
    }
    else if (m == 2) {
        sh[m*use+n] = w_d[n] * e_d[n];
    }
    else if (m == 3) {
        sh[m*use+n] = -1.0 * __log2f(w_d[n]);
    }
    else
        sh[m*use+n] = 0.0;
    __syncthreads();

    int tid = m*use+n;
    for (int i = 2*use_norm; i > 0; i >>= 1)
    {
        if(tid < i){
            sh[tid] = sh[tid] + sh[tid+i];
        }
        __syncthreads();
        /* code */
    }
    __syncthreads();
    if(tid == 0)
        f[0] = sh[0] + (0.5 / p) * F_norm * F_norm; 
    
}

double F_cal(
    double * w_u,
    double * w_d,
    double * e_u,
    double * e_d,
    cuDoubleComplex * F,
    cuDoubleComplex * F_RF,
    cuDoubleComplex * F_BB,
    cuDoubleComplex * lambda,
    int use,
    double p,
    int RF,
    cublasHandle_t cublasH,
    int use_norm){
    
    int M = 32;

    cuDoubleComplex * F_gap;
    double F_norm;
    cudaMalloc((void**)&F_gap, M*use*sizeof(cuDoubleComplex));
    cudaMemcpy(F_gap, F, M*use*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    cuDoubleComplex alpha = {-1.0,0.0};
    cuDoubleComplex beta = {1.0,0.0};
    cuDoubleComplex beta_p = {p,0.0};
    
    cublasZgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        M,use,RF,
        &alpha,
        F_RF, M,
        F_BB, RF,
        &beta,
        F_gap,M
    );

    cublasZgeam(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        M,use,
        &beta,
        F_gap,M,
        &beta_p,
        lambda,M,
        F_gap,M
    );

    cublasDznrm2(
        cublasH, M*use,
        F_gap, 1, &F_norm
    );
    double * f;
    cudaMalloc((void **)&f,sizeof(double));
    f_cal<<<1,dim3(use_norm,use),use_norm*use>>>(f, w_u, w_d, e_u, e_d, F_norm, p, use,use_norm);

    double f_h;
    cudaMemcpy(&f_h, f, sizeof(double), cudaMemcpyDeviceToHost);
    return f_h;

}



//cv_cal

__global__ void norm_Inf(cuDoubleComplex * A,int M, int N,double * res ){
    int m = threadIdx.x; 
    int n = threadIdx.y;

    int len = M*N;
    int idx = m*16+n;
    __shared__ double A_s[512];
    if(idx < len){
        A_s[idx] = cuCabs(A[idx]);
    }
    else{
        A_s[idx] = 0.0f;
    }
    __syncthreads();

    for (int i = 256; i > 0; i >>= 1)
    {
        if(idx < i)
            A_s[idx] = max(A_s[idx],A_s[idx+i]);
        /* code */
        __syncthreads();
    }
    __syncthreads();

    res[0] = A_s[0];
        
}


double cv_cal(
    cuDoubleComplex * F,
    cuDoubleComplex * F_RF,
    cuDoubleComplex * F_BB,
    int M,
    int RF,
    int use,
    cublasHandle_t cublasH,
    int use_norm){
    
    cuDoubleComplex * A;
    cudaMalloc((void**)&A,M*use*sizeof(cuDoubleComplex));
    cudaMemcpy(A,F,M*use*sizeof(cuDoubleComplex),cudaMemcpyHostToHost);

    cuDoubleComplex alpha = {-1.0,0.0};
    cuDoubleComplex beta = {1.0,0.0};

    cublasZgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        M,use,RF,
        &alpha,
        F_RF, M,
        F_BB, RF,
        &beta,
        A,M
    );

    double * inf_norm;
    cudaMalloc((void**)&inf_norm,sizeof(double));

    norm_Inf<<<1,dim3(M,16)>>>(A, M, use, inf_norm);

    double cv;
    cudaMemcpy(&cv,inf_norm,sizeof(double),cudaMemcpyDeviceToHost);
    return cv;

}


void lambda_update(
    cuDoubleComplex * F,
    cuDoubleComplex * F_RF,
    cuDoubleComplex * F_BB,
    cuDoubleComplex * lambda,
    double p,
    int M,
    int RF,
    int use,
    cublasHandle_t cublasH){

    cuDoubleComplex * A;
    cudaMalloc((void**)&A,M*use*sizeof(cuDoubleComplex));
    cudaMemcpy(A,F,M*use*sizeof(cuDoubleComplex),cudaMemcpyHostToHost);

    cuDoubleComplex alpha = {-1.0,0.0};
    cuDoubleComplex beta = {1.0,0.0};
    cuDoubleComplex beta_p = {1/p,0.0};
    cublasZgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        M,use,RF,
        &alpha,
        F_RF, M,
        F_BB, RF,
        &beta,
        A,M
    );
    cublasZgeam(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        M,use,
        &beta,
        lambda,M,
        &beta_p,
        A,M,
        lambda,M
    );

    // cuDoubleComplex test[1024];
    // cudaMemcpy(test,lambda,M*use*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<32;i++){
    //     printf("lambda :%d: %e %e\n",i,test[i].x,test[i].y);
    // }
}

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


__device__ cuDoubleComplex global_v[32] = {0,0.0};



__global__ void w_update(
    double * w_u, 
    double * w_d, 
    double * e_u, 
    double * e_d, 
    int use) {
    int idx = threadIdx.x;
    if (idx < use) {
        w_u[ idx ] = 1.0 / (log(2.0) * e_u[ idx ]);
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
    double I_th,
    int M,
    int use)
    {
    //<<<(4,4);32>>>
    int k = blockIdx.x;
    int s = blockIdx.y;
    int n = threadIdx.x;

    global_v[n] = {0.0,0.0};


    __shared__ cuDoubleComplex H_H[32];
    H_H[n] = cuCmul(cuConj(v_u[s * M + n]) , H_u[k * M + n]);
    //printf("%d %d %d: %f %f\n",k,s,n,v_u[s * M + n].x,v_u[s * M + n].y);
    

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
        global_v[k*use+s] = hh;
    }
    __syncthreads();
    __shared__ double B[32];

    cuDoubleComplex b;
    b = cuCmul(cuConj(v_u[k*M+n]), H_u[k*M+n]);
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


    double bb = B[0]*w_u[k];


    if(s == 0 && n == 0){

        cuDoubleComplex a = {0.0,0.0};
        for(int i=0;i<use;i++){
            a = cuCadd(a, global_v[k*use+i]);
        }

        cuDoubleComplex b = make_cuDoubleComplex(bb, 0.0);

        if(b.x >= 0){
            double p_use = 1.0;
            double b_a = pow(cuCabs(cuCdiv(b , a)), 2);
            double I_h3 = I_th / cuCreal(cuCmul(H3[k], cuConj(H3[k])));

            p_u[k] = min(min(p_use, b_a) , I_h3);
        }
        else
            p_u[k] = 0;

    }
    __syncthreads();

}


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

__global__ void mat_add_I(cuDoubleComplex * x, float p){
    int m = threadIdx.x;
    int n = threadIdx.y;


    int M = 32;
    if(m == n){
        x[m*M+n] = cuCadd(x[m*M+n], make_cuDoubleComplex(p, 0.0));
        x[m*M+n].y = 0.0;
    }
}

// void inv(cuComplex * F, cuComplex * F_inv, int M){
//     int num = 1; 
//     cublasHandle_t cublasH;
//     cublasCreate(&cublasH);

//     cuComplex ** A = new cuComplex*[num];
//     A[0] = F;
//     cuComplex ** A_d ;
//     cudaMalloc((void**)&A_d, num * sizeof(cuComplex *));
//     cudaMemcpy(A_d, A, num * sizeof(cuComplex *), cudaMemcpyHostToDevice);

//     int *info;
//     int *pivot;
//     cudaMalloc((void**)&info, num * sizeof(int));
//     cudaMalloc((void**)&pivot, M * num * sizeof(int));

//     //LU分解
//     cublasCgetrfBatched(cublasH, M, A_d, M, pivot, info, num);

//     cuComplex ** res = new cuComplex *[num];
//     res[0] = F_inv;
//     cuComplex ** res_d;
//     cudaMalloc((void**)&res_d, num * sizeof(cuComplex *));
//     cudaMemcpy(res_d, res, num*sizeof(cuComplex *), cudaMemcpyHostToDevice);

    
//     //求逆
//     cublasCgetriBatched(cublasH, M, A_d, M, pivot, res_d, M, info, num );

// }

__global__ void fun6(cuDoubleComplex*A,int M){
    int n = threadIdx.x;
    A[n*M+n] = make_cuDoubleComplex(1.0, 0.0);
}



void inv(cuDoubleComplex * F, cuDoubleComplex * F_inv, int M){
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);


    int lwork;
    cuDoubleComplex * work;
    int *devIpiv = nullptr;
    int *devInfo = nullptr;
    cusolverDnZgetrf_bufferSize(cusolverH,M,M,F,M,&lwork);
    
    cudaMalloc((void**)&work,lwork*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&devIpiv, M * sizeof(int));
    cusolverDnZgetrf(cusolverH, M, M, F, M, work, devIpiv, devInfo);

    fun6<<<1,M>>>(F_inv,M);

    cusolverDnZgetrs(cusolverH, CUBLAS_OP_N, M, M, F, M, devIpiv, F_inv, M, devInfo);

}





// void inv(cuComplex *F,cuComplex *F_inv,int M){

//     float *S;
//     cuComplex * U, * V;
//     cudaMalloc((void**)&S,M*sizeof(float));
//     cudaMalloc((void**)&U,M*M*sizeof(cuComplex));
//     cudaMalloc((void**)&V,M*M*sizeof(cuComplex));

//     int lwork;
//     cuComplex * work;
//     float * rwork = nullptr;
//     int *devInfo = nullptr;
//     cusolverDnHandle_t cusolverH;
//     cusolverDnCreate(&cusolverH);

//     cusolverDnCgesvd_bufferSize(cusolverH,M,M,&lwork);
//     cudaMalloc((void**)&work,lwork*sizeof(cuComplex));

//     signed char jobu = 'A';
//     signed char jobvt = 'A';

//     cusolverDnCgesvd(
//         cusolverH,jobu,jobvt,
//         M,M,F,M,
//         S,
//         U,M,
//         V,M,
//         work,lwork,rwork,
//         devInfo
//     );
//     cuComplex * VS;
//     cudaMalloc((void**)&VS, M*M*sizeof(cuComplex));
//     v_mul_s<<<1,dim3(M,M)>>>(V,S,VS,M,M);


//     cuComplex alpha = make_cuComplex(1.0,0.0);
//     cuComplex beta = make_cuComplex(0.0,0.0);

//     cublasHandle_t cublasH;
//     cublasCreate(&cublasH);
//     cublasCgemm(
//         cublasH,CUBLAS_OP_N,CUBLAS_OP_C,
//         M,M,M,
//         &alpha,
//         VS,M,
//         U,M,
//         &beta, 
//         F_inv,M
//     );

// }



cuDoubleComplex * F_update(
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
    double p,
    int M,
    int use){


    cuDoubleComplex * X, * Y, * H, *w;
    cudaMalloc((void**)&X, M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&Y, M*use*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&H, M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&w, M*M*sizeof(cuDoubleComplex));


    dim3 block(32,32);
    XY_update<<<3, block>>>(w,H,Y,v_u,v_d,w_u,w_d,H_d,use);

    cuDoubleComplex w_h[1024],test[1024],test2[1024];

    // for(int i=0;i<32;i++){
    //     printf("%d: %f %f\n",i,test[i].x,test[i].y);
    // }
    // for(int i=0;i<64;i++){
    //     printf("%d: %f %f\n",i,test[i].x,test[i].y);
    // }
    
    


    //X Y TODO:可以使用stream
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
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


    double temp = 0.5/p+0.08;
    mat_add_I<<<1,block>>>(H,temp);

    cudaMemcpy(test,H,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    for(int i=0;i<65;i++){
        printf("F %d: %f %f\n",i,test[i].x,test[i].y);
    }

    std::ofstream file("x.txt");


    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            double x,y;
            char c1,c2;
            file << test[j*M+i].x << " + " << test[j*M+i].y << "i ";
        }
        file << std::endl;
    }
    file.close();

    cuDoubleComplex * F_inv,*FF;
    cudaMalloc((void**)&F_inv, M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&FF, M*M*sizeof(cuDoubleComplex));
    cudaMemcpy(FF,H,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToDevice);
    
    inv(H, F_inv, M);



    cuDoubleComplex test5[1024];
    cudaMemcpy(test5,F_inv,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    for(int i=0;i<65;i++){
        printf("F_inv %d: %f %f\n",i,test5[i].x,test5[i].y);
    }



    cuDoubleComplex * E;
    cudaMalloc((void**)&E,M*M*sizeof(cuDoubleComplex));
    cublasZgemm(
        cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
        M,M,M,&alpha,
        FF,M,
        F_inv,M,
        &beta1,
        E,M
    );
    cudaMemcpy(test2,E,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    for(int i=0;i<64;i++){
        printf("E %d: %f %f\n",i,test2[i].x,test2[i].y);
    }

    return FF;




    // float l = 0.5 / p;
    // cuComplex alpha2 = {l,0.0};
    // float N = 8;

    // cublasCgemm(
    //     cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
    //     M, use, N, &alpha2,
    //     F_RF, M,
    //     F_BB, N,
    //     &beta2,
    //     Y,M
    // );
    
    // cudaMemcpy(test2,Y,M*use*sizeof(cuComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<32;i++){
    //     printf("%d: %f %f\n",i,test2[i].x,test2[i].y);
    // }

    // cublasCgemm(
    //     cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
    //     M,use,M,&alpha,
    //     F_inv,M,
    //     Y,M,
    //     &beta1,
    //     F,M
    // );

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
    int use){

    int F_RF_m = 32;
    int F_RF_n = 8;

    double *S;
    cuDoubleComplex * U, * V;
    cudaMalloc((void**)&S,F_RF_m*sizeof(double));
    cudaMalloc((void**)&U,F_RF_m*F_RF_m*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&V,F_RF_n*F_RF_n*sizeof(cuDoubleComplex));
    
    
    int lwork;
    cuDoubleComplex * work;
    double * rwork = nullptr;
    int *devInfo = nullptr;
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    cusolverDnZgesvd_bufferSize(cusolverH,F_RF_m,F_RF_n,&lwork);
    cudaMalloc((void**)&work,lwork*sizeof(cuDoubleComplex));
    
    //SVD
    signed char jobu = 'A';
    signed char jobvt = 'A';

    cusolverDnZgesvd(
        cusolverH,jobu,jobvt,
        F_RF_m,F_RF_n,F_RF,F_RF_m,
        S,
        U,F_RF_m,
        V,F_RF_n,
        work,lwork,rwork,
        devInfo
    );
    
    cuDoubleComplex * VS;
    cudaMalloc((void**)&VS, F_RF_n*F_RF_m*sizeof(cuDoubleComplex));
    v_mul_s<<<1,dim3(F_RF_n,F_RF_m)>>>(V,S,VS,F_RF_m,F_RF_n);

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0,0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0,0.0);
    cuDoubleComplex * F_RF_pinv;
    cudaMalloc((void**)&F_RF_pinv, F_RF_n*F_RF_m*sizeof(cuDoubleComplex));

    
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    cublasZgemm(
        cublasH,CUBLAS_OP_N,CUBLAS_OP_C,
        F_RF_n,F_RF_m,F_RF_m,
        &alpha,
        VS,F_RF_n,
        U,F_RF_m,
        &beta, 
        F_RF_pinv,F_RF_n
    );
    
    cublasZgemm(
        cublasH,CUBLAS_OP_N,CUBLAS_OP_N,
        F_RF_n,use,F_RF_m,
        &alpha,
        F_RF_pinv,F_RF_n,
        F,F_RF_m,
        &beta,
        F_BB,F_RF_n
    );

}

//F_RF_update

__global__ void RF_update(cuComplex * F_RF, cuComplex * A ,cuComplex *B){
    int k = blockIdx.x;
    int n = threadIdx.x;
    int M = 32;
    int N = 8;
    __shared__ cuComplex V[32];
    __shared__ cuComplex x[16];
    if(n < N){
        V[n] = F_RF[n*M+k]; 
    }

    for(int s=0;s<N;s++){
        if(n < N){
            x[n] = cuCmulf(make_cuComplex(-1.0*V[n].x,-1.0*V[n].y),A[s*N+n] );
        }
        else if(n == N){
            x[n] = B[s*N+k];
        }
        else if(n == N+1){
            x[n] = cuCmulf(V[s],A[s*N+s]);
        }
        else{
            x[n] = make_cuComplex(0.0,0.0);
        }
        __syncthreads();

        for(int i = 8; i > 0 ; i >>= 1){
            if(n < i)
                x[n] = cuCaddf(x[n], x[n+i]);
            __syncthreads();
        }
        __syncthreads();
        if(n==0)
            V[s] = cuCdivf(x[0],make_cuComplex(cuCabsf(x[0]),0.0));
        __syncthreads();
    }
    if(n < N){
        F_RF[n*M+k] = V[n] ; 
    }

}


void F_RF_update(
    cublasHandle_t cublasH,
    cuComplex* F_BB,
    cuComplex* F,
    cuComplex* F_RF,
    int use){

    int m = 32;
    int n = 8;

    cuComplex * A, *B;
    cudaMalloc((void**)&A, n*n*sizeof(cuComplex));
    cudaMalloc((void**)&B, m*n*sizeof(cuComplex));

    cuComplex alpha = {1.0,0.0};
    cuComplex beta = {0.0,0.0};
    cublasCgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_C,
        n,n,use,
        &alpha,
        F_BB,n,
        F_BB,n,
        &beta,
        A,n
    );
    cublasCgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_C,
        m,n,use,
        &alpha,
        F,m,
        F_BB,n,
        &beta,
        B,m
    );

    RF_update<<<m,m>>>(F_RF, A, B);

}

//V_u V_d update

__global__ void A_update(
    cuComplex * A,
    float N,
    cuComplex * F,
    float * p_u,
    cuComplex * H_u,
    cuComplex * H_SI,
    cuComplex * I_W2B,
    int use){

    int k = blockIdx.x;
    int m = threadIdx.x;
    int n = threadIdx.y;

    int M = 32;

    __shared__ cuComplex A_s[1024];
    if (k < use) {
        cuComplex a;
        a = cuCmulf(H_u[k*M+m], cuConjf(H_u[k*M+n]));
        A_s[n*M+m] = cuCmulf(make_cuComplex(p_u[k], 0.0), a);
    }
    else if (k <= use*2) {
        __shared__ cuComplex H_v[32];
        int k_u = k - use;
        if(m == 0 && n < M){
            cuComplex a = {0.0,0.0};
            for(int i=0; i<M; i++){
                a = cuCaddf(a, cuCmulf(H_SI[i*M+n], F[k*M+i]));
            }
            H_v[n] = a;
        }
        __syncthreads();

        A_s[n*M+m] = cuCmulf(H_v[m],H_v[n]);
    }

    atomicAdd(&A[n*M+m].x, A_s[n*M+m].x);
    atomicAdd(&A[n*M+m].y, A_s[n*M+m].y);

    if(k == 0){
        A[n*M+m] = cuCaddf(A[n*M+m], I_W2B[n*M+m]);
        if(m == n){
            A[n*M+m] = cuCaddf(A[n*M+m], make_cuComplex(N, 0.0));
        }
    }
}

__global__ void a_update(
    cuComplex * a,
    float N,
    cuComplex * H_V,
    float * p_u,
    cuComplex * H1,
    float * I_W2U,
    int use){
    
    int m = threadIdx.x;

    float a_f = 0;
    for(int i=0;i<use;i++){
        a_f += p_u[i] * pow(cuCabsf(H1[i*use+m]),2);
    }
    
    cuComplex aa;
    aa = cuCaddf(make_cuComplex(N, 0.0), H_V[m*use+m]);
    a_f = a_f + I_W2U[m];
    a[m] = cuCaddf(aa, make_cuComplex(a_f, 0.0));

}

__global__ void B_update(
    cuComplex * B,
    cuComplex * H_u,
    float * p_u){

    int m = threadIdx.x;
    int n = threadIdx.y;
    int M = 32;

    B[n*M+m] = cuCmulf(H_u[n*M+m],make_cuComplex(sqrt(p_u[n]),0.0));


}

__global__ void VE_update(
    cuComplex * v_u,
    cuComplex * v_d,
    cuComplex * A,
    cuComplex * A_inv,
    cuComplex * B,
    cuComplex * a,
    cuComplex * b,
    float * e_u,
    float * e_d,
    int use){
    int k = blockIdx.x;
    int m = threadIdx.x;
    int n = threadIdx.y;
    int M = 32;

    if(k < use){
        __shared__ cuComplex v_u_l[32];
        if(m == 0){
            cuComplex vv = {0.0,0.0};
            for(int i=0;i<M;i++){
                vv = cuCaddf(vv, cuCmulf(A_inv[i*M+n], B[k*M+i]));
            }
            v_u_l[n] = vv;
            v_u[k*M+n] = vv;
        }
        __syncthreads();

        __shared__ float E_u[1024];
        E_u[n*M+m] = cuCmulf(cuConjf(v_u_l[m]), cuCmulf(A[n*M+m], v_u_l[n])).x;

        __syncthreads();

        if(m == 0){
            E_u[n*M+m] = E_u[n*M+m] - 2 * cuCmulf(cuConjf(v_u_l[n]), B[k*M+n]).x;
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
        e_u[k] = E_u[0] + 1.0;
    }
    else if(k < use*2){
        if(m == 0 && n == 0){
            cuComplex v_dd;
            v_dd = cuCdivf(b[k*use+k], a[k]);
            v_d[k] = v_dd;
            e_d[k] = a[k].x * pow(cuCabsf(v_dd),2) - 2 * cuCmulf(v_dd,b[k*use+k]).x + 1;
        }
    }

}

void V_update(
    cublasHandle_t cublasH,
    float N,
    cuComplex * F,
    float * p_u,
    cuComplex * v_u,
    cuComplex * v_d,
    float * e_u,
    float * e_d,
    cuComplex * H_u,
    cuComplex * H_SI,
    cuComplex * H_d,
    cuComplex * H1,
    cuComplex * I_W2B,
    float * I_W2U,
    int use){
    
    int M =32;

    cuComplex *A,*B, *a, *b;
    cudaMalloc((void**)&A, M*M*sizeof(cuComplex));

    A_update<<<use*2, dim3(M,M)>>>(A, N, F, p_u, H_u, H_SI, I_W2B, use);

    cuComplex *H_F,*H_F_t;
    cudaMalloc((void**)&H_F, use*use*sizeof(cuComplex));
    cudaMalloc((void**)&H_F_t, use*use*sizeof(cuComplex));

    cuComplex alpha = {1.0,0.0};
    cuComplex beta = {0.0,0.0};

    cublasCgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        use,use,M,
        &alpha,
        H_d,use,
        F,M,
        &beta,
        H_F,use
    );
    cublasCgemm(
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
    cublasCgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        use,use,M,
        &alpha,
        H_d,use,
        F,M,
        &beta,
        b,use
    );

    cuComplex * A_inv;
    cudaMalloc((void**)&A_inv, M*M*sizeof(cuComplex));
    //inv(A, A_inv, M);
    
    VE_update<<<use*2, dim3(M,M)>>>(v_u, v_d, A, A_inv, B, a, b, e_u, e_d, use);
}


//计算f

__global__ void f_cal( 
    float * f,   
    float * w_u,
    float * w_d,
    float * e_u,
    float * e_d,
    float F_norm,
    int p,
    int use){
    
    int m = threadIdx.x;
    int n = threadIdx.y;

    __shared__ float sh[1024];
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
    __syncthreads();

    int tid = m*use+n;
    for (int i = 2*use; i > 0; i >>= 1)
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

float F_cal(
    cublasHandle_t cublasH,
    float * w_u,
    float * w_d,
    float * e_u,
    float * e_d,
    cuComplex * F,
    cuComplex * F_RF,
    cuComplex * F_BB,
    int use,
    int p,
    int RF){
    
    int M = 32;

    cuComplex * F_gap;
    float F_norm;
    cudaMalloc((void**)&F_gap, M*use*sizeof(cuComplex));
    cudaMemcpy(F_gap, F, M*use*sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cuComplex alpha = {-1.0,0.0};
    cuComplex beta = {1.0,0.0};
    
    cublasCgemm(
        cublasH,
        CUBLAS_OP_N,CUBLAS_OP_N,
        M,use,RF,
        &alpha,
        F_RF, M,
        F_BB, RF,
        &beta,
        F_gap,M
    );

    cublasScnrm2(
        cublasH, M*use,
        F_gap, 1, &F_norm
    );
    float * f;
    cudaMalloc((void **)&f,sizeof(float));
    f_cal<<<1,dim3(4,use)>>>(f, w_u, w_d, e_u, e_d, F_norm, p, use);

    float f_h;
    cudaMemcpy(&f_h, f, sizeof(float), cudaMemcpyDeviceToHost);
    return f_h;

}


// int main() {

//     double e_u[ 4 ] = { 1,2,3,4 };
//     double e_d[ 4 ] = { 1,2,3,4 };

//     int ul_use = 4;

//     double* e_u_d, * e_d_d;
//     cudaMalloc((void**)&e_d_d, 4 * sizeof(double));
//     cudaMalloc((void**)&e_u_d, 4 * sizeof(double));

//     cudaMemcpy(e_u_d, e_u, ul_use * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(e_d_d, e_d, ul_use * sizeof(double), cudaMemcpyHostToDevice);


//     double* sigma_u, * sigma_d;
//     cudaMalloc((void**)&sigma_u, 4 * sizeof(double));
//     cudaMalloc((void**)&sigma_d, ul_use * sizeof(double));

//     printf("%f", e_u[ 0 ]);

//     dim3 blocksize(1, 1, 8);
//     w_updata << < 1, blocksize >> > (sigma_u, sigma_d, e_u_d, e_d_d, ul_use);
//     cudaDeviceSynchronize();



//     return 0;
// }
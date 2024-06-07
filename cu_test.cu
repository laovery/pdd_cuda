#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cusolverDn.h>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
using namespace std;


void fun1(double * a, double * b,double * c){
    double l = 1.0;
    double k = 0.0;
    cublasHandle_t cublasH ;
    cublasCreate (&cublasH);
    cublasDgemm(cublasH,
                CUBLAS_OP_N,CUBLAS_OP_N,
                2,2,2,
                &l,a,2,
                b,2,
                &k,c,2);
                
}


__global__ void fun6(cuDoubleComplex*A){
    int n = threadIdx.x;
    A[n*32+n] = make_cuDoubleComplex(1.0, 0.0);
}

__global__ void fun7(cuComplex*A, cuDoubleComplex *F,int M){
    int m = threadIdx.x;
    int n = threadIdx.y;
    F[n*M+m] = cuComplexFloatToDouble(A[n*M+m]);
}


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

void inv(cuDoubleComplex *F,cuDoubleComplex *F_inv,int M){

    double *S;
    cuDoubleComplex * U, * V;
    cudaMalloc((void**)&S,M*sizeof(double));
    cudaMalloc((void**)&U,M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&V,M*M*sizeof(cuDoubleComplex));

    int lwork;
    cuDoubleComplex * work;
    double * rwork = nullptr;
    int *devInfo = nullptr;
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    cusolverDnCgesvd_bufferSize(cusolverH,M,M,&lwork);
    cudaMalloc((void**)&work,lwork*sizeof(cuDoubleComplex));

    signed char jobu = 'A';
    signed char jobvt = 'A';

    cusolverDnZgesvd(
        cusolverH,jobu,jobvt,
        M,M,F,M,
        S,
        U,M,
        V,M,
        work,lwork,rwork,
        devInfo
    );
    cuDoubleComplex * VS;
    cudaMalloc((void**)&VS, M*M*sizeof(cuDoubleComplex));
    v_mul_s<<<1,dim3(M,M)>>>(V,S,VS,M,M);


    cuDoubleComplex alpha = make_cuDoubleComplex(1.0,0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0,0.0);

    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    cublasZgemm(
        cublasH,CUBLAS_OP_N,CUBLAS_OP_C,
        M,M,M,
        &alpha,
        VS,M,
        U,M,
        &beta, 
        F_inv,M
    );

}


void invv(cuDoubleComplex * F, int M){
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    cuDoubleComplex *F_inv,*FF;
    cudaMalloc((void**)&F_inv,M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&FF,M*M*sizeof(cuDoubleComplex));
    cudaMemcpy(FF,F,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToDevice);


    int lwork;
    cuDoubleComplex * work;
    int *devIpiv = nullptr;
    int *devInfo = nullptr;
    cusolverDnZgetrf_bufferSize(cusolverH,M,M,F,M,&lwork);
    cudaMalloc((void**)&work,lwork*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&devIpiv, M * sizeof(int));
    cusolverDnZgetrf(cusolverH, M, M, F, M, work, devIpiv, devInfo);

    fun6<<<1,M>>>(F_inv);

    cusolverDnZgetrs(cusolverH, CUBLAS_OP_N, M, M, F, M, devIpiv, F_inv, M, devInfo);

    cuDoubleComplex test[1024];
    cudaMemcpy(test,F_inv,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    for(int i=0;i<32;i++){
        printf("inv : %lf %lf\n",test[i].x,test[i].y);
    }
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    cuDoubleComplex alpha = {1.0,0.0};
    cuDoubleComplex beta1 ={0.0,0.0};
    cuDoubleComplex beta2 ={1.0,0.0};
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
    cudaMemcpy(test,E,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    for(int i=0;i<64;i++){
        printf("E %d: %f %f\n",i,test[i].x,test[i].y);
    }
}




__global__ void fun3(cuComplex * b){
    int k = blockIdx.x;
    int m = threadIdx.x;
    int n = threadIdx.y;
    __shared__ cuComplex s[4];

    if(k < 2)
        s[n*2+m] = make_cuComplex(n*2+m, 0.0);
    else if(k < 4)
        s[n*2+m] = make_cuComplex(2, 1.0);
    //printf("%d %d : %f %f\n",k,m,s[m].x,s[m].y);
    

    atomicAdd(&b[n*2+m].x, s[n*2+m].x);
    atomicAdd(&b[n*2+m].y, s[n*2+m].y);

    // if(k < 5){
    //     b[m] = cuCaddf(b[m], s[m]);
    //     __threadfence();
    // }
    // if(k == 3)
    //     b[m] = cuCaddf(b[m], s[m]);
    // __threadfence();
    
}

__global__ void fun4(int c){
    int m = threadIdx.x;
    cuComplex a = {-3029.961426,10077.462891};
    cuComplex b = cuCmulf(a, cuConjf(a));
    printf("%f %f\n",b.x,b.y);

}

int main() {

    float *f ;
    float f_h = 0.1;
    int use = 4;

    
    std::vector<cuComplex> A_h = {
        {1,0},{2, -3},{3,-5},
        {2, 1}};
    float e_u[ 4 ] = { 1,2,3,4 };
    float e_d[ 4 ] = { 2,2,2,2 };
    std::vector<cuComplex> B_h = {
        {1,0},{2,-3},{3,-5},
        {2,3}};
    int ul_use = 4;

    float* e_u_d, * e_d_d;
    cudaMalloc((void**)&e_d_d, 4 * sizeof(float));
    cudaMalloc((void**)&e_u_d, 4 * sizeof(float));

    cudaMemcpy(e_u_d, e_u, ul_use * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(e_d_d, e_d, ul_use * sizeof(float), cudaMemcpyHostToDevice);


    //fun4<<<1,1>>>(1);



    // float* sigma_u, * sigma_d;
    // cudaMalloc((void**)&sigma_u, 4 * sizeof(float));
    // cudaMalloc((void**)&sigma_d, ul_use * sizeof(float));
    
    // w_update<<<1, use*2 >>>(sigma_u,sigma_d,e_u_d,e_d_d,use);

    // float w_u[4];
    // float w_d[4];
    // cudaMemcpy(w_u, sigma_u, 4*sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(w_d, sigma_d, 4*sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i=0;i<4;i++){
    //     printf("%f : %f %f\n",e_u[i],w_u[i],w_d[i]);
    // }

    int M = 32;
    cuDoubleComplex * a_d, * b_d, * c_d, * B_d;
    cudaMalloc((void**)&a_d,M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&c_d,M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&B_d,M*M*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&b_d, M*M*sizeof(cuDoubleComplex));

    std::ifstream file("x.txt");

    cuDoubleComplex A[1024];

    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            double x,y;
            char c1,c2;
            file >> x >> c1 >>y>>c2;
            A[j*M+i] = make_cuDoubleComplex(x, y);
        }
    }
    file.close();

    for (int i=0;i<65;i++) {
        cout<<"F:"<<A[i].x<<" "<<A[i].y<<endl;
    }
    cudaMemcpy(a_d,A,M*M*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    invv(a_d,  M);

    // cuDoubleComplex test[1024];
    // cudaMemcpy(test,c_d,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    // for(int i=0;i<32;i++){
    //     printf("inv : %lf %lf\n",test[i].x,test[i].y);
    // }



    // int x ;
    // cudaDeviceProp porp;
    // cudaGetDeviceProperties(&porp, 0);
    // cudaDeviceGetAttribute(&x, cudaDevAttrMaxGridDimX, 0);
    // std::cout << "Device " << 1 << ": " << porp.name << std::endl;
    // std::cout << "Number of cores: " << porp.maxThreadsPerBlock << std::endl;
    // std::cout << "Number of cores: " << x<< std::endl;

    //统计时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);  // 记录开始时间

    //fun3<<<4,dim3(2,2)>>>( B_d);


    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);   // 记录结束时间
    cudaEventSynchronize(stop); // 等待事件完成

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop); // 计算时间差

    printf("run time: %0.4fms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // cuComplex B[4];
    // cudaMemcpy(B, B_d, 4*sizeof(cuComplex), cudaMemcpyDeviceToHost);
    // for (int i=0; i<4;i++) {
    //     printf("%f %f\n",B[i].x,B[i].y);
    // }



    return 0;
}
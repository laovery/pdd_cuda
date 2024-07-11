#ifndef CU_PDD_H
#define CU_PDD_H


__global__ void w_update(double * w_u, double * w_d , double * e_u,double * e_d,double * A,int size);

__global__ void p_update(
    double* p_u, 
    double* w_u,
    double* w_d,
    cuDoubleComplex* v_u,
    cuDoubleComplex* v_d,
    cuDoubleComplex* H_u,
    cuDoubleComplex* H1, 
    cuDoubleComplex* H3,
    double* A,
    double I_th,
    int M,
    int use);

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
    cusolverDnHandle_t cusolverH);

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
    cusolverDnHandle_t cusolverH);

void F_RF_update(
    cuDoubleComplex* F_BB,
    cuDoubleComplex* F,
    cuDoubleComplex* F_RF,
    cuDoubleComplex* lambda,
    double p,
    int M,
    int N,
    int use,
    cublasHandle_t cublasH);

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
    cusolverDnHandle_t cusolverH);



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
    int use_norm);
 

double cv_cal(
    cuDoubleComplex * F,
    cuDoubleComplex * F_RF,
    cuDoubleComplex * F_BB,
    int M,
    int RF,
    int use,
    cublasHandle_t cublasH,
    int use_norm);


void lambda_update(
    cuDoubleComplex * F,
    cuDoubleComplex * F_RF,
    cuDoubleComplex * F_BB,
    cuDoubleComplex * lambda,
    double p,
    int M,
    int RF,
    int use,
    cublasHandle_t cublasH);

#endif 
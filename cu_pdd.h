#ifndef CU_PDD_H
#define CU_PDD_H


__global__ void w_update(double * w_u, double * w_d , double * e_u,double * e_d,int size);

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
    int use);

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
    int use);

// void F_BB_update(
//     cuDoubleComplex * F_RF,
//     cuDoubleComplex * F,
//     cuDoubleComplex * F_BB,
//     int use);

// void F_RF_update(
//     cublasHandle_t cublasH,
//     cuDoubleComplex* F_BB,
//     cuDoubleComplex* F,
//     cuDoubleComplex* F_RF,
//     int use);

// void V_update(
//     cublasHandle_t cublasH,
//     double N,
//     cuDoubleComplex * F,
//     double * p_u,
//     cuDoubleComplex * v_u,
//     cuDoubleComplex * v_d,
//     double * e_u,
//     double * e_d,
//     cuDoubleComplex * H_u,
//     cuDoubleComplex * H_SI,
//     cuDoubleComplex * H_d,
//     cuDoubleComplex * H1,
//     cuDoubleComplex * I_W2B,
//     double * I_W2U,
//     int use);

// void inv(cuDoubleComplex * F, cuDoubleComplex * F_inv, int M);

// double F_cal(
//     cublasHandle_t cublasH,
//     double * w_u,
//     double * w_d,
//     double * e_u,
//     double * e_d,
//     cuDoubleComplex * F,
//     cuDoubleComplex * F_RF,
//     cuDoubleComplex * F_BB,
//     int use,
//     int p,
//     int RF);
 



#endif 
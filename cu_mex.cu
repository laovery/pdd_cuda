#include <cstdio>
#include <mex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cu_pdd.h"

cuDoubleComplex * init_mat(double * real , double * imag , int size){
    cuDoubleComplex* complexMatrix = new cuDoubleComplex[size];
    for (size_t i = 0; i < size; i++)
    {
        complexMatrix[i] = make_cuDoubleComplex(real[i], imag[i]);
        /* code */
    }
    return complexMatrix;
}

// float * init_vec(double * vecter, int size){
//     float * vec = new float[size];
//     for(int i=0;i<size;i++)
//         vec[i] = static_cast<float>(vecter[i]);
//     return vec;
// }



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //输入
    //ul_user,dl_user,H_SI,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,I_W2B,I_W2U,p_u,V,V_RF,V_BB,W,Q,p,lambda,e_u,e_d;
    // 检查输入参数数量
    if (nrhs != 26) {
        mexErrMsgIdAndTxt("MATLAB:cublasMultiply:invalidNumInputs",
                          "两个输入参数是必需的。");
    }
    
    //读取数据
    int use = mxGetScalar(prhs[0]);
    int RF = mxGetScalar(prhs[1]);
    double * H_si_r = mxGetPr(prhs[2]);
    double * H_si_i = mxGetPi(prhs[2]);
    double * H_u_r = mxGetPr(prhs[3]);
    double * H_u_i = mxGetPi(prhs[3]);
    double * H_d_r = mxGetPr(prhs[4]);
    double * H_d_i = mxGetPi(prhs[4]);
    double * H1_r = mxGetPr(prhs[5]);
    double * H1_i = mxGetPi(prhs[5]);
    double * H2_r = mxGetPr(prhs[6]);
    double * H2_i = mxGetPi(prhs[6]);
    double * H3_r = mxGetPr(prhs[7]);
    double * H3_i = mxGetPi(prhs[7]);
    double N = mxGetScalar(prhs[8]);
    double I = mxGetScalar(prhs[9]);
    int p_use = mxGetScalar(prhs[10]);
    int p_BS = mxGetScalar(prhs[11]);
    int m_BS = mxGetScalar(prhs[12]);
    int n_BS = mxGetScalar(prhs[13]);
    double * I_W2B_r = mxGetPr(prhs[14]);
    double * I_W2B_i = mxGetPi(prhs[14]);
    double * I_W2U = mxGetPr(prhs[15]);
    double * p_u = mxGetPr(prhs[16]);
    double * V_r = mxGetPr(prhs[17]);
    double * V_i = mxGetPi(prhs[17]);
    double * V_RF_r = mxGetPr(prhs[18]);
    double * V_RF_i = mxGetPi(prhs[18]);
    double * V_BB_r = mxGetPr(prhs[19]);
    double * V_BB_i = mxGetPi(prhs[19]);
    double * W_r = mxGetPr(prhs[20]);
    double * W_i = mxGetPi(prhs[20]);
    double * Q_r = mxGetPr(prhs[21]);
    double * Q_i = mxGetPi(prhs[21]);
    double p = mxGetScalar(prhs[22]);
    double * lambda = mxGetPr(prhs[23]);
    double * e_u = mxGetPr(prhs[24]);
    double * e_d = mxGetPr(prhs[25]);
    


    cuDoubleComplex * H_si = init_mat(H_si_r,H_si_i,mxGetN(prhs[2])*mxGetM(prhs[2]));
    cuDoubleComplex * H_u = init_mat(H_u_r,H_u_i,mxGetN(prhs[3])*mxGetM(prhs[3]));
    cuDoubleComplex * H_d = init_mat(H_d_r,H_d_i,mxGetN(prhs[4])*mxGetM(prhs[4]));
    cuDoubleComplex * H1 = init_mat(H1_r,H1_i,mxGetN(prhs[5])*mxGetM(prhs[5]));
    cuDoubleComplex * H2 = init_mat(H2_r,H2_i,mxGetN(prhs[6])*mxGetM(prhs[6]));
    cuDoubleComplex * H3 = init_mat(H3_r,H3_i,mxGetN(prhs[7])*mxGetM(prhs[7]));
    cuDoubleComplex * I_W2B = init_mat(I_W2B_r,I_W2B_i,mxGetN(prhs[14])*mxGetM(prhs[14]));
    cuDoubleComplex * F = init_mat(V_r,V_i,mxGetN(prhs[17])*mxGetM(prhs[17]));
    cuDoubleComplex * F_RF = init_mat(V_RF_r,V_RF_i,mxGetN(prhs[18])*mxGetM(prhs[18]));
    cuDoubleComplex * F_BB = init_mat(V_BB_r,V_BB_i,mxGetN(prhs[19])*mxGetM(prhs[19]));
    cuDoubleComplex * v_u = init_mat(W_r,W_i,mxGetN(prhs[20])*mxGetM(prhs[20]));
    cuDoubleComplex * v_d = init_mat(Q_r,Q_i,mxGetN(prhs[21])*mxGetM(prhs[21]));




    double *I_W2U_d, *p_u_d, *e_u_d, *e_d_d;
    cuDoubleComplex * H_si_d, *H_u_d, *H_d_d, *H1_d, *H2_d, *H3_d, *I_W2B_d, *F_d, *F_RF_d, *F_BB_d, *v_u_d, *v_d_d;


    //转移到gpu上
    cudaMalloc((void **)&e_d_d, use*sizeof(double));
    cudaMalloc((void **)&e_u_d, use*sizeof(double));
    cudaMalloc((void **)&p_u_d, use*sizeof(double));
    cudaMalloc((void **)&v_u_d, m_BS*use*sizeof(cuDoubleComplex));
    cudaMalloc((void **)&v_d_d, use*sizeof(cuDoubleComplex));
    cudaMalloc((void **)&F_d, m_BS*use*sizeof(cuDoubleComplex));
    cudaMalloc((void **)&F_RF_d, m_BS*RF*sizeof(cuDoubleComplex));
    cudaMalloc((void **)&F_BB_d, RF*use*sizeof(cuDoubleComplex));

    cudaMalloc((void **)&H_si_d, m_BS*m_BS*sizeof(cuDoubleComplex));
    cudaMalloc((void **)&H_u_d, m_BS*use*sizeof(cuDoubleComplex));
    cudaMalloc((void **)&H_d_d, use*m_BS*sizeof(cuDoubleComplex));
    cudaMalloc((void **)&H1_d, use*use*sizeof(cuDoubleComplex));
    cudaMalloc((void **)&H2_d, m_BS*sizeof(cuDoubleComplex));
    cudaMalloc((void **)&H3_d, use*sizeof(cuDoubleComplex));





    cudaMemcpy(H_si_d,H_si,m_BS*m_BS*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(H_u_d,H_u,m_BS*use*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(H_d_d,H_d,use*m_BS*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(H1_d,H1,use*use*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(H2_d,H2,m_BS*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(H3_d,H3,use*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    
    //更新参数赋值，之后删除
    cudaMemcpy(e_u_d,e_u,use*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(e_d_d,e_d,use*sizeof(double),cudaMemcpyHostToDevice);
  
    cudaMemcpy(F_d,F,use*m_BS*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(F_RF_d,F_RF,m_BS*RF*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(F_BB_d,F_BB,RF*use*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(v_u_d,v_u,use*m_BS*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(v_d_d,v_d,use*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);




    double * w_u_d, * w_d_d;
    cudaMalloc((void**)&w_u_d,use * sizeof(double));
    cudaMalloc((void**)&w_d_d,use * sizeof(double));


    cuDoubleComplex h_si[16];


    //初始化参数
    double I_th = I/(2*use);
    
    //设置参数
    double s1 = 0.8, c1 = 1E-4, c2 = 1E-2, c3 = 1E-3;
    double cv = c1, delta = c2;

    w_update<<<1,use*2>>>(w_u_d,w_d_d,e_u_d,e_d_d,use);


    double ww[4];
    cudaMemcpy(ww,w_u_d,use*sizeof(double),cudaMemcpyDeviceToHost);
    for(int i=0;i<use;i++){
        printf("%f\n",ww[i]);
    }

    cuDoubleComplex * A;
    p_update<<<dim3(use, use),m_BS>>>(p_u_d, w_u_d,w_d_d, v_u_d, v_d_d,
                                                                        H_u_d, H1_d, H3_d, I_th, m_BS, use);

    double pp[4];
    cudaMemcpy(pp,p_u_d,use*sizeof(double),cudaMemcpyDeviceToHost);
    for(int i=0;i<use;i++){
        printf("%f\n",pp[i]);
    }



    A = F_update(F_d,F_RF_d,F_BB_d,v_u_d,v_d_d,w_u_d,w_d_d,H_d_d,H_si_d,H1_d,p,m_BS,use);

    cuDoubleComplex test3[1024];
    cudaMemcpy(test3,A,m_BS*m_BS*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);


    plhs[0] = mxCreateDoubleMatrix(m_BS,m_BS, mxCOMPLEX);
    double *real = mxGetPr(plhs[0]);
    double *imag = mxGetPi(plhs[0]);

    for (int i=0; i<32; i++) {
        for (int j=0; j<32; j++) {
            real[i*m_BS+j] = (double)test3[i*m_BS+j].x;
            imag[i*m_BS+j] = (double)test3[i*m_BS+j].y;

        }
    }
    // cuComplex ff[128];
    // cudaMemcpy(ff,F_d,m_BS*use*sizeof(cuComplex),cudaMemcpyDeviceToHost);

    // for(int i=0;i<32;i++){
    //     printf("%d: %f %f\n",i,ff[i].x,ff[i].y);
    // }


}
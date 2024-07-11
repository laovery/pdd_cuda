#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
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




    double *I_W2U_d, *p_u_d, *e_u_d, *e_d_d, *A;
    cuDoubleComplex * H_si_d, *H_u_d, *H_d_d, *H1_d, *H2_d, *H3_d, *I_W2B_d, *F_d, *F_RF_d, *F_BB_d, *v_u_d, *v_d_d,*lambda_d;


    //转移到gpu上
    cudaMalloc((void **)&e_d_d, use*sizeof(double));
    cudaMalloc((void **)&e_u_d, use*sizeof(double));
    cudaMalloc((void **)&p_u_d, use*sizeof(double));
    cudaMalloc((void **)&I_W2U_d, use*sizeof(double));
    cudaMalloc((void **)&A, use*sizeof(double));


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
    cudaMalloc((void **)&I_W2B_d, m_BS*m_BS*sizeof(cuDoubleComplex));
    cudaMalloc((void **)&lambda_d, m_BS*use*sizeof(cuDoubleComplex));






    cudaMemcpy(H_si_d,H_si,m_BS*m_BS*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(H_u_d,H_u,m_BS*use*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(H_d_d,H_d,use*m_BS*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(H1_d,H1,use*use*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(H2_d,H2,m_BS*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(H3_d,H3,use*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(H3_d,H3,use*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(I_W2B_d,I_W2B,m_BS*m_BS*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(I_W2U_d,I_W2U,use*sizeof(double),cudaMemcpyHostToDevice);

    
    //更新参数赋值
    cudaMemcpy(e_u_d,e_u,use*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(e_d_d,e_d,use*sizeof(double),cudaMemcpyHostToDevice);
  
    cudaMemcpy(F_d,F,use*m_BS*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(F_RF_d,F_RF,m_BS*RF*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(F_BB_d,F_BB,RF*use*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(v_u_d,v_u,use*m_BS*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(v_d_d,v_d,use*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    //cudaMemcpy(p_u_d,p_u,use*sizeof(double),cudaMemcpyHostToDevice);
    cuDoubleComplex lam[1024] = {0.0,0.0};
    cudaMemcpy(lambda_d, lam, m_BS * use * sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);

    double * w_u_d, * w_d_d;
    cudaMalloc((void**)&w_u_d,use * sizeof(double));
    cudaMalloc((void**)&w_d_d,use * sizeof(double));



    //初始化参数
    double I_th = I/(2*use);
    
    //设置参数
    double s1 = 0.8, c1 = 1E-4, c2 = 1E-2, c3 = 1E-3;
    double cv = c1, delta = c2;

    int use_norm = 8;

    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);
    //统计时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // 记录开始时间

    while(cv >= c1){
        int inner = 0;
        double f[30] = {0.0};
        while (delta >= c2 && inner < 30) {

            w_update<<<1,use*2>>>(w_u_d,w_d_d,e_u_d,e_d_d,A,use);

            //double ww[6];
            // cudaMemcpy(ww,w_u_d,use*sizeof(double),cudaMemcpyDeviceToHost);
            // for(int i=0;i<use;i++){
            //     printf("%e\n",ww[i]);
            // }

            p_update<<<dim3(use, use),m_BS>>>(p_u_d, w_u_d,w_d_d, v_u_d, v_d_d,H_u_d, H1_d, H3_d, A, I_th, m_BS, use);


            // cudaMemcpy(ww,p_u_d,use*sizeof(double),cudaMemcpyDeviceToHost);
            // for(int i=0;i<use;i++){
            //     printf("%e\n",ww[i]);
            // }

            F_update(F_d,F_RF_d,F_BB_d,v_u_d,v_d_d,w_u_d,w_d_d,H_d_d,H_si_d,H1_d,lambda_d,p,m_BS,use,cublasH,cusolverH);

            F_BB_update(F_RF_d,F_d,F_BB_d,lambda_d,p,m_BS,RF,use,cublasH,cusolverH);

            F_RF_update(F_BB_d,F_d,F_RF_d,lambda_d,p,m_BS,RF,use,cublasH);
            
            V_update(N,F_d,p_u_d,v_u_d,v_d_d,e_u_d,e_d_d,H_u_d,H_si_d,H_d_d,H1_d,I_W2B_d,I_W2U_d,use,cublasH,cusolverH);

            f[inner] = F_cal(w_u_d,w_d_d,e_u_d,e_d_d,F_d,F_RF_d,F_BB_d,lambda_d,use,p,RF,cublasH,use_norm);

            //printf("%e\n",f[inner]);

            if(inner >= 1)
                delta = fabs((f[inner] - f[inner-1]) / f[inner]);

            inner+=1;
        }

        delta = c2;

        cv = cv_cal(F_d,F_RF_d,F_BB_d,m_BS,RF,use,cublasH,use_norm);

        //printf("%e\n",cv);

        if(cv > c3)
            p = s1 * p;
        else{
            //printf("lambda_update!!!\n");
            lambda_update(F_d,F_RF_d,F_BB_d,lambda_d,p,m_BS,RF,use,cublasH);
            c3 = s1 * cv;
        }
        //printf("%e %e\n",cv,c1);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);   // 记录结束时间
    cudaEventSynchronize(stop); // 等待事件完成

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop); // 计算时间差

    printf("run time: %0.4fms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    plhs[0] = mxCreateDoubleMatrix(1,use,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(m_BS,use,mxCOMPLEX);
    plhs[2] = mxCreateDoubleMatrix(m_BS,RF,mxCOMPLEX);
    plhs[3] = mxCreateDoubleMatrix(RF,use,mxCOMPLEX);
    plhs[4] = mxCreateDoubleMatrix(1,use,mxREAL);
    plhs[5] = mxCreateDoubleMatrix(1,use,mxREAL);
    plhs[6] = mxCreateDoubleMatrix(m_BS,use,mxCOMPLEX);
    plhs[7] = mxCreateDoubleMatrix(use,1,mxCOMPLEX);



    double * p_u_out = mxGetPr(plhs[0]);
    double * V_r_out = mxGetPr(plhs[1]);
    double * V_i_out = mxGetPi(plhs[1]);
    double * V_RF_r_out = mxGetPr(plhs[2]);
    double * V_RF_i_out = mxGetPi(plhs[2]);
    double * V_BB_r_out = mxGetPr(plhs[3]);
    double * V_BB_i_out = mxGetPi(plhs[3]);
    double * e_u_out = mxGetPr(plhs[4]);
    double * e_d_out = mxGetPr(plhs[5]);
    double * v_u_r_out = mxGetPr(plhs[6]);
    double * v_u_i_out = mxGetPi(plhs[6]);
    double * v_d_r_out = mxGetPr(plhs[7]);
    double * v_d_i_out = mxGetPi(plhs[7]);

    double p_u_h[use],e_u_h[use],e_d_h[use];
    cuDoubleComplex V_h[m_BS*use];
    cuDoubleComplex V_RF_h[m_BS*RF];
    cuDoubleComplex V_BB_h[RF*use];
    cuDoubleComplex v_u_h[m_BS*use];
    cuDoubleComplex v_d_h[use];


    cudaMemcpy(p_u_h,p_u_d,use*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(e_u_h,e_u_d,use*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(e_d_h,e_d_d,use*sizeof(double),cudaMemcpyDeviceToHost);
    
    
    cudaMemcpy(V_h,F_d,use*m_BS*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    cudaMemcpy(V_RF_h,F_RF_d,m_BS*RF*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    cudaMemcpy(V_BB_h,F_BB_d,RF*use*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    cudaMemcpy(v_u_h,v_u_d,use*m_BS*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    cudaMemcpy(v_d_h,v_d_d,use*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);


    for(int i=0;i<use;i++){
        p_u_out[i] = p_u_h[i];
    }
    for(int i=0;i<use;i++){
        e_u_out[i] = e_u_h[i];
    }
    for(int i=0;i<use;i++){
        e_d_out[i] = e_d_h[i];
    }

    for (int i = 0; i < use*m_BS; i++)
    {
        V_r_out[i] = V_h[i].x;
        V_i_out[i] = V_h[i].y;
        /* code */
    }
    for (int  i = 0; i < m_BS*RF; i++)
    {
        V_RF_r_out[i] = V_RF_h[i].x;
        V_RF_i_out[i] = V_RF_h[i].y;
        /* code */
    }
    for (int  i = 0; i < use*RF; i++)
    {
        V_BB_r_out[i] = V_BB_h[i].x;
        V_BB_i_out[i] = V_BB_h[i].y;
        /* code */
    }
    for (int i = 0; i < use*m_BS; i++)
    {
        v_u_r_out[i] = v_u_h[i].x;
        v_u_i_out[i] = v_u_h[i].y;
        /* code */
    }
    for (int i = 0; i < use; i++)
    {
        v_d_r_out[i] = v_d_h[i].x;
        v_d_i_out[i] = v_d_h[i].y;
        /* code */
    }
    

    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);
    cudaDeviceReset();
}
#include<iostream>
#include"cuda_runtime.h"
#include<cublas_v2.h>
#include<stdlib.h>
#include<time.h>
 
 
//矩阵的阶数
#define N 3
//有两个矩阵
#define NUM 2
 
int main()
{
	//开辟一个二维的数组空间
	float **matHost = new float*[NUM];
	for(int i=0;i<NUM;i++)
		matHost[i] = new float[N*N];
	
	//matHost[0] = {-0.997497,0.617481,-0.299417,0.127171,0.170019,
	//0.791925,-0.613392,-0.0402539,0.64568};
	matHost[0][0] = -0.997497;
	matHost[0][1] = 0.617481;
	matHost[0][2] = -0.299417;
	matHost[0][3] = 0.127171;
	matHost[0][4] = 0.170019;
	matHost[0][5] = 0.791925;
	matHost[0][6] = -0.613392;
	matHost[0][7] = -0.0402539;
	matHost[0][8] = 0.64568;
 
 
	//随机初始化矩阵，所有矩阵被初始化成一样的
	for(int j=1;j<NUM;j++)
	{
		for(int i=0;i<N*N;i++)
		{
			matHost[j][i] = matHost[0][i];
		}
	}	
 
	//指针在host端，内容却在device端
	float **srchd = new float*[NUM];
	
	for(int i=0;i<NUM;i++)
	{
		cudaMalloc((void**)&srchd[i],sizeof(float)*N*N);
		cudaMemcpy(srchd[i],matHost[i],sizeof(float)*N*N,cudaMemcpyHostToDevice);
	}
 
	float **srcDptr;
	cudaMalloc((void**)&srcDptr,sizeof(float*)*NUM);
	cudaMemcpy(srcDptr,srchd,sizeof(float*)*NUM,cudaMemcpyHostToDevice);
 
 
	//用来记录LU分解是否成功，0表示分解成功
	int *infoArray;
	cudaMalloc((void**)&infoArray,sizeof(int)*NUM);
 
	int *pivotArray;
	cudaMalloc((void**)&pivotArray,sizeof(int)*N*NUM);
 
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle); 
 
	//LU分解,原地的
	cublasSgetrfBatched(cublasHandle,N,srcDptr,N,pivotArray,infoArray,NUM);
 
	float **resulthd = new float*[NUM];
	for(int i=0;i<NUM;i++)
		cudaMalloc((void**)&resulthd[i],sizeof(float)*N*N);
 
	float **resultDptr;
	cudaMalloc((void**)&resultDptr,sizeof(float*)*NUM);
	cudaMemcpy(resultDptr,resulthd,sizeof(float*)*NUM,cudaMemcpyHostToDevice);
 
	//把LU分解的结果变成逆矩阵
	cublasSgetriBatched(cublasHandle,N,(const float**)srcDptr,N,pivotArray,resultDptr,N,infoArray,NUM);
 
	float **invresult = new float*[NUM];
	for(int i=0;i<NUM;i++)
	{
		invresult[i] = new float[N*N];
		//注意是resulthd[i]而不是resultDptr[i]，否则会出错
		cudaMemcpy(invresult[i],resulthd[i],sizeof(float)*N*N,cudaMemcpyDeviceToHost);
	}
		
 
	int *infoArrayHost = new int[NUM];
	cudaMemcpy(infoArrayHost,infoArray,sizeof(int)*NUM,cudaMemcpyDeviceToHost);
 
	std::cout<<"info array:"<<std::endl;
	for(int i=0;i<NUM;i++)
		std::cout<<infoArrayHost[i]<<"  ";
	std::cout<<std::endl;
 
	cublasDestroy(cublasHandle);
 
	std::cout<<"LU decomposition result:"<<std::endl;
	for(int i=0;i<N*N;i++)
	{	
		if(i%N == 0)
			std::cout<<std::endl;
 
		std::cout<<invresult[0][i]<<"  ";	
	}
	std::cout<<std::endl;
 
	//释放空间
	for(int i=0;i<NUM;i++)
	{
		cudaFree(srchd[i]);
		delete []matHost[i];
		matHost[i] = NULL;
		cudaFree(resulthd[i]);
		delete []invresult[i];
		invresult[i] = NULL;
	}
 
	delete []matHost;
	matHost = NULL;
	delete []resulthd;
	resulthd = NULL;
	delete []invresult;
	invresult = NULL;
 
	delete []infoArrayHost;
	infoArrayHost = NULL;
 
	delete []srchd;
	srchd = NULL;
	
	cudaFree(infoArray);
	cudaFree(pivotArray);
	cudaFree(srcDptr);
	cudaFree(resultDptr);
 
	return 0;
 
}
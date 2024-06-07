
#include <complex.h>
#include <cstdio>
#include <armadillo>
#include <iostream>
using namespace arma;
using namespace std;

int main()
{
    //输入
    //ul_user,dl_user,H_SI,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,I_W2B,I_W2U,p_u,V,V_RF,V_BB,W,Q,p,lambda,e_u,e_d;
    // 检查输入参数数量



    //arma求解X
    std::ifstream file("x.txt");


    int M=32;
    arma::cx_mat A(M,M);
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            double x,y;
            char c1,c2;
            file >> x >> c1 >>y>>c2;
            A(i,j) = complex<double>(x,y);
        }
    }
    file.close();
    for(int i=0;i<2;i++){
        for(int j=0;j<M;j++){
            cout<<A(j,i)<<endl;
        }
    }
    
    cx_mat A_inv = inv(A);

    cout<<"INV:\n";
    for(int i=0;i<2;i++){
        for(int j=0;j<M;j++){
            cout<<A_inv(j,i)<<endl;
        }
    }

    cout <<"mul:\n"<<A*A_inv<<endl;
    


    return 0;


}

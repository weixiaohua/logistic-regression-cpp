#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <map>
#include "data.h"

using std::vector;
using std::pair;
using std::make_pair;

double sigmod(double wx){
    return 1/(1+exp(-wx));
}

double sigmod_wx(vector<double> &w,vector<pair<int,double> > &x){
    double weight;
    for(int i=1;i!=x.size();++i)
        weight += w[x[i].first]*x[i].second;
    std::cout<<"wx:"<<weight+w[0]<<"\tw0:"<<w[0]<<std::endl;
    //return sigmod(weight+w[0]);
    return sigmod(weight);
}
double logloss(int y,vector<double> &w,vector<pair<int,double> > &x){
    double loss;
    double fx = sigmod_wx(w,x);
    loss = y*fx+(1-y)*(1-fx);
    return loss;
}

void printP(pair<int,double> t){
    std::cout<<t.first<<":"<<t.second<<" ";
}
double printVecP(std::vector<pair<int,double> > &vec){
    for_each(vec.begin(),vec.end(),printP);
    std::cout<<std::endl;
}
//template<class T>
void print(double t){
    std::cout<<t<<" ";
}
double printVec(std::vector<double> &vec){
    for_each(vec.begin(),vec.end(),print);
    std::cout<<std::endl;
}

void predict(Data &data ,vector<double> &weight){
    int size = data.rsize;
    double max = 0;
    for(int i = 0 ;i!=size ;++i){
        Item it = data.items[i];
        double pre = sigmod_wx(weight ,it.xi);
        int label = pre < 0.5?0:1;
        std::cout<<"label:"<<label<<"\tpre:"<<pre<<"\ty:"<<it.y<<std::endl;
        if(it.y==label) max++;
    }
    std::cout<<"Precision is "<<max/size<<std::endl;
}
double sgd_updator(Data &data,vector<double> &weight,double lamda,double eps){
    int i = 0;
    int time = 0;
    double oldloss=10.0;
    while(1){
        //std::cout<<"--------------------time"<<time<<"-----------------------"<<std::endl;
        Item it = data.items[i++];
        vector<pair<int,double> > fea = it.xi;
        //vector<pair<int,double> >* f = it.xi;
        int y = it.y==-1?0:it.y;
        //int y = it.y;
        double hx = sigmod_wx(weight,fea);
        double loss_01 = y-hx;
        double gredent=0;
        for(int j=0;j!= fea.size();j++){
            weight[fea[j].first] += lamda*loss_01*fea[j].second;
            gredent += loss_01*fea[j].second;
            std::cout<<loss_01*fea[j].second<<" ";
        }
        std::cout<<std::endl;
        weight[0] += lamda*loss_01;
        //printVec(weight);
        double loss = logloss(y,weight,fea);
        //std::cout<<"logloss is "<<loss<<std::endl;
        gredent = fabs(gredent/fea.size());
        if(gredent<eps){
            std::cout<<"convergence! oldloss is "<<oldloss<<" and logloss is "<<loss<<"mins is "<<oldloss-loss<<std::endl;    
            break;
        }else   oldloss = loss;
        if(i==data.rsize-1)  break;
        //if(time++>220)   break;
        time++;
    }
}
double sgd_l2_updator(Data &data,vector<double> &weight,double lamda,double eps,double l2){
    int i = 0;
    int time = 0;
    double oldloss=10.0;
    double penalty = 1-l2;
    while(1){
        std::cout<<"--------------------time"<<time<<"-----------------------"<<std::endl;
        Item it = data.items[i++];
        vector<pair<int,double> > fea = it.xi;
        //vector<pair<int,double> >* f = it.xi;
        int y = it.y==-1?0:it.y;
        //int y = it.y;
        double hx = sigmod_wx(weight,fea);
        double loss_01 = y-hx;
        double gredent=0;
        for(int j=0;j!= fea.size();j++){
            weight[fea[j].first] += lamda*loss_01*fea[j].second*penalty;
            std::cout<<loss_01*fea[j].second*penalty<<" ";
            gredent += loss_01*fea[j].second*penalty;
        }
        std::cout<<std::endl;
        weight[0] += lamda*loss_01;
        //printVec(weight);
        double loss = logloss(y,weight,fea);
        gredent = fabs(gredent/fea.size());
        //std::cout<<"logloss is "<<loss<<"\tgredent:"<<gredent<<std::endl;
        if(gredent<eps){
            std::cout<<"convergence! oldloss is "<<oldloss<<" and logloss is "<<loss<<"mins is "<<oldloss-loss<<std::endl;    
            break;
        }else   oldloss = loss;
        if(i==data.rsize-1)  break;
        //if(time++>220)   break;
        time++;
    }
}
vector<pair<int,double> > vec_mins(vector<pair<int,double> >& xk,vector<pair<int,double> >& xk1){
    vector<pair<int,double> > sk;
    vector<pair<int,double> >::iterator it1 = xk.begin();
    vector<pair<int,double> >::iterator it2 = xk1.begin();
    while(it1!=xk.end() && it2!=xk1.end()){
        if(it1->first == it2->first){
            sk.push_back(make_pair(it1->first,it1->second - it2->second));
            ++it1;++it2;
        }
        else if (it1->first > it2->first){
            sk.push_back(make_pair(it2->first,it2->second));
            ++it2;
        }
        else if (it1->first < it2->first){
            sk.push_back(make_pair(it1->first,it1->second));
            ++it1;
        }
    }
    while(it1!=xk.end()){
        sk.push_back(make_pair(it1->first,it1->second));
        ++it1;
    }
    while(it2!=xk1.end()){
        sk.push_back(make_pair(it2->first,it2->second));
        ++it2;
    }
    return sk;
}
//lbfgs
double vec_plus(vector<pair<int,double> >& xk,vector<pair<int,double> >& xk1){
    vector<pair<int,double> >::iterator it1 = xk.begin();
    vector<pair<int,double> >::iterator it2 = xk1.begin();
    double ans = 0;
    while(it1!=xk.end() && it2!=xk1.end()){
        if(it1->first == it2->first){
            ans += it1->second * it2->second;
            ++it1;++it2;
        }else if (it1->first > it2->first)  ++it2;
        else if (it1->first < it2->first) ++it1;
    }
    return ans;
}
vector<pair<int,double> > two_loop(int k ,int m,vector<vector<pair<int,double> > > &sk
        ,vector<vector<pair<int,double> > > &yk, vector<pair<int,double> > &gk){
    int sigma = k>m?k-m:0;
    int L = k>m?m:k;
    std::map<int,double> alf;
    vector<pair<int,double> > qL = gk;
    for(int i=L-1;i!=-1;--i){
        int j = i+sigma;
        double ai = 1/vec_plus(yk[i],sk[i]) * vec_plus(sk[i],qL);
        alf[i] = ai;
        //ql = qi+1 - ai * yk[j];
        vector<pair<int,double> > yk_temp(yk[i]);
        for(int n=0;n!=yk[i].size();++n)
            yk_temp[n].second *= ai;
        qL = vec_mins(qL , yk_temp);
        //std::cout<<"b_qL test: i="<<i<<std::endl;
        //printVecP(qL);
    }
    //r = q0;
    for(int i=0;i!=L;++i){
        int j = i+sigma;
        double betai = 1/vec_plus(yk[i],sk[i]) * vec_plus(yk[i],qL);
        vector<pair<int,double> > sk_temp(sk[i]);
        //ri+1 = r + (alf[i] - betai) * sk[j];
        double a_b = alf[i] - betai;
        for(int n=0;n!=sk[i].size();++n)
            sk_temp[n].second *= -a_b;
        qL = vec_mins(qL ,sk_temp);
        //std::cout<<"f_qL test: i="<<i<<std::endl;
        //printVecP(qL);
    }
    return qL;
}
vector<pair<int,double> > two_loop_ori(int k ,int m,vector<vector<pair<int,double> > > &sk
        ,vector<vector<pair<int,double> > > &yk, vector<pair<int,double> > &gk){
    int sigma = k>m?k-m:0;
    int L = k>m?m:k;
    std::map<int,double> alf;
    vector<pair<int,double> > qL = gk;
    for(int i=L-1;i!=-1;--i){
        int j = i+sigma;
        double ai = 1/vec_plus(yk[j],sk[j]) * vec_plus(sk[j],qL);
        //double ai = 1/vec_plus(yk[i],sk[i]) * vec_plus(sk[i],qL);
        alf[i] = ai;
        //ql = qi+1 - ai * yk[j];
        vector<pair<int,double> > yk_temp(yk[j]);
        for(int n=0;n!=yk[j].size();++n)
            yk_temp[n].second *= ai;
        qL = vec_mins(qL , yk_temp);
        std::cout<<"b_qL test: i="<<i<<std::endl;
        printVecP(qL);
    }
    //r = q0;
    for(int i=0;i!=L;++i){
        int j = i+sigma;
        double betai = 1/vec_plus(yk[j],sk[j]) * vec_plus(yk[j],qL);
        vector<pair<int,double> > sk_temp(sk[j]);
        //ri+1 = r + (alf[i] - betai) * sk[j];
        double a_b = alf[i] - betai;
        for(int n=0;n!=sk[j].size();++n)
            sk_temp[n].second *= -a_b;
        qL = vec_mins(qL ,sk_temp);
        std::cout<<"f_qL test: i="<<i<<std::endl;
        printVecP(qL);
    }
    return qL;
}
double lbfgs_updator(Data &data,vector<double> &weight,double lamda,double eps){
    int k = 0;
    int time = 0;
    double oldloss=10.0;
    int bound = 3;//usually small <10
    vector<vector<pair<int,double> > > sks;
    vector<vector<pair<int,double> > > yks; 
    vector<pair<int,double> > gko;
    while(1){
        std::cout<<"--------------------------------"<<k<<"---------------"<<std::endl;
        Item it = data.items[k];       
        int y = it.y==-1?0:it.y;

        double hx = sigmod_wx(weight,it.xi);
        double loss_01 = hx-y;
  
        vector<pair<int,double> > gk;
        for(int j=0;j!= it.xi.size();++j){       
            gk.push_back(pair<int,double>(it.xi[j].first, loss_01 * it.xi[j].second));
        }
        //std::cout<<"loss_01:"<<loss_01<<"\tdebeg gko"<<std::endl;
        printVecP(gko);
        vector<pair<int,double> > deta_w;
        if(k>0){
        //yk = gk-gko
        if(sks.size()>bound && yks.size()>bound){ 
            sks.erase(sks.begin());
            yks.erase(yks.begin());
        }
        //std::cout<<"size sks:"<<sks.size()<<"size yks:"<<yks.size()<<std::endl;
        //push sk,yk;   
        std::cout<<"vec mins xk-xk_1\t";
        vector<pair<int,double> > t_sk = vec_mins(it.xi , data.items[k-1].xi); 
        vector<pair<int,double> > t_yk = vec_mins(gk , gko);
        printVecP(t_sk);
        std::cout<<"vec mins yk-yk_1\t";
        printVecP(t_yk);
        
        sks.push_back( vec_mins(it.xi , data.items[k-1].xi));
        yks.push_back( vec_mins(gk , gko));
        //keep old gk;
        gko = vector<pair<int,double> >(gk);
        
        deta_w = two_loop(k,bound,sks,yks,gko);
        
        }
        else    deta_w = gk; 

        std::cout<<"gk\t";
        printVecP(gk);
        std::cout<<"deta_w\t";
        printVecP(deta_w);

        double gredent = 0;
        for(int j=0;j!= deta_w.size();++j){
            weight[ deta_w[j].first ] -= lamda*deta_w[j].second;
            gredent += fabs(deta_w[j].second);
        }
        weight[0] += lamda*loss_01;
        
        double loss = logloss(y,weight,it.xi);
        gredent = fabs(gredent/deta_w.size());
        std::cout<<"logloss is:"<<loss<<"\t loss_01 is:"<<loss_01<<"\tgredent:"<<gredent<<std::endl;
        if(gredent<0.000001){
            std::cout<<"convergence! and logloss is "<<loss<<std::endl;    
            break;
        }else   oldloss = loss;
           
        //if(++k>220)   break;
        ++k;
        std::cout<<"weigth\t";
        printVec(weight);
    }
}
double trainer(Data &data , double lamda,double eps,double l2){
    std::vector<double> weight;
    for(int i=0;i!=data.wsize+1;i++)
        weight.push_back((rand()%10000)*0.0001f);
    std::cout<<"init weight ..."<<std::endl;
    printVec(weight);
    std::cout<<"begin train ..."<<std::endl;
    
    //sgd_l2_updator(data,weight,lamda,eps,l2);
    lbfgs_updator(data,weight,lamda,eps);
    
    printVec(weight);
    std::cout<<"_---------P_-----------"<<std::endl;
    predict(data,weight);

}
double trainer_ori(Data &data ,Data &test, double lamda,double eps,double l2,int way){
    std::vector<double> weight;
    for(int i=0;i!=data.wsize+1;i++)
        weight.push_back((rand()%10000)*0.0001f);
    std::cout<<"init weight ..."<<std::endl;
    printVec(weight);
    if(way==1){
        std::cout<<"begin train sgd..."<<std::endl;
        sgd_updator(data,weight,lamda,eps);
    }
    else if(way==2){
        std::cout<<"begin train sgd_l2..."<<std::endl;
        sgd_l2_updator(data,weight,lamda,eps,l2);
    }
    else if(way==3){ 
        std::cout<<"begin train lbfgs..."<<std::endl;
        lbfgs_updator(data,weight,lamda,eps);
    }
    
    printVec(weight);
    std::cout<<"_---------P_-----------"<<std::endl;
    predict(test,weight);

}
double train(Data &data , double lamda){
    std::vector<double> weight;
    for(int i=0;i!=data.wsize+1;i++)
        weight.push_back((rand()%10000)*0.0001f);
    std::cout<<"time:0"<<std::endl;
    printVec(weight);
    int i = 0;
    int time = 1;
    double oldloss = 10.0;
    double eps = 0.001;
    while(1){
        //std::cout<<"--------------------time"<<time<<"-----------------------"<<std::endl;
        Item it = data.items[i++];
        vector<pair<int,double> > fea = it.xi;
        //vector<pair<int,double> >* f = it.xi;
        int y = it.y==-1?0:it.y;
        //int y = it.y;
        double hx = sigmod_wx(weight,fea);
        double loss_01 = y-hx;
        double gredent = 0;
        for(int j=0;j!= fea.size();++j){
            weight[fea[j].first] += lamda*loss_01*fea[j].second;
            gredent += fabs(loss_01*fea[j].second);
        }
        
        weight[0] += lamda*loss_01;
        //printVec(weight);
        double loss = logloss(y,weight,fea);
        std::cout<<"logloss is "<<loss<<"\tgredent:"<<gredent/fea.size()<<std::endl;
        if(gredent/fea.size()<eps){//if(fabs(oldloss-loss)<eps){
            std::cout<<"convergence! oldloss is "<<oldloss<<" and logloss is "<<loss<<"mins is "<<oldloss-loss<<std::endl;    
            break;
        }else   oldloss = loss;
        //if(i==data.ins_size-1)  i=0;
        if(time++>220)   break;
    }
    printVec(weight);
    std::cout<<"_---------P_-----------"<<std::endl;
    predict(data,weight);
}
double sgd(){}
double bgd(){}
double lbfgs(){}
int test(){
    vector<pair<int,double> > v1,v2,ans;
    for(int i=0;i!=10;i++){
        v1.push_back(pair<int,double>(i,i+0.5));
        if(i%2==0)
            v2.push_back(pair<int,double>(i,i+0.3));
    }
    printVecP(v1);
    std::cout<<"----"<<std::endl;
    printVecP(v2);
    //ans = vec_mins(v1,v2);
    double s = vec_plus(v1,v2);
    std::cout<<"-------"<<std::endl;
    //printVecP(ans);
    std::cout<<s<<std::endl;

}
int main(int argv,char* args[]){
    //test();
    Data data;
    const char* fnameTrain = "agaricus.txt.train";
    const char* fnameTest = "agaricus.txt.test";
    Data test;
    double eps = 0.00001;
    double l2 = 0.01;
    data.loadSVMData(fnameTrain); 
    test.loadSVMData(fnameTest);

    double lamda = atof(args[1]);
    int way = atoi(args[2]);
    //train(data,lamda);
    trainer_ori(data,test,lamda,eps,l2,way);
    
}

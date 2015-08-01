#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>

struct iv_pair{
    int index;
    double value;
    iv_pair(int i,float v){index=i;value=v;}
};
struct Item{
    int y;
    std::vector<iv_pair> values;
};
struct SVMData{
    int wgt_size;
    int ins_size;
    std::vector<Item> instance;
};
void loadSVMData(const char* fname,std::vector<Item> Instances){
    std::cout<<"loadSVMData start fname is"<<fname<<std::endl;
    std::fstream fs(fname,std::fstream::in);
    std::string line,word;
    while(!fs.eof()){
        getline(fs,line);
        if(line.empty()) continue;
        std::stringstream ss(line);
        Item it;
        while(ss>>word){
            std::string::size_type pox = word.find(":");
            if(pox==std::string::npos)  it.y = atoi(word.c_str());
            else    it.values.push_back(iv_pair(atoi(word.substr(0,pox).c_str()),atof(word.substr(pox+1).c_str())));
        }        
        Instances.push_back(it);
    }
    fs.close();
    std::cout<<"load end size is "<<Instances.size()<<std::endl;
}

void loadSVMData(const char* fname,SVMData& data){
    std::cout<<"loadSVMData start fname is"<<fname<<std::endl;
    std::fstream fs(fname,std::fstream::in);
    std::string line,word;
    int max_wgt_size=0;
    while(!fs.eof()){
        getline(fs,line);
        if(line.empty()) continue;
        std::stringstream ss(line);
        Item it;
        while(ss>>word){
            std::string::size_type pox = word.find(":");
            if(pox==std::string::npos)  it.y = atoi(word.c_str());
            else{
                int index = atoi(word.substr(0,pox).c_str());
                it.values.push_back(iv_pair(index, atof(word.substr(pox+1).c_str())));
                if(index>max_wgt_size) max_wgt_size = index;
            }
        }        
        data.instance.push_back(it);
    }
    fs.close();
    data.wgt_size = max_wgt_size;
    data.ins_size = data.instance.size();
    std::cout<<"load end! total data is "<<data.ins_size<<"  and weight size is "<<data.wgt_size<<std::endl;
}

double sigmod(double wx){
    return 1/(1+exp(-wx));
}

double sigmod_wx(std::vector<double> &w,std::vector<iv_pair> &x){
    double weight;
    for(int i=1;i!=x.size();i++)
        weight += w[x[i].index]*x[i].value;
    return sigmod(weight+w[0]);
    return sigmod(weight);
}
double logloss(int y,std::vector<double> &w,std::vector<iv_pair> &x){
    double loss;
    double fx = sigmod_wx(w,x);
    loss = y*fx+(1-y)*(1-fx);
    return loss;
}

//template<typename T>
void print(double t){
    std::cout<<t<<" ";
}
double printVec(std::vector<double> &vec){
    for_each(vec.begin(),vec.end(),print);
    std::cout<<std::endl;
}
void predict(SVMData &data  ,std::vector<double> &weight){
    int size = data.ins_size;
    double max = 0;
    for(int i = size-1 ;i!=size -50 ;i--){
        Item it = data.instance[i];
        double pre = sigmod_wx(weight ,it.values);
        int label = pre < 0.5?-1:1;
        if(it.y==label) max++;
    }
    std::cout<<"Precision is "<<max/50<<std::endl;
}
double train(SVMData &data , double lamda){
    std::vector<double> weight;
    for(int i=0;i!=data.wgt_size+1;i++)
        weight.push_back((rand()%10000)*0.0001f);
    std::cout<<"time:0"<<std::endl;
    printVec(weight);
    int i = 0;
    int time = 1;
    double oldloss = 10.0;
    double eps = 0.001;
    while(1){
        std::cout<<"--------------------time"<<time<<"-----------------------"<<std::endl;
        Item it = data.instance[i++];
        std::vector<iv_pair> fea = it.values;
        int y = it.y;
        double hx = sigmod_wx(weight,fea);
        double loss_01 = y-hx;
        for(int j=0;j!=fea.size();j++){
            weight[fea[j].index] += lamda*loss_01*fea[j].value;
        }
        weight[0] += lamda*loss_01;
        printVec(weight);
        double loss = logloss(y,weight,fea);
        std::cout<<"logloss is "<<loss<<std::endl;
        if(fabs(oldloss-loss)<eps){
            std::cout<<"convergence! oldloss is "<<oldloss<<" and logloss is "<<loss<<"mins is "<<oldloss-loss<<std::endl;    
            break;
        }else   oldloss = loss;
        if(i==data.ins_size-1)  i=0;
        if(time++>220)   break;
    }
    printVec(weight);
    std::cout<<"_---------P_-----------"<<std::endl;
    predict(data,weight);
}

double sgd(){}
double bgd(){}
double lbfgs(){}

int main(int argv,char* args[]){
    SVMData data;
    const char* fname = "heart_scale";
    loadSVMData(fname,data);
    double lamda = 0.01;
    train(data,lamda);
}

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>

using std::pair;
using std::vector;
//using std::string;
struct Item{int y;vector<pair<int,double> > xi;};

class Data{
    public:
        //int ysize;
        int rsize;
        int wsize;
        vector<Item> items;
        Data(){}
        ~Data(){};
        void loadSVMData(const char* fname);
};
void Data::loadSVMData(const char* fname){
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
                pair<int,double> kv(index,atof(word.substr(pox+1).c_str()));
                it.xi.push_back(kv);
                if(index>max_wgt_size) max_wgt_size = index;
            }
        }        
        items.push_back(it);
    }
    fs.close();
    wsize = max_wgt_size;
    rsize = items.size();
    std::cout<<"load end! total data is "<<rsize<<"  and weight size is "<<wsize<<std::endl;
}
/*
double sigmod(double wx){
    return 1/(1+exp(-wx));
}

double sigmod_wx(std::vector<double> &w,std::vector<iv_pair> &x){
    double weight;
    for(int i=1;i!=x.size();++i)
        weight += w[x[i].index]*x[i].value;
    return sigmod(weight+w[0]);
    //return sigmod(weight);
}
double logloss(int y,std::vector<double> &w,std::vector<iv_pair> &x){
    double loss;
    double fx = sigmod_wx(w,x);
    loss = y*fx+(1-y)*(1-fx);
    return loss;
}
*/

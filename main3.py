import numpy as np
import pandas as pd
import math
import numpy as np
import pandas as pd
import math
from graphviz import Digraph
def draw_tree(tree, edges):
    dot = Digraph()
    for key,value in tree.items():
        if value[0]=="leaf":
            label=f"{key}\nLeaf: {value[1]}"
        elif len(value)==2:
            label=f"{key}\n{value[1]}"
        else:
            label=f"{key}\n{value[1]} <= {value[2]}"
        if value[0]=="leaf":
            dot.node(str(key),label,shape="circle",style="filled")
        else:
            dot.node(str(key),label,shape="box",style="rounded")
    for father,children in edges.items():
        for edge_label,child in children:
            dot.edge(str(father),str(child),label=edge_label)
    return dot
def Data_Read(str)->list:
    Data=pd.read_csv(str)
    return Data
class Decision_Tree:
    def __init__(self,Data,label):
        self.Data=Data
        self.Feature_Data=self.stat_feature()
        self.label=label
        self.dfs_clock=0
        self.Edge={}
        self.root=1
        self.Decision_Tree={}
    def is_digital(self,s)->bool:
        try:
            float(s)
            return True
        except ValueError:
            return False
    def check(self,D,feature)->bool:
        col=self.Data.columns.get_loc(feature)
        for i in D:
            if not self.is_digital(self.Data.iloc[i-1,col]):
                return False
        return True
    def stat_feature(self)->dict:
        Feature_Data1={}
        Features=self.Data.columns.tolist()[1:-1]
        items=list(range(1,len(self.Data)+1))
        for feature_i in Features:
            if self.check(items,feature_i):continue
            Feature_Data1[feature_i]=[]
            for feature_val in self.Data[feature_i]:
                if not feature_val in Feature_Data1[feature_i]:
                    Feature_Data1[feature_i].append(feature_val)
        return Feature_Data1
    def Ent(self,D)->float:
        mp={}
        for i in D:
            value=self.Data.loc[i-1,self.label]
            if value in mp:
                mp[value]+=1
            else:
                mp[value]=1
        ans=0.
        total=len(D)
        for key,value in mp.items():
            pk=1.0*value/total
            ans-=pk*math.log2(pk)
        return ans
    def make_feature_contains(self,D,feature_i)->dict:
        dict1={}
        col=self.Data.columns.get_loc(feature_i)
        for i in D:
            dict1[i]=self.Data.iloc[i-1,col]
        return dict1
    def continuous_case(self,D,feature_i,feature_contains)->tuple[float,float,bool]:
        to_be_sort=[]
        for key,value in feature_contains.items():
            to_be_sort.append(float(value))
        to_be_sort.sort()
        ans=-0x3f3f3f3f
        point=0
        for i in range(len(to_be_sort)-1):
            t=1.0*(to_be_sort[i]+to_be_sort[i+1])/2
            new_feature_contains=feature_contains.copy()
            for key,value in new_feature_contains.items():
                if float(value)<=t:new_feature_contains[key]="<"+str(t)
                else :new_feature_contains[key]=">"+str(t)
            ans_i,p_i,b_i=self.discrete_case(D,feature_i,new_feature_contains)
            if ans<ans_i:
                ans=ans_i
                point=t
        return ans,point,True    
    def discrete_case(self,D,feature_i,feature_contains)->tuple[float,float,bool]:
        mp={}
        ans=self.Ent(D)
        total=len(D)
        for key,value in feature_contains.items():
            if value in mp:
                mp[value].append(key)
            else:
                mp[value]=[key]
        for key,value in mp.items():
            Dv=len(value)
            t=self.Ent(value)
            ans-=1.0*Dv*t/total 
        return ans,0,False
    def Gain(self,D,feature_i,feature_contains)->tuple[float,float,bool]:
        if self.check(D,feature_i):
            return self.continuous_case(D,feature_i,feature_contains)
        else:
            return self.discrete_case(D,feature_i,feature_contains)
    def check_same(self,D,features)->bool:
        for feature_i in features:
            col=self.Data.columns.get_loc(feature_i)
            mp={}
            for i in D:
                val=self.Data.iloc[i-1,col]
                if val in mp:
                    mp[val]+=1
                else:
                    mp[val]=1
            if len(mp)>1:
                return False
        return True
    def end_case1(self,items,features,father,edge_info):
        self.dfs_clock+=1
        label_val=self.Data.iloc[items[0]-1,self.Data.columns.get_loc(self.label)]
        self.Decision_Tree[self.dfs_clock]=("leaf",label_val)
        if father:
            if father in self.Edge:
                self.Edge[father].append((edge_info,self.dfs_clock))
            else:
                self.Edge[father]=[(edge_info,self.dfs_clock)]
        return
    def end_case2(self,items,features,father,edge_info):
        self.dfs_clock+=1
        col=self.Data.columns.get_loc(self.label)
        mp={}
        for i in items:
            val=self.Data.iloc[i-1,col]
            if not val in mp:
                mp[val]=1
            else: mp[val]+=1
        mx=-1
        val_0=""
        for key,value in mp.items():
            if mx<value:
                mx=value
                val_0=key
        self.Decision_Tree[self.dfs_clock]=("leaf",val_0)
        if father:
            if father in self.Edge:
                self.Edge[father].append((edge_info,self.dfs_clock))
            else:
                self.Edge[father]=[(edge_info,self.dfs_clock)]
        return
    def end_case3(self,items,features,father,edge_info):
        self.end_case2(items,features,father,edge_info)
    def check_label(self,items,features):
        mp={}
        for i in items:
            value=self.Data.loc[i-1,self.label]
            if value in mp:
                mp[value].append(i)
            else:
                mp[value]=[i] 
        if len(mp)==1:return True
        return False
    def processinfo(self,items,feature,point,use_point)->dict:
        branch1={}
        if use_point:
            branch1["<"+str(point)]=[]
            branch1[">"+str(point)]=[]
            col=self.Data.columns.get_loc(feature)
            for i in items:
                val=float(self.Data.iloc[i-1,col])
                if val<point:
                    branch1["<"+str(point)].append(i)
                else:
                    branch1[">"+str(point)].append(i)
            return branch1
        for feature_i in self.Feature_Data[feature]:
            branch1[feature_i]=[]
        col=self.Data.columns.get_loc(feature)
        for i in items:
            branch1[self.Data.iloc[i-1,col]].append(i)
        return branch1
    def build(self,items,features,father,edge_info):
        if len(features)==0 or self.check_same(items,features):
            self.end_case2(items,features,father,edge_info)
            return
        if self.check_label(items,features):
            self.end_case1(items,features,father,edge_info)
            return 
        self.dfs_clock+=1
        mp={}
        point=0x3f3f3f3f
        mx=-point
        feature_0=""
        use_point=False
        for feature_i in features:
            gain,point_i,bb=self.Gain(items,feature_i,self.make_feature_contains(items,feature_i))
            if mx<gain:
                mx=gain
                feature_0=feature_i
                point=point_i
                use_point=bb
        if use_point:
            self.Decision_Tree[self.dfs_clock]=("branch",feature_0,point)
        else:
            self.Decision_Tree[self.dfs_clock]=("branch",feature_0)
        braches=self.processinfo(items,feature_0,point,use_point)
        cur=self.dfs_clock
        if father:
            if father in self.Edge:
                self.Edge[father].append((edge_info,cur))
            else:
                self.Edge[father]=[(edge_info,cur)]
        for key,value in braches.items():
            new_features=features.copy()
            new_features.remove(feature_0)
            if len(value)==0:
                self.end_case3(items,features,cur,key)
            else:
                self.build(value,new_features,cur,key)
    def predict(self,features)->bool:
        cur=self.root
        while(self.Decision_Tree[cur][0]=="branch"):
            Q_val=features[self.Decision_Tree[cur][1]]
            if len(self.Decision_Tree[cur])==3:
                threshold=self.Decision_Tree[cur][2]
                val=float(Q_val)
                if val<threshold:
                    target="<"+str(threshold)
                else:target=">"+str(threshold)
            else:target=Q_val
            for val,next in self.Edge[cur]:
                if val==target:
                    cur=next
                    break
        result=self.Decision_Tree[cur][1]
        return result
    def evaluate(self,Data_name):
        test_data=pd.read_csv(Data_name)
        cnt=0
        for i in range(len(test_data)):
            be_predict=test_data.iloc[i].to_dict()
            if self.predict(be_predict)==be_predict[self.label]:
                cnt+=1
        return 1.0*cnt/len(test_data)
if __name__=="__main__":
    Dataset=Data_Read("train.csv")
    Tree1=Decision_Tree(Dataset,"好瓜")
    Features=Dataset.columns.tolist()[1:-1]
    Tree1.build(list(range(1,len(Dataset)+1)),Features,0,"?")
    """
    for key,value in Tree1.Decision_Tree.items():
        print(key,' ',value)
    for key,value in Tree1.Edge.items():
        print(key,'--',value)
    """
    dot = draw_tree(Tree1.Decision_Tree,Tree1.Edge)
    dot.render("Decision_tree",format="png",view=True)
    """
    """
    print("evaulate result=",Tree1.evaluate("test.csv")*100,"%")
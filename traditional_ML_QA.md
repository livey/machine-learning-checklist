- [ ] GBDT
  * v.s. random forest
* XGboost 
  * self defined loss function  
  * second order approximation
  * regularizer 
  * boosting speed
    * sorting feature values
    * subsampling features and data samples
    * parallel computing
  * deal with sparse data
  * Tunable parameters 
    * maximal tree depth 
    * learning rate  
    * number of trees
    * data or feature subsampling
    * w and T regularizer 
    * imbalance parameter 
* Decision tree [ref](https://zhuanlan.zhihu.com/p/85731206)
  - [ ] how to split a node
    * regression tree 
      * mse + \alpha*|T| (regularize for tree size) 
    * decision tree 
      * miss classification rate
      * Gini index (sum of variance of multinomial distribution )
      * entropy 

  - [ ] how to cut branches 
    * regression tree 
      * merge node according to the mse, construct a sequence of subtree untill there is only one node, then there is a subtree which is optimal 
  * deal with missing value 
    * discard samples with missing value
    * add a feature indicate there are missing values
    * construct surrogate function (make default branching)
* random forest 
  * bootstrap data 
  * randomly select m features 
  * build a tree 
- [ ] k-means and k-means ++
  * con
    * number of clusters is hard to determin 
    * initial points matter (image choosed two very close points as the initial centers) 
- [ ] coding AUC
 
 

* dropout 
  - [ ] improve generalization 
  - [ ] how it is worked - drop some input features with probability p, at inference time, use all but multiply by p for the dropout layers. 
  - [ ] as Bayesian approximation
* Batch Normalization
 - [ ] more stable training
 - [ ] generalization
 - [ ] how it works
   - covariat shift
   - more smooth loss function 
 - [ ] why not use moving average in batch normalization
   - batch differ, will result in large variance when updating the weight
   - in traditional BN, we backpropagate and count the input feature
  

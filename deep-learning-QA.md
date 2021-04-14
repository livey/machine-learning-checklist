* dropout 
  - [ ] improve generalization 
  - [ ] how it is worked - drop some input features with probability p, at inference time, use all but multiply by p for the dropout layers. 
  - [ ] [as Bayesian approximation](https://zhuanlan.zhihu.com/p/82108924)
* Batch Normalization
  - [ ] more stable training
  - [ ] generalization
  - [ ] how it works
    - covariat shift
    - more smooth loss function 
  - [ ] why not use moving average in batch normalization
    - batch differ, will result in large variance when updating the weight
    - in traditional BN, we backpropagate and count the input feature
 * Convolutional kernels 
   - [ ] [types of kernels](https://towardsdatascience.com/types-of-convolution-kernels-simplified-f040cb307c37)
      - 1, 2, 3 D kernels 
      - Transposed convolution
      - Seperable convolution 
      - Dilate convolution
      - Deformable convolution
    - [ ] small kernels v.s. large kernels
      - small -> less parameters
      - stacked small -> similar perception field as large kernels, less parameters while similar performance
 * Residual network 
   - [ ] x->(relu)->(relu) + x
   - [ ] why at least two layers 
   - [ ] how it facilitate deeper networks
 
 * Computer Vision
   * two stage detection
     - [ ] R-CNN
       * Bottleneck 
         * region proposal is slow 
         * repeated cnn features 
         * svm require storage and computation is slow
     - [ ] Fast R-CNN
     - [ ] Faster R-CNN
   * one stage detection
     - [ ] YOLO
     - [ ] SSD
   * segmentation
      

* Activation function
  * principles 
    * monotonic -- stable training 
    * simple & non-parametric 
    * non-linear
    * range (finite range --> stable training, infinite range --> small learning rate) 
    * continuously differentiable 
  * linear, sigmoid (nice interpretation while vanishing gradient), relu (large gradient), leaky-relu (has gradient when it is less than 0), selu (self normalized), tanh (value [-1,1], gradien is better than sigmoid), soft-plus ( log(e^x+1) ), soft-max (probability interpretation and used in cross entropy loss) 
* LSTM 
  * [structure](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
* Dropout 
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
 * Optimizer 
   - [ ] SGD 
   - [ ] AdaGrad
     * should take larger steps on less updated parameters (intuition: imagine sparse input features) 
   - [ ] RMSprop 
     * \beta_2 = 0.99 
     * add decay factor to the sum of the gradients 
   - [ ] Adam 
     * add momentum for the gradient 
     * \beta_1 = 0.9, \beta_2 = 0.999 
   - [ ] AMSGrad 
     * gradient accumulation are clipped 
   * how to choose good optimizer
     * first try Adam 
     * SGD should attempt with good learning rate 
   * first try 0.01 as the default learning rate --> try some grid search, candidates are {1, 0.1, 0.01, 10^(-4), 10^(-5)}
 * Computer Vision
   * two stage detection
     - [ ] R-CNN
       * IoU (intersection over union), RoI(region of interested), NMS(non-max supression), mAP(mean average precision) 
       * use NMS for each class at test time 
       * resion proposal done by selective search 
       * warp region 
       * svm classifier
         * need store the feature vectors 
         * for each class (one vs all), IoU>=0.3 was considered as positive 
       * region regressor
         * (x,y,w,h) normalization and nonlinear transformation 
         * Huber loss 
       * training 
         * =>0.5 IoU as positive, other as negative 
         * each batch contains 32 positive and 92 background 
       * Bottleneck 
         * region proposal is slow 
         * repeated cnn features 
         * svm require storage and computation is slow
     - [ ] Fast R-CNN
       * RoI is independent of the network
       * shared CNN feature map 
       * RoI pooling to get fixed-size input of the DNN(classification & regression) 
       * K+1 (background classes), 4 location values for each of the class 
       * loss function 
         * classification cross-entropy loss
         * if object exist, location loss is added
         * Huber loss for location loss 
       * training
         * flip (image augmentation) 
         * each batch from two images 
         * =>0.5 IoU as positive and used in training
         * sample the rest from (0.1, 0.5] IoU as background image class (partially overlapped patches are harder to classify [hard mining] )
         * non-max suppression is used at inference stage
         * low-rank approximation is used to speed up inference
     - [ ] Faster R-CNN
       * shared CNN feature for region proposal and detection network
       * region proposal network (RPN)
         * sliding window on CNN feature map
         * anchor
           * each is centered at the sliding window with different scale and aspect rations (bonus: multi scale) 
           * each anchor corresponding to positive -- negative 
           * positive =>0.7 IoU or anchor with highest IoU (in case no one match the first condition)
           * negative =< 0.3 (except the positive one)  
           * 4k coordinates(k anchor, each has 4 coordinates) + 2k scores
       * loss function = classification loss (k+1 classes) + regression loss (only positive ones have regression loss) 
       * training
         * alternating between region proposal network and object detection network
       
   * one stage detection
     - [ ] YOLO
       * divide the image into $S \times S$ cells
       * each cell produce $S \times S\times(5B+C))$ tensor
         * 4B location, (x,y,w,h), (x,y)--relative to bounds of the cell, (w, h)--relative to the whole image 
         * B confidence Pr(obj)x IoU 
         * C classes  
       * one box with highest IoU are responsible to prediciton
       * l_2 norm for the location and classification 
       * improve generalization: Dropout (p=0.5), image augmentation (random scaling, xposure and sturation) 
       * each cell only responsible for one class, so small dense objects will miss
       * condidence score used for non-max suppression in inference time
     - [ ] SSD
       * image pyramid 
       * sliding window is fixed (3x3xp), 
       * predicted location is relative to the default box shape
       * (c+1) classes score
       * matching strategy: => 0.5 jaccard overlap (treated as positive samples) 
       * location loss = Huber loss and only account for the positive samples 
       * sort the score and choose larger one as the negative samples (negative : postive = 3:1) 
   * segmentation
      - [ ] Mask R-CNN
        * add segmentation network in additino to faster R-CNN 
        * use RoIAlign to fine tune the region (linear interpolation) 

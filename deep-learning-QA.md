* Generalization 
  * how to improve (reduce overfitting) 
    * more dataset
    * dataset augmentation 
    * label smoothing
    * reduce capacity (reduce depth, width; use bottleneck network. not recommended)
    * regularization and weight decay 
    * dropout     
    * early stopping ( use validation data to monitor the error, if goes up, early stopping) 
    * adversarial learning
    * ensemble learning (bagging--reduce variance, boosting--reduce biase)
    * batch normalization (mini-batch introduce some noise, which increases generalization)
* overfitting (similar to improve generalization)
  
* Data 
  * small size of data 
    * reduce machine capacity(reduce complex and add regularizer)  
    * data augmentation 
    * transfer learning 
    * few shot learning 
  * imbalanced data 
    * oversampling
    * undersampling
    * data augmentation
    * use different weights on the loss of different class
    * use precision/recal, f1 score 
    * anomaly detection
  * few labeled data
    - [ ] [semi-supervised learning](https://yassouali.github.io/ml-blog/deep-semi-supervised/)  
   

* Activation function
  * principles 
    * monotonic -- stable training 
    * simple & non-parametric 
    * non-linear
    * range (finite range --> stable training, infinite range --> small learning rate) 
    * continuously differentiable 
  * linear, sigmoid (nice interpretation while vanishing gradient), relu (large gradient), leaky-relu (has gradient when it is less than 0), selu (self normalized), tanh (value [-1,1], gradien is better than sigmoid), soft-plus ( log(e^x+1) ), soft-max (probability interpretation and used in cross entropy loss) 
  * cos (solving differential equations) 
  * soft-max 
    * v.s. hard max 
      * soft-max can easily achieve one-hot shape (due to the exponential transform) even that feature are not so different
      * as a result of previous item, training is much more efficient  
* LSTM 
  * [structure](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
  * why use tanh? state value should increase or decrease. if sigmoid, only decrease. tanh \in [-1,1], represents increase and decrease. 
* Dropout 
  * each input (neuron) is selected with probability p (20% ~ 50% ) (not the each weight element is randomly selected)
  * improve generalization (ultimate goal [good answer](https://stats.stackexchange.com/questions/241645/how-to-explain-dropout-regularization-in-simple-terms))
    * prevent co-updating
    * ensemble learning
    * Bayesian approximation 
  * how it is worked - drop some input features with probability p, at inference time, use all but multiply by p for the dropout layers. 
  * [as Bayesian approximation](https://zhuanlan.zhihu.com/p/82108924)
  * how it works 
    * much stable training (at each training, pruning out parts weights. small weights are easier to train)
    * prevent co-updating of parameters
* Batch Normalization
  * applied just before activation function 
  * in CNN, each element in the feature map use the same scale and biase parameter (that is why we can do acceleration of BN at inference time, also is why we can do network slimming, batch-size x width x hight elments used to compute the mean and variance) 
  * more stable training
  * generalization
  * how it works
    - covariat shift
    - more smooth loss function 
  * why not use moving average in batch normalization
    - batch differ, will result in large variance when updating the weight
    - in traditional BN, we backpropagate and count the input feature
  * batch as a convolutional layer (expedite inference) 
  * image -> conv -> BN -> Relu ([ref](https://zhuanlan.zhihu.com/p/94138640))
  * network slimming (L_1 norm regularizer on scale factor) 
  * usually used as pre-activation
 
 * word2vector
 
 * CNN 
   - [ ] premise to use CNN (data distribution is even, almost indentical?)
   * why use CNN 
     * parameter sharing (one filter should be universal useful on any part of the image) 
     * sparsity of connection (receptive field is limitted) 
   * size of feature map (w' = floor((W+2p-k)/s + 1)
   * receptive field 
 
   * convolutional kernels  
     * size of feature map (w' = floor((W+2p-k)/s + 1)
     * receptive field 
     * [types of kernels](https://towardsdatascience.com/types-of-convolution-kernels-simplified-f040cb307c37)
        - 1, 2, 3 D kernels 
        - Transposed convolution
        - Seperable convolution 
        - Dilate convolution
        - Deformable convolution
      * small kernels v.s. large kernels
        - small -> less parameters
        - stacked small -> similar perception field as large kernels, less parameters while similar performance
        - computational efficiency 
      - [ ] kernel dimension is an odd number 
    * padding 
      * 'same' -- if with stride 1, will output same dimension as the dimension of input
      * 'valid' -- discard feature no cover the kernel 
    * pooling 
      * why? 
        * reduce feature dimension -> reduce parameters 
        * summarize the feature in a region -> robust to position variation   
        * max-pooling extract sharp features (edge, point) while average-pooling extract smooth feature
      * types (average, max, global) 
        - [ ] soft-pool
      * parameters (size, stride, types) 
 * Residual network 
   * x->(relu)->(relu) + x
   * why at least two layers 
   * how it facilitate deeper networks
* [How to choose the number of hidden layers and nodes in a feedforward neural network?](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)
  * determin input and output dimension (dimension of features and kind of loss)
  * number of hidden neurons should in between the dimension of input and output.
  * experiments
  * intuition
  * go for depth
  * borrow ideas 
  * search (pruning, random, grid, huristic, exhaustive)


 * choose a good initialization
   * too small ones lead to vanishing gradient 
   * large ones lead to exploding 
   * mean of activation should be zero
   * variance of activations should stay the same
   * types of initialization 
     * Xavier ( W ~ N(0, 1/dimension of input), b=0) 
     * Uniform 
     
* loss function  
  * 0-1 loss
  * l1, l2, l_\infty
  * KL divergence
  * cross entropy
  * hinge 
  * Huber loss
  * maximal log likelihood
  * exponential   
 * Training 
   * mini-batch(large batch-- stable gradient--high computation and storage, small unstable training loss) 
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
 * gradient vanishing and explosion
   * gradient clipping and weight decay 
   * batch normalization 
   * resnet, lstm
   * fine tuning
   * weight initialization 
   * choose right activation function
  
 * Computer Vision
   * image classification 
     - [ ] [ten papers related to image classification](https://towardsdatascience.com/10-papers-you-should-read-to-understand-image-classification-in-the-deep-learning-era-4b9d792f45a7)
     - [ ] [inception](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202) 
   * two stage detection
     * R-CNN
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
     *  Fast R-CNN
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
     * Faster R-CNN
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
       * sort the score and choose larger ones as the negative samples (negative : postive = 3:1) 
   * segmentation
      - [ ] Mask R-CNN
        * add segmentation network in additino to faster R-CNN 
        * use RoIAlign to fine tune the region (linear interpolation) 
   - [ ] online hard example mining
   - [ ] Fine-grained image categorization 

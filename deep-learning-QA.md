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
         * >=0.5 IoU as positive, other as negative 
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
         * 
     - [ ] Faster R-CNN
   * one stage detection
     - [ ] YOLO
     - [ ] SSD
   * segmentation
      

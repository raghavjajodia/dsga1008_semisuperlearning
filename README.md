# DSGA1008_SemiSupervisedLearning (Course by Prof Yann LeCun)
Semi-supervised learning competition for course Deep Learning DS-GA 1008 (Spring 2019) at New York University. This course was taught by Prof. Yann LeCun. Below is a post by him about this competition.
https://www.facebook.com/yann.lecun/posts/10155962251157143

Following dataset was given - 
- unlabeled set: 512k images from 1000 classes from ImageNet 22k resized to 96x96 pixels.
- Labeled set 64k images from 1000 different classes from ImageNet 22k (64 images/category) resized to 96x96.

Task is to learn category detector from the given data (No pretrained models/external dataset is allowed)

We solved this problem by using Self-supervised learning, by predicting Image Rotations. Please see the following paper for more details -
https://arxiv.org/abs/1803.07728  
https://github.com/gidariss/FeatureLearningRotNet

Libraries - Pytorch  

Results  
We achieved 61% Top-5 accuracy on validation set. The winning team had around 65% top accuracy. We were amongst top5 teams in the competition.  

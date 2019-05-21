# DSGA1008_SemiSupervisedLearning (Course by Prof Yann LeCun)
Semi-supervised learning competition for course Deep Learning DS-GA 1008 (Spring 2019) at New York University. This course was taught by Prof. Yann LeCun. Below is a post by him about this competition.
https://www.facebook.com/yann.lecun/posts/10155962251157143

## Following dataset was given - 
- unlabeled set: 512k images from 1000 classes from ImageNet 22k resized to 96x96 pixels.
- Labeled set 64k images from 1000 different classes from ImageNet 22k (64 images/category) resized to 96x96.

Task is to learn category detector from the given data (No pretrained models/external dataset is allowed)

We solved this problem by using Self-supervised learning, by predicting Image Rotations. Please see the following paper for more details -
https://arxiv.org/abs/1803.07728  
https://github.com/gidariss/FeatureLearningRotNet

## Libraries - Pytorch  

## Instructions to run the code  
The repo consists code to train following models  
- DCGAN - Standard code in pytorch examples
- Gan_Supervised - Supervised training on GAN discriminator, from the output of DCGAN above.  
- Rotation Learning - Run the RotationLearning.py with appropriate parameters. For example,  
  
python ./dsga1008_semisuperlearning/RotationLearning/RotationLearning.py --verbIter 1000  --dataroot /home/rj1408/dl_proj/unsupervised_split --workers 4 --batchSize 64 --imageSize 224 --lr 0.001 --cuda --ngpu 1 --outf /scratch/rj1408/dl_proj/models/rotLearn --manualSeed 9000  
  
The --dataroot is path to unlabeled data directory. Images should be present in subfolder(s) in this directory. For example {dataroot}/0/img_1.png  

- Rotation FineTuning - Fine tune the learned rotation model above with supervised training dataset. For example  
  
python ./dsga1008_semisuperlearning/RotationLearning/ResnetFineTuning.py --net /scratch/rj1408/dl_proj/models/rotLearn/net_epoch_23.pth  --dataroot /scratch/rj1408/dl_proj/ssl_data_96/supervised --workers 4 --batchSize 64  --imageSize 224 --lr 0.001 --cuda --ngpu 1 --outf '/scratch/rj1408/dl_proj/models/rotLearn/finetune' --trainSamples 64 --manualSeed 9000 --niter 100  
  
 The --dataroot is path to labeled data directory. Images should be present in the folders  
   
 ### train - {dataroot}/train/{class_1}/img.png  
 ### validation - {dataroot}/val/{class_1}/img2.png

## Results  
We achieved 61% Top-5 accuracy on validation set. The winning team had around 65% top accuracy. We were amongst top5 teams in the competition.  

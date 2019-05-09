
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
import argparse
import json
import random
import codecs
from torchvision import transforms
from torchvision import models


# In[2]:


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


# In[8]:


data_dir = '/scratch/sa5154/ssl_data_96'
classes = os.listdir(data_dir + '/unsupervised')
classes = classes[:200]

# In[18]:


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
m1 = np.ones((200, 1, 224, 224)) *mean[0]
m2 = np.ones((200, 1, 224, 224)) *mean[1]
m3 = np.ones((200, 1, 224, 224)) *mean[2]
m = np.concatenate((m1, m2, m3), axis=1)
s1 = np.ones((200, 1, 224, 224)) *std[0]
s2 = np.ones((200, 1, 224, 224)) *std[1]
s3 = np.ones((200, 1, 224, 224)) *std[2]
s = np.concatenate((s1, s2, s3), axis=1)


# In[16]:


def batch_iterator(image_list, batch_size=100, shape=(224, 224)):
    random.shuffle(image_list)
    while len(image_list) != 0:
        batch_keys = image_list[:batch_size]

        images = []
        images_path = []

        for key in batch_keys:
            image = cv2.imread(key)
            image = cv2.resize(image, dsize=shape)

            images.append(image)
            images_path.append(key)

        images = np.array(images)
        images = np.reshape(images, newshape=[-1, 3, 224, 224])
#         images = images.astype(np.float64)
#         images = normalize(images, mean, std)
        images_path = np.array(images_path)
        yield images, images_path

        del image_list[:batch_size]


# In[19]:


def data_sampling(model, maxk):
    sampling_dictionary = {}
    model.eval()
    with torch.no_grad():
        
        for each_class in classes:
            print("class name: ", each_class)
            download_path = data_dir + '/unsupervised/'
            if os.path.isdir(download_path + each_class):
                image_path = download_path + each_class + "/*.JPEG"
                all_image_path = glob.glob(image_path)
                print("image data count: ", len(all_image_path))
                for batch_image, batch_image_path in batch_iterator(all_image_path, 200, (224,224)):
                    batch_image = (batch_image - m)/s
                    batch_image = torch.cuda.FloatTensor(batch_image)
                    output = model(batch_image)
                    softmax_output = F.softmax(output, dim=-1)
                    _, top_p = softmax_output.topk(maxk, 1, True, True)

                    # make sampling dictionary
                    for top in top_p.t():
                        for idx, i in enumerate(top):
                            num = i.data.cpu().numpy()
                            value = float(softmax_output[idx][i].data.cpu().numpy())
                            if str(num) in sampling_dictionary:
                                sampling_dictionary[str(num)].append([batch_image_path[idx], value])
                            else:
                                sampling_dictionary[str(num)] = [[batch_image_path[idx], value]]
            else:
                print("Can't find directory")
    print("Saving.. sampling_dict")
    j = json.dumps(sampling_dictionary)
    with open("sampling_dict1.json", "w") as f:
        f.write(j)


# In[6]:


def select_top_k(k=100):
    sampled_image_dict = {}
    sampled_image_dict["all"] = []
    with codecs.open("./sampling_dict1.json", "r", encoding="utf-8", errors="ignore") as f:
        load_data = json.load(f)

        for key in load_data.keys():
            print("label: ", key)
            all_items = load_data[key]
            all_items.sort(key=lambda x: x[1], reverse=True)
            all_items = np.array(all_items)
            print("each label item count: ", len(all_items))
            if(len(all_items) < k):
                for index in range(0, len(all_items)):
                    sampled_image_dict["all"].append([all_items[index][0], int(key)])
            else:
                for index in range(0, k):
                    sampled_image_dict["all"].append([all_items[index][0], int(key)])

    print("Saving.. selected image json")
    j = json.dumps(sampled_image_dict)
    with open("selected_image.json", "w") as f:
        f.write(j)


# In[13]:


resnet = models.resnet34().to(device)


# In[14]:


resnet.load_state_dict(torch.load('./saved_models/resnet-sgd-lr0.1_further1.pth'))


# In[ ]:


data_sampling(resnet, 10)


# In[7]:


select_top_k(100)



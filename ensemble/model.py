import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Architecture1
        model_ft = models.resnet34(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 1000)
        
        # Architecture2
        model_rotation = models.resnet34(pretrained=False)
        num_ftrs2 = model_rotation.fc.in_features
        model_rotation.fc = nn.Linear(num_ftrs2, 1000)

        #netD.load_state_dict(torch.load(opt.netCont, map_location=device))

        self.resnet_supervised = model_ft
        self.model_rotation = model_rotation

        # Load pre-trained model
        self.load_weights('/scratch/rj1408/dl_proj/models/ensemble/resnet-sgd-lr0.1_further3.pth','/scratch/rj1408/dl_proj/models/rotLearn/finetune5/net_epoch_13.pth')

    def load_weights(self, pretrained_model_path, pretrained_model_path2, cuda=True):
        # Load pretrained model
        pretrained_model = torch.load(f=pretrained_model_path, map_location="cuda" if cuda else "cpu")
        pretrained_model2 = torch.load(f=pretrained_model_path2, map_location="cuda" if cuda else "cpu")

        # Load pre-trained weights in current model
        with torch.no_grad():
            self.resnet_supervised.load_state_dict(pretrained_model, strict=True)
            self.model_rotation.load_state_dict(pretrained_model2, strict=True)
            
        # Debug loading
        print('Parameters found in pretrained model:')
        pretrained_layers = pretrained_model.keys()
        for l in pretrained_layers:
            print('\t' + l)
        print('')
        # Debug loading
        print('Parameters found in pretrained model2:')
        pretrained_layers2 = pretrained_model2.keys()
        for l in pretrained_layers2:
            print('\t' + l)
        print('')

        # for name, module in self.state_dict().items():
        #     if name in pretrained_layers:
        #         assert torch.equal(pretrained_model[name].cpu(), module.cpu())
        #         print('{} have been loaded correctly in current model.'.format(name))
        #     else:
        #         raise ValueError("state_dict() keys do not match")

    def forward(self, x):
        # TODO
        return 0.5* (F.softmax(self.resnet_supervised(x), dim=-1) + F.softmax(self.model_rotation(x), dim=-1))

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
import time
import copy

def load_unsuper_data(data_dir, batch_size):
    """ Method returning a data loader for labeled data """
    # TODO: add data transformations if needed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )
    data = datasets.ImageFolder(f'{data_dir}/unsupervised/', transform=transform)
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return data_loader

def load_super_data(data_dir, batch_size, split):
    """ Method returning a data loader for labeled data """
    # TODO: add data transformations if needed
    if split == "train":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        )

    data = datasets.ImageFolder(f'{data_dir}/supervised/{split}', transform=transform)
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return data_loader

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            #state_size (ndf*8) x 6 x 6
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            #state_size (ndf*4) x 12 x 12
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            #state_size (ndf*8) x 24 x 24
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),
            #state_size (ndf*8) x 48 x 48
            nn.ConvTranspose2d(ngf * 1,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            #state_size (ndf*8) x 3 x 3
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.get_features = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            # state size. (ndf) x 48 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            # state size. (ndf*2) x 24 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            # state size. (ndf*4) x 12 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            # state size. (ndf*8) x 6 x 6,
            nn.Conv2d(ndf * 8, nfeat, 6, 1, 0, bias=False),
            Flatten()
        )
        
        #Feature size of 64 | If adjusting, change
        self.get_logits = nn.Sequential(nn.Linear(nfeat, 1000))

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            features = nn.parallel.data_parallel(self.get_features, input, range(self.ngpu))
            logits = nn.parallel.data_parallel(self.get_logits, features, range(self.ngpu))
        else:
            features = self.get_features(input)
            logits = self.get_logits(features)
    
        #return features.view(-1, 1).squeeze(1), logits.view(-1, 1).squeeze(1)
        return features, logits

def get_unsupervised_dataiter():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )

    data = datasets.ImageFolder(f'{data_dir}/unsupervised/', transform=transform)
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return iter(data_loader)

def train_model(generator, discriminator, criterion, scheduler, num_epochs = 25, log = False):
    netG, netG_optimizer = generator
    netD, netD_optimizer = discriminator
    netG_criterion, netD_criterion_supervised, netD_criterion_unsupervised = criterion
    unsupervised_dataiter = get_unsupervised_dataiter()
    
    since = time.time()

    best_model_wts = copy.deepcopy(netD.state_dict())
    best_acc = 0.0

    if log:
        f = open("improved_gan_log.txt", "w+")

    for epoch in range(num_epochs):
        if log:
            f.write('-' * 10 + '\n')
            f.write(f'Epoch {epoch + 1} of {num_epochs}\n')
            f.write('-' * 10 + '\n')
            f.flush()
        else:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                netG.train()  # Set model to training mode
                netD.train()
            else:
                netG.eval()   # Set model to evaluate mode
                netD.eval()

            netG_loss = 0.0
            netD_loss = 0.0
            n_correct_top_1 = 0
            n_correct_top_k = 0
            n_samples = 0
            
            for supervised_inputs, supervised_labels in dataloaders[phase]:
                # Get unsupervised data from dataiter; If exhausted (unlikely), reload.
                    
                supervised_inputs = supervised_inputs.to(device)
                supervised_labels = supervised_labels.to(device)
                
                n_samples += batch_size
                netD.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):   
                    supervised_features, supervised_logits = netD(supervised_inputs)
                    softmax_output = torch.nn.functional.log_softmax(supervised_logits, 1)
                    
                    if phase == 'train':
                        # Unsupervised data needed only for train
                        try:
                            unsupervised_inputs, _ = unsupervised_dataiter.next()
                        except StopIteration as e:
                            unsupervised_dataiter = get_unsupervised_dataiter()
                            unsupervised_inputs, _ = unsupervised_dataiter.next()
                        
                        unsupervised_inputs = unsupervised_inputs.to(device)
                        #########################################################
                        # (1) netD: Compute loss for unsupervised real data
                        #########################################################
                        # train with real data
                        labels = torch.full((batch_size,), real_label, device=device)
                        real_features, real_logits = netD(unsupervised_inputs)
                        netD_error_unsupervised_real = unsupervised_criterion(torch.sigmoid(torch.logsumexp(real_logits, 1)), labels)
                        netD_error_unsupervised_real.backward()
                        
                        #########################################################
                        # (2) netD: Compute loss for generated fake data
                        #########################################################
                        # train with fake data
                        noise = torch.randn(batch_size, nz, 1, 1, device=device)
                        fake_inputs = netG(noise)
                        labels.fill_(fake_label)
                        fake_features, fake_logits = netD(fake_inputs.detach())
                        netD_error_unsupervised_fake = unsupervised_criterion(torch.sigmoid(torch.logsumexp(fake_logits, 1)), labels)
                        netD_error_unsupervised_fake.backward()
                        
                        #########################################################
                        # (3) netD: Compute loss for supervised real data
                        #########################################################
                        netD_error_supervised = netD_criterion_supervised(softmax_output, supervised_labels)
                        netD_error_supervised.backward()
                        netD_error = netD_error_supervised + netD_error_unsupervised_real + netD_error_unsupervised_fake
    
                        netD_optimizer.step()
                        
                        #########################################################
                        # (4) netG: Compute feature matching loss
                        #########################################################
                        netG.zero_grad()
                        fake_features, _ = netD(fake_inputs)
                        real_features, _ = netD(unsupervised_inputs)
                        netG_error = netG_criterion(fake_features, real_features)
                        netG_error.backward()

                        netG_optimizer.step()    


                # statistics
                netG_loss += netG_error.item()
                netD_loss += netD_error.item()
                
                # Top 1 accuracy
                pred_top_1 = torch.topk(supervised_logits, k=1, dim=1)[1]
                n_correct_top_1 += pred_top_1.eq(supervised_labels.view_as(pred_top_1)).int().sum().item()

                # Top k accuracy
                pred_top_k = torch.topk(supervised_logits, k=top_k, dim=1)[1]
                target_top_k = supervised_labels.view(-1, 1).expand(batch_size, top_k)
                n_correct_top_k += pred_top_k.eq(target_top_k).int().sum().item()
                
                # Log every 100 batches
                if n_samples % (batch_size * 100) == 0:
                    if log:
                        f.write(f"Phase: {phase}, Generator Loss: {netG_loss / n_samples:.4f}, Discriminator Loss: {netD_loss / n_samples:.4f}, " +
                                f"Top 1: {n_correct_top_1 / n_samples:.4f}, Top {top_k}: {n_correct_top_k / n_samples:.4f}\n")
                        f.flush()
                    else:
                        print(f"Phase: {phase}, Generator Loss: {netG_loss / n_samples:.4f}, Discriminator Loss: {netD_loss / n_samples:.4f}, " +
                              f"Top 1: {n_correct_top_1 / n_samples:.4f}, Top {top_k}: {n_correct_top_k / n_samples:.4f}")
                
            epoch_acc = n_correct_top_k / n_samples

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(netD.state_dict())
                torch.save(netG.state_dict(), './saved_models/improvedgan_netG_best.pth')
                torch.save(netD.state_dict(), './saved_models/improvedgan_netD_best.pth')

        #print("=" * 60)
        #print(f"Best Validation Accuracy: {best_acc:.4f}")
        #print("=" * 60)

    time_elapsed = time.time() - since
    if log:
        f.write('='*50 + '\n')
        f.write(f'\nTraining complete in {time_elapsed // 60:.0f} minutes and {time_elapsed % 60:.0f} seconds\n')
        f.write(f'Best Validation Accuracy: {best_acc:.4f}')
        f.write('='*50 + '\n')
        f.flush()
        f.close()

    else:
        print(f'Training complete in {time_elapsed // 60:.0f} minutes and {time_elapsed % 60:.0f} seconds')
        print(f'Best Validation Accuracy: {best_acc:.4f}')

    # load best model weights
    netD.load_state_dict(best_model_wts)
    return netD

data_dir = '/scratch/iav225/dsga1008/project/ssl_data_96'

nz = 100
ngf = 64
ndf = 64
nc = 3

#Feature space size
nfeat = 256

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 10
batch_size = 64
real_label = 1
fake_label = 0
top_k = 5

netG = Generator(1).to(device)
netD = Discriminator(1).to(device)

netD_optimzer = optim.Adam(netD.parameters())
netG_optimizer = optim.Adam(netG.parameters(), lr = 0.0001)

unsupervised_criterion = nn.BCELoss()
supervised_criterion = nn.CrossEntropyLoss()
gen_criterion = nn.MSELoss()

dataloaders = {x: load_super_data(data_dir, batch_size, x) for x in ['train', 'val']}

netD = train_model((netG, netG_optimizer), (netD, netD_optimzer), (gen_criterion, supervised_criterion, unsupervised_criterion), "insert_scheduler", 25, True)

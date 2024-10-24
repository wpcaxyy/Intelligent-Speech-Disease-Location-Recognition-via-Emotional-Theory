from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import numpy as np

from newEncoder import TransformerEncoderLayer as newEncoderlayer

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.path1 = nn.Sequential(
            nn.Conv2d(1, 32, (9, 1), padding='same', stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 9), padding='same', stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        
        self.path3 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding='same', stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        self.CNN = nn.Sequential(
            nn.Conv2d(96, 128, (3, 3), padding='same', stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),
            
            nn.Conv2d(128, 256, (3, 3), padding='same', stride=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2),
            
            nn.Conv2d(256, 512, (1, 1), padding='same', stride=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.3),
        )
        
        self.encoderlayer = nn.TransformerEncoderLayer(512,4,512)

        self.classifier = nn.Sequential(
            nn.Linear(512,4)
        )

    # ModuleList can act as an iterable, or be indexed using ints
    def forward(self, x):
        #x = x.permute(0,2,1)   #(32,40,219)
        x = x.unsqueeze(dim=1)  #(32,1,40,219)

        x1 = self.path1(x)  #(32,32,20,109)
        x2 = self.path2(x)  #(32,32,20,109)
        x3 = self.path3(x)  #(32,32,20,109)

        x = torch.cat([x1,x2,x3],dim=1) #(32,96,20,109)

        x = self.CNN(x) #(32,512,1,1)

        x = x.reshape(x.shape[0],-1)    #(32,512)

        if self.training:
            old = x

            x = x.unsqueeze(dim=1)  #(32,1,512)
     
            x = self.encoderlayer(x)

            x = x.squeeze(dim=1)    #(32,512)

            x = torch.cat([old,x],dim=0) #(64,512)

        score = self.classifier(x)
        return score



class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()

        self.path1 = nn.Sequential(
            nn.Conv2d(1, 32, (9, 1), padding='same', stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 9), padding='same', stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        
        self.path3 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding='same', stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        self.CNN = nn.Sequential(
            nn.Conv2d(96, 128, (3, 3), padding='same', stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),
            
            nn.Conv2d(128, 256, (3, 3), padding='same', stride=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2),
            
            nn.Conv2d(256, 512, (1, 1), padding='same', stride=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.3),
        )
        
        self.encoderlayer = nn.TransformerEncoderLayer(512,4,512)
        # self.encoderlayer = newEncoderlayer(512,4,512)

        self.proj = nn.Linear(4,1)
        
        self.encoderlayer2 = nn.TransformerEncoderLayer(512,4,512)

        self.classifier = nn.Linear(512,4)

    # ModuleList can act as an iterable, or be indexed using ints
    def forward(self, x):
        x = x.unsqueeze(dim=1)  #(32,1,40,219)

        x1 = self.path1(x)  #(32,32,20,109)
        x2 = self.path2(x)  #(32,32,20,109)
        x3 = self.path3(x)  #(32,32,20,109)

        x = torch.cat([x1,x2,x3],dim=1) #(32,96,20,109)

        x = self.CNN(x) #(32,512,1,1)

        x = x.reshape(x.shape[0],-1)    #(32,512)

        if self.training:
            old = x

            x = x.unsqueeze(dim=1)  #(32,1,512)
            memorybank = x
            for i in range(3):
                x = self.encoderlayer(x)
                memorybank = torch.cat([memorybank,x],dim=1)    #(32,4,512)
        
            x = self.encoderlayer2(memorybank.permute(1,0,2))   #(4,32,512)
            x = self.proj(x.permute(1,2,0)) #(32,512,1)
            x = x.squeeze(dim=2)    #(32,512)

            suf = x

            x = torch.cat([old,x],dim=0) #(64,512)

        score = self.classifier(x)
        # return old,suf,score
        return score

class CGModel(nn.Module):
    def __init__(self):
        super(CGModel, self).__init__()

        self.path1 = nn.Sequential(
            nn.Conv2d(1, 32, (9, 1), padding='same', stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 9), padding='same', stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        
        self.path3 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding='same', stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        self.CNN = nn.Sequential(
            nn.Conv2d(96, 128, (3, 3), padding='same', stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),
            
            nn.Conv2d(128, 256, (3, 3), padding='same', stride=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2),
            
            nn.Conv2d(256, 512, (1, 1), padding='same', stride=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.3),
        )
        
        self.encoderlayer = nn.TransformerEncoderLayer(512,4,512)
        # self.encoderlayer = newEncoderlayer(512,4,512)

        self.classifier = nn.Sequential(
            nn.Linear(512,4)
        )

    # ModuleList can act as an iterable, or be indexed using ints
    def forward(self, x):
        #x = x.permute(0,2,1)   #(32,40,219)
        x = x.unsqueeze(dim=1)  #(32,1,40,219)

        x1 = self.path1(x)  #(32,32,20,109)
        x2 = self.path2(x)  #(32,32,20,109)
        x3 = self.path3(x)  #(32,32,20,109)

        x = torch.cat([x1,x2,x3],dim=1) #(32,96,20,109)

        x = self.CNN(x) #(32,512,1,1)

        x = x.reshape(x.shape[0],-1)    #(32,512)

        # pre = x
        x = x.unsqueeze(dim=1)  #(32,1,512)
        x = self.encoderlayer(x)
        x = x.squeeze(dim=1)    #(32,512)
        # suf = x

        score = self.classifier(x)
        # return pre,suf,score

        return score

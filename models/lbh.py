import torch
import torch.nn as nn
import torch.nn.functional as F

from visdom import Visdom
viz = Visdom(server='http://127.0.0.1', port=8097)



class Conv(nn.Module):
    def __init__(self, input_channels = 3, output_channels = 16 ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv = nn.Conv2d(
                        in_channels=self.input_channels,
                        out_channels=self.output_channels,
                        kernel_size=(3,3),
                        padding=1,
                        stride=1
        )

        self.bn = nn.BatchNorm2d(num_features=self.output_channels)

        self.act = nn.ReLU()

        self.model = nn.Sequential(
                        nn.Conv2d(
                        in_channels=self.input_channels,
                        out_channels=self.output_channels,
                        kernel_size=(3,3),
                        padding=1,
                        stride=1
                        ),
                        nn.BatchNorm2d(num_features=self.output_channels),
                        nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        return x
    


if __name__ == "__main__":

    pic = torch.randn(2,3,512,512)

    model = Conv(input_channels=3, output_channels=16)

    print(model)

    result = model(pic)

    print(result.shape)





        


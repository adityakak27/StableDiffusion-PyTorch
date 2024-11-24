import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size = 3, padding = 1), #(batch_size, channel, height, width) -> (batch_size, 128, height, width)

            VAE_ResidualBlock(128, 128), #does not change the size of the image;
            
            VAE_ResidualBlock(128, 128), #another residual block, does not change
            
            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 0), #visualise this convolution and others on https://ezyang.github.io/convolution-visualizer/index.html
            #in the above convolution, dimensions remain the same, except the height and width are both halved. this reduces the size, considerably reducing computations.
            
            VAE_ResidualBlock(128, 256), #dimension changes from (batches, 128, height / 2, width / 2) -> (batches, 256, height / 2, width / 2)
            
            VAE_ResidualBlock(256, 256), #dimension stays from (batches, 128, height / 2, width / 2) -> (batches, 256, height / 2, width / 2)
            
            nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 0), #(batches, 256, height / 2, width / 2) -> (batches, 256, height / 4, width / 4)

            VAE_ResidualBlock(256, 512), #(batches, 256, height / 4, width / 4) -> (batches, 512, height / 4, width / 4)

            VAE_ResidualBlock(512, 512), #(batches, 512, height / 4, width / 4) -> (batches, 512, height / 4, width / 4)

            nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 0), #(batches, 512, height / 4, width / 4) -> (batches, 512, height / 8, width / 8)

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512), #no change up till here in the shape and size of the image

            nn.GroupNorm(32, 512),

            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size = 3, padding = 1),

            nn.Conv2d(8, 8, kernel_size = 3, padding = 0),
        )

    def forward(self, x : torch.Tensor, noise : torch.Tensor) -> torch.Tensor:
        #x : (batch_size, channel(3), height(512), width(512))
        #noise : same size as the output of the encoder : (batch_size, output_channels, height / 8, width / 8)

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(0, 1, 0, 1)
            x = module(x)
        
        mean, log_variance = torch.chunk(x, 2, dim = 1)

        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        std_dev = variance.sqrt()

        x = mean + std_dev * noise

        x *= 0.18215

        return x
            

import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64):
        super(UNetGenerator, self).__init__()
        
        # --- Construct U-Net Structure ---
        # The innermost layer (bottleneck)
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)
        
        # Intermediate layers
        for i in range(num_downs - 5):
            unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block)
        
        # Gradually upsampling
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block)
        
        # The outermost layer
        self.model = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, input):
        return self.model(input)

class UNetSkipConnectionBlock(nn.Module):
    """Defines the U-Net submodule with skip connection.
    X -------------------identity-----------------------> X
      |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False):
        super(UNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        if input_nc is None:
            input_nc = outer_nc
            
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()] # Tanh to scale output to [-1, 1]
            model = down + [submodule] + up
            
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
            
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            
            # This is the "Skip Connection" magic:
            # We wrap the submodule so we can concatenate its output with the input later
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connection
            # Concatenate input (x) with the output of the submodule
            return torch.cat([x, self.model(x)], 1)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64, n_layers=3):
        """
        Input_nc is 6 because we feed it (Silhouette + Color) stacked together.
        The discriminator sees BOTH the question and the answer.
        """
        super(PatchGANDiscriminator, self).__init__()
        
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):  # gradually increase filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)] # Output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

# Helper to initialize weights (Important for GAN stability)
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
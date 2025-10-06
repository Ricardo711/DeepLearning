import math
import numpy as np

################### Upsample1d Class Components ###################################
# batch_size:   N;      type: scalar;   batch size
# in_channels:  C;      type: scalar;   number of input channels
# input_width:  W_in;   type: scalar;   width of input channels
# output_width: W_out;  type: scalar;   width of output channels
# upsampling_factor:    k;  type: scalar;   upsampling factor \in Z^+ 
###################################################################################
# A:    type: Matrix of N x C x W_in;   pre-upsampling values
# Z:    type: Matrix of N x C x W_out;   post-upsampling values
# dLdZ: type: Matrix of N x C x W_out;   gradient of Loss w.r.t. output Z
# dLdA: type: Matrix of N x C x W_in;   gradient of Loss w.r.t. input A
###################################################################################
class Upsample1d:

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        # implement Z
        N, C, Win = A.shape
        Wout = self.upsampling_factor*(Win-1)+1
        Z = np.zeros((N,C,Wout))
        Z[:,:,::self.upsampling_factor]=A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        N, C, Wout = dLdZ.shape
        Win = int((Wout-1)/self.upsampling_factor) +1
        dLdA = dLdZ[:, :, ::self.upsampling_factor].copy()
        return dLdA


################### Downsample1d Class Components ###################################
# downsampling_factor:    k;  type: scalar;   downsampling factor \in Z^+ 
###################################################################################
# A:    type: Matrix of N x C x W_in;   pre-downsampling values
# Z:    type: Matrix of N x C x W_out;   post-downsampling values
# dLdZ: type: Matrix of N x C x W_out;   gradient of Loss w.r.t. output Z
# dLdA: type: Matrix of N x C x W_in;   gradient of Loss w.r.t. input A
###################################################################################
class Downsample1d:

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        self.A_shape = A.shape
        k = self.downsampling_factor
        Z = A[:, :, ::k].copy()
        return Z

    def backward(self, dLdZ):
        N, C, Win = self.A_shape
        k = self.downsampling_factor
        dLdA = np.zeros((N, C, Win))
        dLdA[:, :, ::k] = dLdZ
        return dLdA


################### Upsample2d Class Components ###################################
# upsampling_factor:    k;  type: scalar;   upsampling factor \in Z^+ 
###################################################################################
# A:    type: Matrix of N x C x H_in x W_in;   pre-upsampling values
# Z:    type: Matrix of N x C x H_out x W_out;   post-upsampling values
# dLdZ: type: Matrix of N x C x H_out x W_out;   gradient of Loss w.r.t. output Z
# dLdA: type: Matrix of N x C x H_in x W_in;   gradient of Loss w.r.t. input A
###################################################################################
# 2D upsampling is used for inputs like images where upsampling is performed in 
# both the x and y direction. 
# Please consider the following class structure.
###################################################################################
class Upsample2d:

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        N, C, Hin, Win = A.shape
        Hout = self.upsampling_factor*(Hin-1)+1
        Wout = self.upsampling_factor*(Win-1)+1

        Z = np.zeros((N,C,Hout,Wout))
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, C, Hout, Wout = dLdZ.shape
        Hin = (Hout-1)*self.upsampling_factor +1
        Win = (Wout-1)*self.upsampling_factor +1

        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor].copy()

        return dLdA


################### Downsample2d Class Components ###################################
# downsampling_factor:    k;  type: scalar;   downsampling factor \in Z^+ 
###################################################################################
# A:    type: Matrix of N x C x H_in x W_in;   pre-downsampling values
# Z:    type: Matrix of N x C x H_out x W_out;   post-downsampling values
# dLdZ: type: Matrix of N x C x H_out x W_out;   gradient of Loss w.r.t. output Z
# dLdA: type: Matrix of N x C x H_in x W_in;   gradient of Loss w.r.t. input A
###################################################################################
# In downsampling, the input features are reduced by a factor of k in 
# both x and y dimensions.
###################################################################################
class Downsample2d:

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        N, C, self.Hin, self.Win = A.shape
        Hout = (self.Hin - 1) // self.downsampling_factor + 1
        Wout =  (self.Win - 1) // self.downsampling_factor + 1

        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor].copy()

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, C, Hout, Wout = dLdZ.shape

        dLdA = np.zeros((N, C, self.Hin, self.Win))
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

        return dLdA

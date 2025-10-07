import numpy as np
from resampling import *

################### Class Components #################################################
# kernel size:  K;  type scalar;    kernel size
# stride:           type: scalar;   stride
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x H_in x W_in;     data input 
# Z:    type: Matrix of N x C_in x H_out x W_out;  features after pooling
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_in x H_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x H_in x W_in;     how changes in inputs affect loss
######################################################################################

class MaxPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        N, C, H, W = A.shape
        K = self.kernel
        Hout, Wout = H - K + 1, W - K + 1
        Z = np.empty((N, C, Hout, Wout), dtype=A.dtype)
        for i in range(Hout):
            for j in range(Wout):
                window = A[:, :, i:i+K, j:j+K]
                Z[:, :, i, j] = window.max(axis=(2,3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        N, C, Hout, Wout = dLdZ.shape
        K = self.kernel
        dLdA = np.zeros_like(self.A)
        for i in range(Hout):
            for j in range(Wout):
                window = self.A[:, :, i:i+K, j:j+K]
                max_vals = window.max(axis=(2,3), keepdims=True)
                mask = (window == max_vals).astype(dLdZ.dtype)
                dLdA[:, :, i:i+K, j:j+K] += mask * dLdZ[:, :, i:i+1, j:j+1]
        return dLdA


class MeanPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        N, C, H, W = A.shape
        K = self.kernel
        Hout, Wout = H - K + 1, W - K + 1
        Z = np.empty((N, C, Hout, Wout), dtype=A.dtype)
        for i in range(Hout):
            for j in range(Wout):
                window = A[:, :, i:i+K, j:j+K]
                Z[:, :, i, j] = window.mean(axis=(2,3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        N, C, Hout, Wout = dLdZ.shape
        K = self.kernel
        dLdA = np.zeros_like(self.A)
        scale = 1.0 / (K*K)
        for i in range(Hout):
            for j in range(Wout):
                dLdA[:, :, i:i+K, j:j+K] += dLdZ[:, :, i:i+1, j:j+1] * scale
        return dLdA


class MaxPool2d:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ)
        return dLdA


class MeanPool2d:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MeanPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ)
        return dLdA

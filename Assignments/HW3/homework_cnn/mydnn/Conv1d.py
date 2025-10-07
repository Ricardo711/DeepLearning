# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

################### Conv1d_stride1 and Conv1d Class Components ###################################
# kernel size:  K;      type scalar;        kernel size
# stride:               type: scalar;       equivalent to downsampling factor
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x W_in;    data input for convolution
# Z:    type: Matrix of N x C_out x W_out;  features after conv1d with stride
# ------------------------------------------------------------------------------------
# W:    type: Matrix of C_out x C_in X K;   weight parameters, i.e. kernels
# b:    type: Matrix of C_out x 1;          bias parameters
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x W_in;    how changes in inputs affect loss
# dLdW: type: Matrix of C_out x C_in X K;   gradient of Loss w.r.t. weights
# dLdb: type: Matrix of C_out x 1;          gradient of Loss w.r.t. bias
###################################################################################
class Conv1d_stride1:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        N, Ci, Win = A.shape
        K = self.kernel_size
        Wout = Win - K + 1
        Z = np.zeros((N, self.out_channels, Wout), dtype=A.dtype)

        for n in range(N):
            for co in range(self.out_channels):
                for t in range(Wout):
                    acc = 0.0
                    for ci in range(Ci):
                        for k in range(K):
                            acc += self.W[co, ci, k] * A[n, ci, t + k]
                    Z[n, co, t] = acc + self.b[co]
        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        N, Co, Wout = dLdZ.shape
        Ci = self.in_channels
        K = self.kernel_size

        self.dLdb = dLdZ.sum(axis=(0, 2))
        self.dLdW = np.zeros_like(self.W)
        dLdA = np.zeros_like(self.A)

        for n in range(N):
            for co in range(Co):
                for t in range(Wout):
                    grad = dLdZ[n, co, t]
                    for ci in range(Ci):
                        for k in range(K):
                            self.dLdW[co, ci, k] += grad * self.A[n, ci, t + k]
                            dLdA[n, ci, t + k] += grad * self.W[co, ci, k]

        return dLdA


class Conv1d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Calculate Z
        # Line 1: Pad with zeros
        # Line 2: Conv1d forward
        # Line 3: Downsample1d forward
        if self.pad > 0:
            Ap = np.pad(A, ((0,0),(0,0),(self.pad,self.pad)), mode='constant')
        else:
            Ap = A

        Zs1 = self.conv1d_stride1.forward(Ap)
        Z = self.downsample1d.forward(Zs1)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Calculate dLdA
        # Line 1: Downsample1d backward
        # Line 2: Conv1d backward
        # Line 3: Unpad
        dZ_s1 = self.downsample1d.backward(dLdZ)
        dAp = self.conv1d_stride1.backward(dZ_s1)
        if self.pad > 0:
            dLdA = dAp[:, :, self.pad:-self.pad]
        else:
            dLdA = dAp
        return dLdA
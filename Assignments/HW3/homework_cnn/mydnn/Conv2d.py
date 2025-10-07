import numpy as np
from resampling import *

################### Conv2d_stride1 and Conv2d Class Components ###################################
# kernel size:  K;  type scalar;    kernel size
# stride:           type: scalar;   downsampling factor
# ------------------------------------------------------------------------------------
# A:    type: Matrix of N x C_in x H_in x W_in;     data input for convolution
# Z:    type: Matrix of N x C_out x H_out x W_out;  features after conv2d with stride 1
# ------------------------------------------------------------------------------------
# W:    type: Matrix of C_out x C_in X K X K;   weight parameters, i.e. kernels
# b:    type: Matrix of C_out x 1;              bias parameters
# ------------------------------------------------------------------------------------
# dLdZ: type: Matrix of N x C_out x H_out x W_out;  how changes in outputs affect loss
# dLdA: type: Matrix of N x C_in x H_in x W_in;     how changes in inputs affect loss
# dLdW: type: Matrix of C_out x C_in X K X K;       how changes in weights affect loss
# dLdb: type: Matrix of C_out x 1;                  how changes in bias affect loss
######################################################################################

class Conv2d_stride1:
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
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size)
            )
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        N, Ci, Hin, Win = A.shape
        K = self.kernel_size
        Hout = Hin - K + 1
        Wout = Win - K + 1
        Z = np.zeros((N, self.out_channels, Hout, Wout), dtype=A.dtype)
        for n in range(N):
            for co in range(self.out_channels):
                for i in range(Hout):
                    for j in range(Wout):
                        acc = 0.0
                        for ci in range(Ci):
                            for u in range(K):
                                for v in range(K):
                                    acc += self.W[co, ci, u, v] * A[n, ci, i+u, j+v]
                        Z[n, co, i, j] = acc + self.b[co]
        self.output_height = Hout
        self.output_width = Wout
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        N, Ci, Hin, Win = self.A.shape
        Co = self.out_channels
        K = self.kernel_size
        Hout, Wout = self.output_height, self.output_width
        self.dLdb = dLdZ.sum(axis=(0,2,3))
        self.dLdW = np.zeros_like(self.W)
        dLdA = np.zeros_like(self.A)
        for n in range(N):
            for co in range(Co):
                for i in range(Hout):
                    for j in range(Wout):
                        grad = dLdZ[n, co, i, j]
                        for ci in range(Ci):
                            for u in range(K):
                                for v in range(K):
                                    self.dLdW[co, ci, u, v] += grad * self.A[n, ci, i+u, j+v]
                                    dLdA[n, ci, i+u, j+v] += grad * self.W[co, ci, u, v]
        return dLdA


class Conv2d:
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

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 =  Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        if self.pad > 0:
            Ap = np.pad(A, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), mode='constant')
        else:
            Ap = A
        Zs1 = self.conv2d_stride1.forward(Ap)
        Z = self.downsample2d.forward(Zs1)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dZs1 = self.downsample2d.backward(dLdZ)
        dAp = self.conv2d_stride1.backward(dZs1)
        if self.pad > 0:
            dLdA = dAp[:, :, self.pad:-self.pad, self.pad:-self.pad]
        else:
            dLdA = dAp
        return dLdA

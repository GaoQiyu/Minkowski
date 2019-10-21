# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Function

from SparseTensor import SparseTensor
from MinkowskiPooling import MinkowskiGlobalPooling
from MinkowskiBroadcast import MinkowskiBroadcastAddition, MinkowskiBroadcastMultiplication, OperationType, operation_type_to_int
import MinkowskiEngineBackend as MEB
from MinkowskiCoords import CoordsKey
from Common import get_postfix


class MinkowskiBatchNorm(Module):
    r"""A batch normalization layer for a sparse tensor.

    See the pytorch :attr:`torch.nn.BatchNorm1d` for more details.
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(MinkowskiBatchNorm, self).__init__()
        self.bn = torch.nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats)

    def forward(self, input):
        output = self.bn(input.F)
        return SparseTensor(
            output,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)

    def __repr__(self):
        s = '({}, eps={}, momentum={}, affine={}, track_running_stats={})'.format(
            self.bn.num_features, self.bn.eps, self.bn.momentum, self.bn.affine,
            self.bn.track_running_stats)
        return self.__class__.__name__ + s


class MinkowskiInstanceNormFunction(Function):

    @staticmethod
    def forward(ctx,
                in_feat,
                in_coords_key=None,
                glob_coords_key=None,
                coords_manager=None):
        if glob_coords_key is None:
            glob_coords_key = CoordsKey(in_coords_key.D)

        gpool_forward = getattr(MEB,
                                'GlobalPoolingForward' + get_postfix(in_feat))
        broadcast_forward = getattr(MEB,
                                    'BroadcastForward' + get_postfix(in_feat))
        add = operation_type_to_int(OperationType.ADDITION)
        multiply = operation_type_to_int(OperationType.MULTIPLICATION)

        mean = in_feat.new()
        num_nonzero = in_feat.new()

        cpp_in_coords_key = in_coords_key.CPPCoordsKey
        cpp_glob_coords_key = glob_coords_key.CPPCoordsKey
        cpp_coords_manager = coords_manager.CPPCoordsManager

        gpool_forward(in_feat, mean, num_nonzero, cpp_in_coords_key,
                      cpp_glob_coords_key, cpp_coords_manager, True)
        # X - \mu
        centered_feat = in_feat.new()
        broadcast_forward(in_feat, -mean, centered_feat, add, cpp_in_coords_key,
                          cpp_glob_coords_key, cpp_coords_manager)

        # Variance = 1/N \sum (X - \mu) ** 2
        variance = in_feat.new()
        gpool_forward(centered_feat**2, variance, num_nonzero,
                      cpp_in_coords_key, cpp_glob_coords_key,
                      cpp_coords_manager, True)

        # norm_feat = (X - \mu) / \sigma
        inv_std = 1 / (variance + 1e-8).sqrt()
        norm_feat = in_feat.new()
        broadcast_forward(centered_feat, inv_std, norm_feat, multiply,
                          cpp_in_coords_key, cpp_glob_coords_key,
                          cpp_coords_manager)

        ctx.in_coords_key, ctx.glob_coords_key = in_coords_key, glob_coords_key
        ctx.coords_manager = coords_manager
        # For GPU tensors, must use save_for_backward.
        ctx.save_for_backward(inv_std, norm_feat)
        return norm_feat

    @staticmethod
    def backward(ctx, out_grad):
        # https://kevinzakka.github.io/2016/09/14/batch_normalization/
        in_coords_key, glob_coords_key = ctx.in_coords_key, ctx.glob_coords_key
        coords_manager = ctx.coords_manager

        # To prevent the memory leakage, compute the norm again
        inv_std, norm_feat = ctx.saved_variables

        gpool_forward = getattr(MEB,
                                'GlobalPoolingForward' + get_postfix(out_grad))
        broadcast_forward = getattr(MEB,
                                    'BroadcastForward' + get_postfix(out_grad))
        add = operation_type_to_int(OperationType.ADDITION)
        multiply = operation_type_to_int(OperationType.MULTIPLICATION)

        cpp_in_coords_key = in_coords_key.CPPCoordsKey
        cpp_glob_coords_key = glob_coords_key.CPPCoordsKey
        cpp_coords_manager = coords_manager.CPPCoordsManager

        # 1/N \sum dout
        num_nonzero = out_grad.new()
        mean_dout = out_grad.new()
        gpool_forward(out_grad, mean_dout, num_nonzero, cpp_in_coords_key,
                      cpp_glob_coords_key, cpp_coords_manager, True)

        # 1/N \sum (dout * out)
        mean_dout_feat = out_grad.new()
        gpool_forward(out_grad * norm_feat, mean_dout_feat, num_nonzero,
                      cpp_in_coords_key, cpp_glob_coords_key,
                      cpp_coords_manager, True)

        # out * 1/N \sum (dout * out)
        feat_mean_dout_feat = out_grad.new()
        broadcast_forward(norm_feat, mean_dout_feat, feat_mean_dout_feat,
                          multiply, cpp_in_coords_key, cpp_glob_coords_key,
                          cpp_coords_manager)

        unnorm_din = out_grad.new()
        broadcast_forward(out_grad - feat_mean_dout_feat, -mean_dout,
                          unnorm_din, add, cpp_in_coords_key,
                          cpp_glob_coords_key, cpp_coords_manager)

        norm_din = out_grad.new()
        broadcast_forward(unnorm_din, inv_std, norm_din, multiply,
                          cpp_in_coords_key, cpp_glob_coords_key,
                          cpp_coords_manager)

        return norm_din, None, None, None, None


class MinkowskiStableInstanceNorm(Module):

    def __init__(self, num_features, dimension=-1):
        Module.__init__(self)
        self.num_features = num_features
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))

        self.mean_in = MinkowskiGlobalPooling(dimension=dimension)
        self.glob_sum = MinkowskiBroadcastAddition(dimension=dimension)
        self.glob_sum2 = MinkowskiBroadcastAddition(dimension=dimension)
        self.glob_mean = MinkowskiGlobalPooling(dimension=dimension)
        self.glob_times = MinkowskiBroadcastMultiplication(dimension=dimension)
        self.dimension = dimension
        self.reset_parameters()

    def __repr__(self):
        s = f'(nchannels={self.num_features}, D={self.dimension})'
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        neg_mean_in = self.mean_in(
            SparseTensor(
                -x.F, coords_key=x.coords_key, coords_manager=x.coords_man))
        centered_in = self.glob_sum(x, neg_mean_in)
        temp = SparseTensor(
            centered_in.F**2,
            coords_key=centered_in.coords_key,
            coords_manager=centered_in.coords_man)
        var_in = self.glob_mean(temp)
        instd_in = SparseTensor(
            1 / (var_in.F + self.eps).sqrt(),
            coords_key=var_in.coords_key,
            coords_manager=var_in.coords_man)

        x = self.glob_times(self.glob_sum2(x, neg_mean_in), instd_in)
        return SparseTensor(
            x.F * self.weight + self.bias,
            coords_key=x.coords_key,
            coords_manager=x.coords_man)


class MinkowskiInstanceNorm(Module):
    r"""A instance normalization layer for a sparse tensor.

    """

    def __init__(self, num_features, dimension=-1):
        r"""
        Args:

            num_features (int): the dimension of the input feautres.

            dimension (int): the spatial dimension of the input tensor.

        """
        Module.__init__(self)
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.dimension = dimension
        self.reset_parameters()
        self.inst_norm = MinkowskiInstanceNormFunction()

    def __repr__(self):
        s = f'(nchannels={self.num_features}, D={self.dimension})'
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        output = self.inst_norm.apply(input.F, input.coords_key, None,
                                      input.coords_man)
        output = output * self.weight + self.bias

        return SparseTensor(
            output,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)

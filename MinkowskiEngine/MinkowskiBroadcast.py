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
from enum import Enum

import torch
from torch.nn import Module
from torch.autograd import Function

import MinkowskiEngineBackend as MEB
from SparseTensor import SparseTensor
from Common import get_postfix


class OperationType(Enum):
    ADDITION = 0
    MULTIPLICATION = 1


op_to_int = {i: i.value for i in OperationType}


def operation_type_to_int(op):
    assert isinstance(op, OperationType)
    return op_to_int[op]


class MinkowskiBroadcastFunction(Function):

    @staticmethod
    def forward(ctx, input_features, input_features_global, operation_type,
                in_coords_key, glob_coords_key, coords_manager):
        assert input_features.shape[1] == input_features_global.shape[1]
        assert input_features.type() == input_features_global.type()
        assert isinstance(operation_type, OperationType)
        if not input_features.is_contiguous():
            input_features = input_features.contiguous()
        if not input_features_global.is_contiguous():
            input_features_global = input_features_global.contiguous()

        ctx.op = operation_type_to_int(operation_type)

        ctx.in_feat = input_features
        ctx.in_feat_glob = input_features_global
        ctx.in_coords_key = in_coords_key
        ctx.glob_coords_key = glob_coords_key
        ctx.coords_manager = coords_manager

        out_feat = input_features.new()

        fw_fn = getattr(MEB, 'BroadcastForward' + get_postfix(input_features))
        fw_fn(ctx.in_feat, ctx.in_feat_glob, out_feat, ctx.op,
              ctx.in_coords_key.CPPCoordsKey, ctx.glob_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()

        grad_in_feat = grad_out_feat.new()
        grad_in_feat_glob = grad_out_feat.new()
        bw_fn = getattr(MEB, 'BroadcastBackward' + get_postfix(grad_out_feat))
        bw_fn(ctx.in_feat, grad_in_feat, ctx.in_feat_glob, grad_in_feat_glob,
              grad_out_feat, ctx.op, ctx.in_coords_key.CPPCoordsKey,
              ctx.glob_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager)
        return grad_in_feat, grad_in_feat_glob, None, None, None, None


class AbstractMinkowskiBroadcast(Module):

    def __init__(self, operation_type, dimension=-1):
        super(AbstractMinkowskiBroadcast, self).__init__()
        assert isinstance(operation_type, OperationType)
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        self.operation_type = operation_type
        self.dimension = dimension

        self.broadcast = MinkowskiBroadcastFunction()

    def forward(self, input, input_glob):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        output = self.broadcast.apply(input.F, input_glob.F,
                                      self.operation_type, input.coords_key,
                                      input_glob.coords_key, input.coords_man)
        return SparseTensor(
            output,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)

    def __repr__(self):
        return self.__class__.__name__


class MinkowskiBroadcastAddition(AbstractMinkowskiBroadcast):
    r"""Broadcast the reduced features to all input coordinates.

    .. math::

        \mathbf{y}_\mathbf{u} = \mathbf{x}_{1, \mathbf{u}} + \mathbf{x}_2
        \; \text{for} \; \mathbf{u} \in \mathcal{C}^\text{in}


    For all input :math:`\mathbf{x}_\mathbf{u}`, add :math:`\mathbf{x}_2`. The
    output coordinates will be the same as the input coordinates
    :math:`\mathcal{C}^\text{in} = \mathcal{C}^\text{out}`.

    .. note::
        The first argument takes a sparse tensor; the second argument takes
        features that are reduced to the origin. This can be typically done with
        the global reduction such as the :attr:`MinkowskiGlobalPooling`.

    """

    def __init__(self, dimension=-1):
        r"""a broadcast addition layer.

        Args:
            :attr:`dimension` (int): the dimension of the space where all the
            inputs and the network is defined. For example, images are in a 2D
            space, meshes and 3D shapes are in a 3D space.

        """
        super(MinkowskiBroadcastAddition,
              self).__init__(OperationType.ADDITION, dimension)


class MinkowskiBroadcastMultiplication(AbstractMinkowskiBroadcast):
    r"""Broadcast reduced features to all input coordinates.

    .. math::

        \mathbf{y}_\mathbf{u} = \mathbf{x}_{1, \mathbf{u}} \times \mathbf{x}_2
        \; \text{for} \; \mathbf{u} \in \mathcal{C}^\text{in}


    For all input :math:`\mathbf{x}_\mathbf{u}`, multiply :math:`\mathbf{x}_2`
    element-wise. The output coordinates will be the same as the input
    coordinates :math:`\mathcal{C}^\text{in} = \mathcal{C}^\text{out}`.

    .. note::
        The first argument takes a sparse tensor; the second argument takes
        features that are reduced to the origin. This can be typically done with
        the global reduction such as the :attr:`MinkowskiGlobalPooling`.

    """

    def __init__(self, dimension=-1):
        r"""a broadcast multiplication layer.

        Args:
            :attr:`dimension` (int): the dimension of the space where all the
            inputs and the network is defined. For example, images are in a 2D
            space, meshes and 3D shapes are in a 3D space.

        """
        super(MinkowskiBroadcastMultiplication,
              self).__init__(OperationType.MULTIPLICATION, dimension)


class MinkowskiBroadcast(Module):
    r"""Broadcast reduced features to all input coordinates.

    .. math::

        \mathbf{y}_\mathbf{u} = \mathbf{x}_2 \; \text{for} \; \mathbf{u} \in
        \mathcal{C}^\text{in}


    For all input :math:`\mathbf{x}_\mathbf{u}`, copy value :math:`\mathbf{x}_2`
    element-wise. The output coordinates will be the same as the input
    coordinates :math:`\mathcal{C}^\text{in} = \mathcal{C}^\text{out}`. The
    first input :math:`\mathbf{x}_1` is only used for defining the output
    coordinates.

    .. note::
        The first argument takes a sparse tensor; the second argument takes
        features that are reduced to the origin. This can be typically done with
        the global reduction such as the :attr:`MinkowskiGlobalPooling`.

    """

    def __init__(self, dimension=-1):
        r"""broadcast layer.

        Args:
            :attr:`dimension` (int): the dimension of the space where all the
            inputs and the network is defined. For example, images are in a 2D
            space, meshes and 3D shapes are in a 3D space.

        """
        super(MinkowskiBroadcast, self).__init__()
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        self.dimension = dimension

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, input, input_glob):
        assert isinstance(input, SparseTensor)
        assert isinstance(input_glob, SparseTensor)
        assert input.D == self.dimension

        coo = input.coords_man.get_coo_broadcast_coords(input.coords_key)
        perm_mat = torch.sparse_coo_tensor(
            coo,
            torch.ones(len(input), dtype=input.dtype, device=input.device),
            requires_grad=False)

        broadcasted_input_glob = perm_mat.mm(input_glob.F)
        return SparseTensor(
            broadcasted_input_glob,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)


class MinkowskiBroadcastConcatenation(MinkowskiBroadcast):
    r"""Broadcast reduced features to all input coordinates and concatenate to the input.

    .. math::

        \mathbf{y}_\mathbf{u} = [\mathbf{x}_{1,\mathbf{u}}, \mathbf{x}_2] \;
        \text{for} \; \mathbf{u} \in \mathcal{C}^\text{in}


    For all input :math:`\mathbf{x}_\mathbf{u}`, concatenate vector
    :math:`\mathbf{x}_2`. :math:`[\cdot, \cdot]` is a concatenation operator.
    The output coordinates will be the same as the input coordinates
    :math:`\mathcal{C}^\text{in} = \mathcal{C}^\text{out}`.

    .. note::
        The first argument takes a sparse tensor; the second argument takes
        features that are reduced to the origin. This can be typically done with
        the global reduction such as the :attr:`MinkowskiGlobalPooling`.

    """

    def forward(self, input, input_glob):
        assert isinstance(input, SparseTensor)
        assert isinstance(input_glob, SparseTensor)
        assert input.D == self.dimension

        coo = input.coords_man.get_coo_broadcast_coords(input.coords_key)
        perm_mat = torch.sparse_coo_tensor(
            coo,
            torch.ones(len(input), dtype=input.dtype, device=input.device),
            requires_grad=False)

        broadcasted_input_glob = perm_mat.mm(input_glob.F)
        broadcast_cat = torch.cat((input.F, broadcasted_input_glob), dim=1)
        return SparseTensor(
            broadcast_cat,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)

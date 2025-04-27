##
# @file   move_boundary.py
# @author Yibo Lin
# @date   Jun 2018
#

import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.move_boundary.move_boundary_cpp as move_boundary_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.move_boundary.move_boundary_cuda as move_boundary_cuda


class MoveBoundaryFunction(Function):
    """ 
    @brief Bound cells into layout boundary, perform in-place update 
    """
    @staticmethod
    def forward(pos, node_size_x, node_size_y, node_size_z, 
                xl, yl, zl, xh, yh, zh,
                num_movable_nodes, num_filler_nodes, num_bins_z):
        if pos.is_cuda:
            func = move_boundary_cuda.forward
        else:
            func = move_boundary_cpp.forward  # 3D modify  num_bins_z 将z轴坐标分层
        output = func(pos.view(pos.numel()), node_size_x, node_size_y, node_size_z, 
                      xl, yl, zl, xh, yh, zh,
                    num_movable_nodes, num_filler_nodes, num_bins_z)
        # print("move boundary output:\n",output)
        return output


class MoveBoundary(object):
    """
    将z轴固定在确定值当中
    @brief Bound cells into layout boundary, perform in-place update 
    """
    def __init__(self, node_size_x, node_size_y, node_size_z,
                 xl, yl, zl, xh, yh, zh,
                 num_movable_nodes, num_filler_nodes, num_bins_z):
        super(MoveBoundary, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.node_size_z = node_size_z
        self.xl = xl
        self.yl = yl
        self.zl = zl
        self.xh = xh
        self.yh = yh
        self.zh = zh
        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes
        self.num_bins_z = num_bins_z

    def forward(self, pos):
        return MoveBoundaryFunction.forward(
            pos,
            node_size_x=self.node_size_x,
            node_size_y=self.node_size_y,
            node_size_z=self.node_size_z,
            xl=self.xl,
            yl=self.yl,
            zl=self.zl,
            xh=self.xh,
            yh=self.yh,
            zh=self.zh,
            num_movable_nodes=self.num_movable_nodes,
            num_filler_nodes=self.num_filler_nodes,
            num_bins_z= self.num_bins_z)

    def __call__(self, pos):
        return self.forward(pos)

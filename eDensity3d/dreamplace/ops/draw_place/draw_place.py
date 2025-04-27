##
# @file   draw_place.py
# @author Yibo Lin
# @date   Jan 2019
# @brief  Plot placement to an image 
#

import os 
import sys 
import torch 
from torch.autograd import Function

import dreamplace.ops.draw_place.draw_place_cpp as draw_place_cpp
import dreamplace.ops.draw_place.PlaceDrawer as PlaceDrawer 

class DrawPlaceFunction(Function):
    @staticmethod
    def forward(
            pos, 
            node_size_x, node_size_y, node_size_z,
            pin_offset_x, pin_offset_y, pin_offset_z,
            pin2node_map, 
            xl, yl, zl, xh, yh, zh,
            site_width, row_height, 
            bin_size_x, bin_size_y, bin_size_z,
            num_movable_nodes, num_filler_nodes, filename,
            flat_net2pin_map,flat_net2pin_start_map,
            draw_mode = "3D"
            ):
        ret = None
        # 屏蔽C++的绘图函数
        # ret = draw_place_cpp.forward(
        #         pos, 
        #         node_size_x, node_size_y, 
        #         pin_offset_x, pin_offset_y, 
        #         pin2node_map, 
        #         xl, yl, xh, yh, 
        #         site_width, row_height, 
        #         bin_size_x, bin_size_y, 
        #         num_movable_nodes, num_filler_nodes, 
        #         filename
        #         )
        
        
        # if C/C++ API failed, try with python implementation 
        if not filename.endswith(".gds") and not ret:
            if draw_mode == "3D":
                ret = PlaceDrawer.PlaceDrawer3D.forward(
                    pos, 
                    node_size_x, node_size_y, node_size_z,
                    pin_offset_x, pin_offset_y, pin_offset_z,
                    pin2node_map, 
                    xl, yl, zl, xh, yh, zh,
                    site_width, row_height, 
                    bin_size_x, bin_size_y, bin_size_z,
                    num_movable_nodes, num_filler_nodes, filename,
                    flat_net2pin_map,flat_net2pin_start_map
                    )

            elif draw_mode == "3DCube":
                ret = PlaceDrawer.PlaceDrawer3DCube.forward(
                        pos, 
                        node_size_x, node_size_y, node_size_z,
                        pin_offset_x, pin_offset_y, pin_offset_z,
                        pin2node_map, 
                        xl, yl, zl, xh, yh, zh,
                        site_width, row_height, 
                        bin_size_x, bin_size_y, bin_size_z,
                        num_movable_nodes, num_filler_nodes, filename,
                        flat_net2pin_map,flat_net2pin_start_map
                        )
        return ret 

class DrawPlaceFunction2D(Function):
    @staticmethod
    def forward(
            pos, 
            node_size_x, node_size_y, 
            pin_offset_x, pin_offset_y, 
            pin2node_map, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            bin_size_x, bin_size_y,
            num_movable_nodes, num_filler_nodes, filename
            ):
        ret = None
        # 屏蔽C++的绘图函数
        ret = draw_place_cpp.forward(
                pos, 
                node_size_x, node_size_y, 
                pin_offset_x, pin_offset_y, 
                pin2node_map, 
                xl, yl, xh, yh, 
                site_width, row_height, 
                bin_size_x, bin_size_y, 
                num_movable_nodes, num_filler_nodes, 
                filename
                )
        
        
        # if C/C++ API failed, try with python implementation 
        if not ret:
           print("plot error!!")
        return ret 

class DrawPlace(object):
    """ 
    @brief Draw placement
    """
    def __init__(self, placedb):
        """
        @brief initialization 
        """
        # 3D modify
        self.node_size_x = torch.from_numpy(placedb.node_size_x)
        self.node_size_y = torch.from_numpy(placedb.node_size_y)
        self.node_size_z = torch.from_numpy(placedb.node_size_z)
        self.pin_offset_x = torch.from_numpy(placedb.pin_offset_x)
        self.pin_offset_y = torch.from_numpy(placedb.pin_offset_y)
        self.pin_offset_z = torch.from_numpy(placedb.pin_offset_z)
        self.pin2node_map = torch.from_numpy(placedb.pin2node_map) # ([ 0, 11,  0,  1, 12,  1,  1,  2, 12,  2,  2,  3,  3])
        self.xl = placedb.xl 
        self.yl = placedb.yl 
        self.zl = placedb.zl 
        self.xh = placedb.xh 
        self.yh = placedb.yh 
        self.zh = placedb.zh 
        self.site_width = placedb.site_width
        self.row_height = placedb.row_height 
        self.bin_size_x = placedb.bin_size_x 
        self.bin_size_y = placedb.bin_size_y
        self.bin_size_z = placedb.bin_size_z
        self.num_movable_nodes = placedb.num_movable_nodes
        self.num_filler_nodes = placedb.num_filler_nodes

        self.flat_net2pin_map = torch.from_numpy(placedb.flat_net2pin_map) #[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
        self.flat_net2pin_start_map = torch.from_numpy(placedb.flat_net2pin_start_map) # [ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

    def forward(self, pos, filename, draw_mode = "3D"): 
        """ 
        @param pos cell locations, array of x locations and then y locations 
        @param filename suffix specifies the format 
        """
        return DrawPlaceFunction.forward(
                pos, 
                self.node_size_x, 
                self.node_size_y, 
                self.node_size_z, 
                self.pin_offset_x, 
                self.pin_offset_y, 
                self.pin_offset_z, 
                self.pin2node_map, 
                self.xl, 
                self.yl, 
                self.zl, 
                self.xh, 
                self.yh, 
                self.zh, 
                self.site_width, 
                self.row_height, 
                self.bin_size_x, 
                self.bin_size_y, 
                self.bin_size_z, 
                self.num_movable_nodes, 
                self.num_filler_nodes, 
                filename, 
                self.flat_net2pin_map,
                self.flat_net2pin_start_map,
                draw_mode
                )

    def __call__(self, pos, filename, draw_mode = "3D"):
        """
        @brief top API 
        @param pos cell locations, array of x locations and then y locations 
        @param filename suffix specifies the format 
        """
        return self.forward(pos, filename, draw_mode)

class DrawPlace2D(object):
    """ 
    @brief Draw placement
    """
    def __init__(self, placedb):
        """
        @brief initialization 
        """
        self.node_size_x = torch.from_numpy(placedb.node_size_x)
        self.node_size_y = torch.from_numpy(placedb.node_size_y)
        self.pin_offset_x = torch.from_numpy(placedb.pin_offset_x)
        self.pin_offset_y = torch.from_numpy(placedb.pin_offset_y)
        self.pin2node_map = torch.from_numpy(placedb.pin2node_map)
        self.xl = placedb.xl 
        self.yl = placedb.yl 
        self.xh = placedb.xh 
        self.yh = placedb.yh 
        self.site_width = placedb.site_width
        self.row_height = placedb.row_height 
        self.bin_size_x = placedb.bin_size_x 
        self.bin_size_y = placedb.bin_size_y
        self.num_movable_nodes = placedb.num_movable_nodes
        self.num_filler_nodes = placedb.num_filler_nodes

    def forward(self, pos, filename, node_size_x, node_size_y, num_movable_nodes, num_filler_nodes): 
        """ 
        @param pos cell locations, array of x locations and then y locations 
        @param filename suffix specifies the format 
        """
        return DrawPlaceFunction2D.forward(
                pos, 
                node_size_x, 
                node_size_y, 
                self.pin_offset_x, 
                self.pin_offset_y, 
                self.pin2node_map, 
                self.xl, 
                self.yl, 
                self.xh, 
                self.yh, 
                self.site_width, 
                self.row_height, 
                self.bin_size_x, 
                self.bin_size_y, 
                num_movable_nodes, 
                num_filler_nodes,  # 后续要改
                filename
                )

    def __call__(self, pos, filename, node_size_x, node_size_y, num_movable_nodes, num_filler_nodes):
        """
        @brief top API 
        @param pos cell locations, array of x locations and then y locations 
        @param filename suffix specifies the format 
        """
        return self.forward(pos, filename, node_size_x, node_size_y, num_movable_nodes, num_filler_nodes)

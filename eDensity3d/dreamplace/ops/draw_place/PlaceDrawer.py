##
# @file   PlaceDrawer.py
# @author Yibo Lin
# @date   Mar 2019
# @brief  A python implementation of placement drawer as an alternative when cairo C/C++ API is not available.
#

import sys
import os
import time
import math
import cairocffi as cairo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import Normalize
from concurrent.futures import ThreadPoolExecutor
# from PlaceDB import z_max

class PlaceDrawer(object):
    """
    @brief A python implementation of placement drawer as an alternative when cairo C/C++ API is not available. 
    """
    @staticmethod
    def forward(pos,
                node_size_x,
                node_size_y,
                node_size_z,
                pin_offset_x,
                pin_offset_y,
                pin_offset_z,
                pin2node_map,
                xl,
                yl,
                zl,
                xh,
                yh,
                zh,
                site_width,
                row_height,
                bin_size_x,
                bin_size_y,
                bin_size_z,
                num_movable_nodes,
                num_filler_nodes,
                filename,
                iteration=None):
        """
        @brief python implementation of placement drawer.  
        @param pos locations of cells 
        @param node_size_x array of cell width 
        @param node_size_y array of cell height 
        @param pin_offset_x pin offset to cell origin 
        @param pin_offset_y pin offset to cell origin 
        @param pin2node_map map pin to cell 
        @param xl left boundary 
        @param yl bottom boundary 
        @param xh right boundary 
        @param yh top boundary 
        @param site_width width of placement site 
        @param row_height height of placement row, equivalent to height of placement site 
        @param bin_size_x bin width 
        @param bin_size_y bin height 
        @param num_movable_nodes number of movable cells 
        @param num_filler_nodes number of filler cells 
        @param filename output filename 
        @param iteration current optimization step 
        """
        num_nodes = len(pos) // 3
        num_movable_nodes = num_movable_nodes
        num_filler_nodes = num_filler_nodes
        num_physical_nodes = num_nodes - num_filler_nodes
        num_bins_x = int(math.ceil((xh - xl) / bin_size_x))
        num_bins_y = int(math.ceil((yh - yl) / bin_size_y))
        num_bins_y = int(math.ceil((zh - zl) / bin_size_z))
        x = np.array(pos[:num_nodes])
        y = np.array(pos[num_nodes:num_nodes*2])
        z = np.array(pos[num_nodes*2:])
        node_size_x = np.array(node_size_x)
        node_size_y = np.array(node_size_y)
        node_size_z = np.array(node_size_z)
        pin_offset_x = np.array(pin_offset_x)
        pin_offset_y = np.array(pin_offset_y)
        pin_offset_z = np.array(pin_offset_z)
        pin2node_map = np.array(pin2node_map)
        try:
            tt = time.time()
            if xh - xl < yh - yl:
                height = 800 
                width = round(height * (xh - xl) / (yh - yl))
            else:
                width = 800 
                height = round(width * (yh - yl) / (xh -xl))
            line_width = 0.1
            padding = 0
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
            ctx = cairo.Context(surface)
            # Do not use scale function.
            # This is not compatible with show_text

            if num_movable_nodes < num_physical_nodes:
                layout_xl = min(
                    np.amin(x[num_movable_nodes:num_physical_nodes]), xl)
                layout_yl = min(
                    np.amin(y[num_movable_nodes:num_physical_nodes]), yl)
                layout_xh = max(
                    np.amax(x[num_movable_nodes:num_physical_nodes] +
                            node_size_x[num_movable_nodes:num_physical_nodes]),
                    xh)
                layout_yh = max(
                    np.amax(y[num_movable_nodes:num_physical_nodes] +
                            node_size_y[num_movable_nodes:num_physical_nodes]),
                    yh)
            else:
                layout_xl = xl
                layout_yl = yl
                layout_xh = xh
                layout_yh = yh

            def bin_xl(id_x):
                """
                @param id_x horizontal index 
                @return bin xl
                """
                return xl + id_x * bin_size_x

            def bin_xh(id_x):
                """
                @param id_x horizontal index 
                @return bin xh
                """
                return min(bin_xl(id_x) + bin_size_x, xh)

            def bin_yl(id_y):
                """
                @param id_y vertical index 
                @return bin yl
                """
                return yl + id_y * bin_size_y

            def bin_yh(id_y):
                """
                @param id_y vertical index 
                @return bin yh
                """
                return min(bin_yl(id_y) + bin_size_y, yh)

            def normalize_x(xx):
                return (xx - (layout_xl - padding * bin_size_x)) / (
                    layout_xh - layout_xl + padding * 2 * bin_size_x) * width

            def normalize_y(xx):
                return (xx - (layout_yl - padding * bin_size_y)) / (
                    layout_yh - layout_yl + padding * 2 * bin_size_y) * height

            def draw_rect(x1, y1, x2, y2):
                ctx.move_to(x1, y1)
                ctx.line_to(x1, y2)
                ctx.line_to(x2, y2)
                ctx.line_to(x2, y1)
                ctx.close_path()
                ctx.stroke()

            # draw layout region
            ctx.set_source_rgb(1, 1, 1)
            draw_layout_xl = normalize_x(layout_xl - padding * bin_size_x)
            draw_layout_yl = normalize_y(layout_yl - padding * bin_size_y)
            draw_layout_xh = normalize_x(layout_xh + padding * bin_size_x)
            draw_layout_yh = normalize_y(layout_yh + padding * bin_size_y)
            ctx.rectangle(draw_layout_xl, draw_layout_yl, draw_layout_xh,
                          draw_layout_yh)
            ctx.fill()
            ctx.set_line_width(line_width)
            ctx.set_source_rgba(0.1, 0.1, 0.1, alpha=0.8)
            ctx.move_to(normalize_x(xl), normalize_y(yl))
            ctx.line_to(normalize_x(xl), normalize_y(yh))
            ctx.line_to(normalize_x(xh), normalize_y(yh))
            ctx.line_to(normalize_x(xh), normalize_y(yl))
            ctx.close_path()
            ctx.stroke()
            ## draw bins
            for i in range(1, num_bins_x):
               ctx.move_to(normalize_x(bin_xl(i)), normalize_y(yl))
               ctx.line_to(normalize_x(bin_xl(i)), normalize_y(yh))
               ctx.close_path()
               ctx.stroke()
            for i in range(1, num_bins_y):
               ctx.move_to(normalize_x(xl), normalize_y(bin_yl(i)))
               ctx.line_to(normalize_x(xh), normalize_y(bin_yl(i)))
               ctx.close_path()
               ctx.stroke()

            # draw cells
            ctx.set_font_size(16)
            ctx.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                                 cairo.FONT_WEIGHT_NORMAL)
            node_xl = x
            node_yl = layout_yl + layout_yh - (y + node_size_y[0:len(y)]
                                               )  # flip y
            node_xh = node_xl + node_size_x[0:len(x)]
            node_yh = layout_yl + layout_yh - y  # flip y
            node_xl = normalize_x(node_xl)
            node_yl = normalize_y(node_yl)
            node_xh = normalize_x(node_xh)
            node_yh = normalize_y(node_yh)
            ctx.set_line_width(line_width)
            #print("plot layout")
            # draw fixed macros
            ctx.set_source_rgba(1, 0, 0, alpha=0.5)
            for i in range(num_movable_nodes, num_physical_nodes):
                ctx.rectangle(node_xl[i], node_yl[i], node_xh[i] - node_xl[i],
                              node_yh[i] -
                              node_yl[i])  # Rectangle(xl, yl, w, h)
                ctx.fill()
            ctx.set_source_rgba(0, 0, 0, alpha=1.0)  # Solid color
            for i in range(num_movable_nodes, num_physical_nodes):
                draw_rect(node_xl[i], node_yl[i], node_xh[i], node_yh[i])
            # draw fillers
            if len(node_xl) > num_physical_nodes:  # filler is included
                ctx.set_line_width(line_width)
                ctx.set_source_rgba(115 / 255.0,
                                    115 / 255.0,
                                    125 / 255.0,
                                    alpha=0.5)  # Solid color
                for i in range(num_physical_nodes, num_nodes):
                    ctx.rectangle(node_xl[i], node_yl[i],
                                  node_xh[i] - node_xl[i], node_yh[i] -
                                  node_yl[i])  # Rectangle(xl, yl, w, h)
                    ctx.fill()
                ctx.set_source_rgba(230 / 255.0,
                                    230 / 255.0,
                                    250 / 255.0,
                                    alpha=0.3)  # Solid color
                for i in range(num_physical_nodes, num_nodes):
                    draw_rect(node_xl[i], node_yl[i], node_xh[i], node_yh[i])
            # draw cells
            ctx.set_line_width(line_width * 2)
            ctx.set_source_rgba(0, 0, 1, alpha=0.5)  # Solid color
            for i in range(num_movable_nodes):
                ctx.rectangle(node_xl[i], node_yl[i], node_xh[i] - node_xl[i],
                              node_yh[i] -
                              node_yl[i])  # Rectangle(xl, yl, w, h)
                ctx.fill()
            ctx.set_source_rgba(0, 0, 0.8, alpha=0.8)  # Solid color
            for i in range(num_movable_nodes):
                draw_rect(node_xl[i], node_yl[i], node_xh[i], node_yh[i])
            ## draw cell indices
            #for i in range(num_nodes):
            #    ctx.move_to((node_xl[i]+node_xh[i])/2, (node_yl[i]+node_yh[i])/2)
            #    ctx.show_text("%d" % (i))

            # show iteration
            if iteration:
                ctx.set_source_rgb(0, 0, 0)
                ctx.set_line_width(line_width * 10)
                ctx.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                                     cairo.FONT_WEIGHT_NORMAL)
                ctx.set_font_size(32)
                ctx.move_to(normalize_x((xl + xh) / 2),
                            normalize_y((yl + yh) / 2))
                ctx.show_text('{:04}'.format(iteration))

            surface.write_to_png(filename)  # Output to PNG
            print("[I] plotting to %s takes %.3f seconds" %
                  (filename, time.time() - tt))
        except Exception as e:
            print("[E] failed to plot")
            print(str(e))
            return 0

        return 1

class PlaceDrawer3D(object):
    """
    3D 绘制函数
    """
    def plot_rect(ax, index, rectangles, num_movable_nodes, num_physical_nodes):
        rect = rectangles[index]
        x = rect['x']
        y = rect['y']
        z = rect['z']
        xx, yy = np.meshgrid(x, y)
        verts = [list(zip(np.ravel(xx), np.ravel(yy), np.zeros_like(np.ravel(xx)) + z))]
        verts = np.array(verts).reshape(len(verts[0]), 3)
        color=(1,0,0)
        if index < num_movable_nodes:
            # 可移动
            color='skyblue'
            # ax.scatter(x, y, z, color=color, marker='o')  # 使用scatter绘制点
        elif index < num_physical_nodes:
            # macro
            color=(1, 0, 0) # 设置为红色
        else:
            # filler
            color=(0.5, 0.5, 0.5) # 设置为灰色
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], color=color, alpha=0.5)

    """绘制点和连接线"""
    def plot_points(ax, points):
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        z = [point[2] for point in points]
        ax.scatter(x, y, z, c='lime', marker='o')

        if len(points) > 2:
            # 计算几何中心
            center = np.mean(points, axis=0)
            ax.scatter(center[0], center[1], center[2], c='cyan', marker='x')  # 绘制几何中心虚拟点

            # 连接点到几何中心
            for point in points:
                ax.plot([point[0], center[0]], [point[1], center[1]], [point[2], center[2]], c='r', linewidth=0.5)
        elif len(points) == 2:
            # 直接连接两个点
            ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], [points[0][2], points[1][2]], c='r', linewidth=0.5)


    """返回pin的xyz坐标
    #  [(21, 22, 23), (3, 4, 5), (35, 6, 7)],[(12, 13, 14), (14, 15, 16)]
    """
    @staticmethod
    def calPinPosition(pin2node_map, flat_net2pin_map, flat_net2pin_start_map,
                       xx,yy,zz,pin_offset_x,pin_offset_y,pin_offset_z):
        result = []
        for i in range(0, len(flat_net2pin_start_map) - 1, 1):
            start = flat_net2pin_start_map[i]
            end = flat_net2pin_start_map[i+1]
            temp = []
            for j in range(start, end):
                pinId = flat_net2pin_map[j]
                nodeId = pin2node_map[pinId]
                x = xx[nodeId] + pin_offset_x[pinId]
                y = yy[nodeId] + pin_offset_y[pinId]
                z = zz[nodeId] + pin_offset_z[pinId]
                temp.append((x,y,z))
            result.append(temp)
        return result

    @staticmethod
    def forward(
                pos,        
                node_size_x,node_size_y,node_size_z,
                pin_offset_x,pin_offset_y,pin_offset_z,
                pin2node_map,
                xl,yl,zl,xh,yh,zh,
                site_width,
                row_height,
                bin_size_x, bin_size_y, bin_size_z,
                num_movable_nodes,
                num_filler_nodes,
                filename,flat_net2pin_map,flat_net2pin_start_map,
                iteration=None):
        num_nodes = len(pos) // 3
        num_movable_nodes = num_movable_nodes
        num_filler_nodes = num_filler_nodes
        num_physical_nodes = num_nodes - num_filler_nodes
        num_bins_x = int(math.ceil((xh - xl) / bin_size_x))
        num_bins_y = int(math.ceil((yh - yl) / bin_size_y))
        num_bins_y = int(math.ceil((zh - zl) / bin_size_z))
        x = np.array(pos[:num_nodes])
        y = np.array(pos[num_nodes:num_nodes*2])
        z = np.array(pos[num_nodes*2:])
        node_size_x = np.array(node_size_x)
        node_size_y = np.array(node_size_y)
        node_size_z = np.array(node_size_z)
        pin_offset_x = np.array(pin_offset_x)
        pin_offset_y = np.array(pin_offset_y)
        pin_offset_z = np.array(pin_offset_z)
        pin2node_map = np.array(pin2node_map)
        try:
            tt = time.time()
            if xh - xl < yh - yl:
                height = 800 
                width = round(height * (xh - xl) / (yh - yl))
            else:
                width = 800 
                height = round(width * (yh - yl) / (xh -xl))
            depth = zh
          

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

             # 设置视角和辅助线间隔
            ax.view_init(elev=12, azim=15)  
            ax.grid(color='gray', linestyle='-', linewidth=0.5)

            # 设置刻度间隔
            ax.xaxis.set_major_locator(plt.MultipleLocator((xh - xl)/4))
            ax.yaxis.set_major_locator(plt.MultipleLocator((yh - yl)/4))
            ax.zaxis.set_major_locator(plt.MultipleLocator((zh - zl)/4))
            # 隐藏刻度
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])

            # 设置坐标轴范围
            ax.set_xlim([0, xh - xl])
            ax.set_ylim([0, yh - yl])
            ax.set_zlim([0, zh - zl])
            
            
            node_xl = x
            node_yl = y
            node_xh = node_xl + node_size_x[0:len(x)]
            node_yh = node_yl + node_size_y[0:len(y)]

            rectangles = []
            # 不考虑fillers  num_movable_nodes 换成 num_nodes 就是考虑filler  , num_physical_nodes是考虑宏块
            for i in range(num_nodes):
                t = {}
                t['x'] = [node_xl[i], node_xh[i], node_xl[i], node_xh[i]]
                t['y'] = [node_yl[i], node_yh[i], node_yl[i], node_yh[i]]
                t['z'] = z[i]
                rectangles.append(t)

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(PlaceDrawer3D.plot_rect, ax, i, rectangles, num_movable_nodes, num_physical_nodes)
                           for i in range(num_nodes)]

                for future in futures:
                    future.result()
            # rectangles = [
            #     {'x': [0, 1, 1, 0], 'y': [0, 0, 1, 1], 'z': 0},
            #     {'x': [1, 2, 2, 1], 'y': [0, 0, 1, 1], 'z': 1},
            #     {'x': [2, 3, 3, 2], 'y': [0, 0, 1, 1], 'z': 50},
            # ]
            ######## 绘制连接线
            #构造pin坐标组，以net为单位
            '''
            points_groups = PlaceDrawer3DCube.calPinPosition(pin2node_map, flat_net2pin_map, flat_net2pin_start_map,
                                           x,y,z,pin_offset_x,pin_offset_y,pin_offset_z)  

            # 绘制net的点和线
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(PlaceDrawer3D.plot_points, ax, points) for points in points_groups]

                for future in futures:
                    future.result() #'''

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Placement '+ filename)

            # 设置坐标轴边界颜色和宽度
            ax.xaxis.pane.set_edgecolor('black')
            ax.xaxis.pane.set_linewidth(2)
            ax.yaxis.pane.set_edgecolor('black')
            ax.yaxis.pane.set_linewidth(2)
            ax.zaxis.pane.set_edgecolor('black')
            ax.zaxis.pane.set_linewidth(2)

            plt.savefig(filename, bbox_inches='tight')
            print("[I] plotting to %s takes %.3f seconds" %
                    (filename, time.time() - tt))
            plt.close('all')
        except Exception as e:
            print("[E] failed to plot")
            print(str(e))
            return 0

        return 1

class PlaceDrawer3DCube(object):
    """
    3D 绘制函数
    """
    @staticmethod
    def calculate_cube_vertices(x, y, z, width, depth, height):
        vertices = [
            [x, y, z],
            [x + width, y, z],
            [x + width, y + depth, z],
            [x, y + depth, z],
            [x, y, z + height],
            [x + width, y, z + height],
            [x + width, y + depth, z + height],
            [x, y + depth, z + height]
        ]
        return vertices

    @staticmethod
    def plot_cube(ax, cube_data, index, num_movable_nodes, num_physical_nodes, num_nodes):
        cube_info = cube_data[index]
        x, y, z = cube_info['x'], cube_info['y'], cube_info['z']
        width, depth, height = cube_info['width'], cube_info['depth'], cube_info['height']
        if index < num_movable_nodes:
            color = 'skyblue'
        elif index < num_physical_nodes:
            color = (1, 0, 0, 0.7) # macro为红色
        else:
            color = (0.5, 0.5, 0.5, 0.3) # fillers为灰色
        vertices = PlaceDrawer3DCube.calculate_cube_vertices(x, y, z, width, depth, height)
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom surface
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top surface
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side 1
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Side 2
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Side 3
            [vertices[3], vertices[0], vertices[4], vertices[7]]   # Side 4
        ]

        # Plot the faces of the cube with gradient color
        for face in faces:
            square = np.array(face)
            poly = Poly3DCollection([square], alpha=0.5, facecolors=[color]*4, linewidths=1, edgecolors='black')
            ax.add_collection3d(poly)

    """返回pin的xyz坐标
    #  [(21, 22, 23), (3, 4, 5), (35, 6, 7)],[(12, 13, 14), (14, 15, 16)]
    """
    @staticmethod
    def calPinPosition(pin2node_map, flat_net2pin_map, flat_net2pin_start_map,
                       xx,yy,zz,pin_offset_x,pin_offset_y,pin_offset_z):
        result = []
        for i in range(0, len(flat_net2pin_start_map) - 1, 1):
            start = flat_net2pin_start_map[i]
            end = flat_net2pin_start_map[i+1]
            temp = []
            for j in range(start, end):
                pinId = flat_net2pin_map[j]
                nodeId = pin2node_map[pinId]
                x = xx[nodeId] + pin_offset_x[pinId]
                y = yy[nodeId] + pin_offset_y[pinId]
                z = zz[nodeId] + pin_offset_z[pinId]
                temp.append((x,y,z))
            result.append(temp)
        return result

    @staticmethod
    def forward(
                pos,        
                node_size_x,node_size_y,node_size_z,
                pin_offset_x,pin_offset_y,pin_offset_z,
                pin2node_map,
                xl,yl,zl,xh,yh,zh,
                site_width,
                row_height,
                bin_size_x, bin_size_y, bin_size_z,
                num_movable_nodes,
                num_filler_nodes,
                filename,flat_net2pin_map,flat_net2pin_start_map,
                iteration=None):
        
        num_nodes = len(pos) // 3
        num_movable_nodes = num_movable_nodes
        num_filler_nodes = num_filler_nodes
        num_physical_nodes = num_nodes - num_filler_nodes
        num_bins_x = int(math.ceil((xh - xl) / bin_size_x))
        num_bins_y = int(math.ceil((yh - yl) / bin_size_y))
        num_bins_z = int(math.ceil((zh - zl) / bin_size_z))
        xx = np.array(pos[:num_nodes])
        yy = np.array(pos[num_nodes:num_nodes*2])
        zz = np.array(pos[num_nodes*2:])
        node_size_x = np.array(node_size_x)
        node_size_y = np.array(node_size_y)
        node_size_z = np.array(node_size_z)
        pin_offset_x = np.array(pin_offset_x)
        pin_offset_y = np.array(pin_offset_y)
        pin_offset_z = np.array(pin_offset_z)
        pin2node_map = np.array(pin2node_map)
        flat_net2pin_map = np.array(flat_net2pin_map)
        flat_net2pin_start_map = np.array(flat_net2pin_start_map)


        cube_data = []
        for i in range(num_physical_nodes):
            cube_data.append({'x': xx[i], 'y': yy[i], 'z': zz[i], 'width': node_size_x[i],
                              'depth': node_size_y[i], 'height': node_size_z[i]})
        # filler_data = []
        # for i in range(num_physical_nodes+1, num_nodes):
        #     filler_data.append({'x': xx[i], 'y': yy[i], 'z': zz[i], 'width': node_size_x[i],
        #                       'depth': node_size_y[i], 'height': node_size_z[i]})

        try:
            tt = time.time()
            if xh - xl < yh - yl:
                height = 800 
                width = round(height * (xh - xl) / (yh - yl))
            else:
                width = 800 
                height = round(width * (yh - yl) / (xh -xl))
            depth0 = zh
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            # 设置坐标轴范围
            ax.set_xlim([0, xh - xl])
            ax.set_ylim([0, yh - yl])
            ax.set_zlim([0, zh - zl])

            # 设置视角和辅助线间隔
            ax.view_init(elev=12, azim=15)  
            ax.grid(color='gray', linestyle='-', linewidth=0.5)
            
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(PlaceDrawer3DCube.plot_cube, ax, cube_data, i, num_movable_nodes, num_physical_nodes, num_nodes)
                           for i in range(num_physical_nodes)]  # 此处的range确定是否放置宏块
                for future in futures:
                    future.result()

            # Set plot parameters
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Placement '+ filename)

            # 设置刻度间隔
            # ax.xaxis.set_major_locator(plt.MultipleLocator(20))
            # ax.yaxis.set_major_locator(plt.MultipleLocator(20))
            # ax.zaxis.set_major_locator(plt.MultipleLocator(25))

            # 隐藏刻度
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])

            # 设置坐标轴边界颜色和宽度
            ax.xaxis.pane.set_edgecolor('black')
            ax.xaxis.pane.set_linewidth(2)
            ax.yaxis.pane.set_edgecolor('black')
            ax.yaxis.pane.set_linewidth(2)
            ax.zaxis.pane.set_edgecolor('black')
            ax.zaxis.pane.set_linewidth(2)

            ######## 绘制连接线
            #构造pin坐标组，以net为单位
            points_groups = PlaceDrawer3DCube.calPinPosition(pin2node_map, flat_net2pin_map, flat_net2pin_start_map,
                                           xx,yy,zz,pin_offset_x,pin_offset_y,pin_offset_z)  

            '''
            # 逐组绘制点和连接线
            for points in points_groups:
                x = [point[0] for point in points]
                y = [point[1] for point in points]
                z = [point[2] for point in points]
                ax.scatter(x, y, z, c='lime', marker='o')

                if len(points) > 2:
                    # 计算几何中心
                    center = np.mean(points, axis=0)
                    ax.scatter(center[0], center[1], center[2], c='cyan', marker='x')  # 绘制几何中心虚拟点

                    # 连接点到几何中心
                    for point in points:
                        ax.plot([point[0], center[0]], [point[1], center[1]], [point[2], center[2]], c='r', linewidth=0.5)
                elif len(points) == 2:
                    # 直接连接两个点
                    ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], [points[0][2], points[1][2]], c='r', linewidth=0.5)
            '''
            """
            ###### 添加颜色条
            # 创建颜色映射对象
            norm = Normalize(vmin=zz.min(), vmax=zz.max())
            sm = cm.ScalarMappable(cmap=plt.cm.RdBu, norm=norm)
            sm.set_array([]) # 空数组，但是必须设置
            cbar = plt.colorbar(sm, ax=ax, pad=0.1)
            cbar.set_label('Color Legend')
            #####
            """

            plt.savefig(filename, bbox_inches='tight')
            print("[I] plotting to %s takes %.3f seconds" %
                    (filename, time.time() - tt))
            plt.close('all')
        except Exception as e:
            print("[E] failed to plot")
            print(str(e))
            return 0

        return 1
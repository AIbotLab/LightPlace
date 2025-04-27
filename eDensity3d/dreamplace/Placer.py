##
# @file   Placer.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  Main file to run the entire placement flow.
#

from math import gamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
import logging
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
root_dir = os.path.dirname(root_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
import dreamplace.configure as configure
import Params
import PlaceDB
import Timer
import NonLinearPlace
import pdb
from PIL import Image
import re


def place(params):
    """
    @brief Top API to run the entire placement flow.
    @param params parameters
    """

    assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
            "CANNOT enable GPU without CUDA compiled"

    np.random.seed(params.random_seed)
    # read database
    tt = time.time()
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    logging.info("reading database takes %.2f seconds" % (time.time() - tt))

    # Read timing constraints provided in the benchmarks into out timing analysis
    # engine and then pass the timer into the placement core.
    timer = None
    if hasattr(params,"timing_opt_flag") and params.timing_opt_flag:
        tt = time.time()
        timer = Timer.Timer()
        timer(params, placedb)
        # This must be done to explicitly execute the parser builders.
        # The parsers in OpenTimer are all in lazy mode.
        timer.update_timing()
        logging.info("reading timer takes %.2f seconds" % (time.time() - tt))

        # Dump example here. Some dump functions are defined.
        # Check instance methods defined in Timer.py for debugging.
        # timer.dump_pin_cap("pin_caps.txt")
        # timer.dump_graph("timing_graph.txt")

    # solve placement
    tt = time.time()
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    logging.info("non-linear placement initialization takes %.2f seconds" %
                 (time.time() - tt))
    metrics,density_weight_list,gamma_list,node2layer,ori_pos = placer(params, placedb)
    logging.info("non-linear placement takes %.2f seconds" %
                 (time.time() - tt))

    # write placement solution
    path = "%s/%s" % (params.result_dir, params.design_name())
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    gp_out_file = os.path.join(path, "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
    placedb.write(params, gp_out_file)
    print("[INFO   ] Place3D - writing to %s" % (gp_out_file))

    #写文件
    layer_file = os.path.join(path, "%s.layer.txt" % (params.design_name()))
    with open(layer_file, 'w') as file:
        for i in range(placedb.num_movable_nodes):
            file.write("%s %d\n" % (placedb.node_names[i].decode('utf-8'), node2layer[i]))
    print("[INFO   ] Place3D - writing to %s" % (layer_file))
    
    z_pos_file = os.path.join(path, "%s.zPos.txt" % (params.design_name()))
    with open(z_pos_file, 'w') as file:
        for i in range(placedb.num_movable_nodes):
            file.write("%s %.2f\n" % (placedb.node_names[i].decode('utf-8'), ori_pos[i]))
    print("[INFO   ] Place3D - writing to %s" % (z_pos_file))

    # call external detailed placement
    # TODO: support more external placers, currently only support
    # 1. NTUplace3/NTUplace4h with Bookshelf format
    # 2. NTUplace_4dr with LEF/DEF format
    if params.detailed_place_engine and os.path.exists(
            params.detailed_place_engine):
        logging.info("Use external detailed placement engine %s" %
                     (params.detailed_place_engine))
        if params.solution_file_suffix() == "pl" and any(
                dp_engine in params.detailed_place_engine
                for dp_engine in ['ntuplace3', 'ntuplace4h']):
            dp_out_file = gp_out_file.replace(".gp.pl", "")
            # add target density constraint if provided
            target_density_cmd = ""
            if params.target_density < 1.0 and not params.routability_opt_flag:
                target_density_cmd = " -util %f" % (params.target_density)
            cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (
                params.detailed_place_engine, params.aux_input, gp_out_file,
                target_density_cmd, dp_out_file, params.detailed_place_command)
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info("External detailed placement takes %.2f seconds" %
                         (time.time() - tt))

            if params.plot_flag:
                # read solution and evaluate
                placedb.read_pl(params, dp_out_file + ".ntup.pl")
                iteration = len(metrics)
                pos = placer.init_pos
                pos[0:placedb.num_physical_nodes] = placedb.node_x
                pos[placedb.num_nodes:placedb.num_nodes +
                    placedb.num_physical_nodes] = placedb.node_y
                hpwl, density_overflow, max_density = placer.validate(
                    placedb, pos, iteration)
                logging.info(
                    "iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E"
                    % (iteration, hpwl, density_overflow, max_density))
                placer.plot(params, placedb, iteration, pos)
        elif 'ntuplace_4dr' in params.detailed_place_engine:
            dp_out_file = gp_out_file.replace(".gp.def", "")
            cmd = "%s" % (params.detailed_place_engine)
            for lef in params.lef_input:
                if "tech.lef" in lef:
                    cmd += " -tech_lef %s" % (lef)
                else:
                    cmd += " -cell_lef %s" % (lef)
                benchmark_dir = os.path.dirname(lef)
            cmd += " -floorplan_def %s" % (gp_out_file)
            if(params.verilog_input):
                cmd += " -verilog %s" % (params.verilog_input)
            cmd += " -out ntuplace_4dr_out"
            cmd += " -placement_constraints %s/placement.constraints" % (
                # os.path.dirname(params.verilog_input))
                benchmark_dir)
            cmd += " -noglobal %s ; " % (params.detailed_place_command)
            # cmd += " %s ; " % (params.detailed_place_command) ## test whole flow
            cmd += "mv ntuplace_4dr_out.fence.plt %s.fence.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out.init.plt %s.init.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out %s.ntup.def ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.overflow.plt %s.ntup.overflow.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.plt %s.ntup.plt ; " % (
                dp_out_file)
            if os.path.exists("%s/dat" % (os.path.dirname(dp_out_file))):
                cmd += "rm -r %s/dat ; " % (os.path.dirname(dp_out_file))
            cmd += "mv dat %s/ ; " % (os.path.dirname(dp_out_file))
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info("External detailed placement takes %.2f seconds" %
                         (time.time() - tt))
        else:
            logging.warning(
                "External detailed placement only supports NTUplace3/NTUplace4dr API"
            )
    elif params.detailed_place_engine:
        logging.warning(
            "External detailed placement engine %s or aux file NOT found" %
            (params.detailed_place_engine))

    return metrics,density_weight_list,gamma_list

"""
将文件夹下的所有图片转成gif
"""
def picture2gif(folder_path, gif_path):
    # 获取文件夹下的所有图片文件名（排除开头为下划线的文件）
    image_files = [f for f in os.listdir(folder_path) if (f.endswith(".png") or f.endswith(".jpg")) and not f.startswith("_")]

    if len(image_files) == 0:
        logging.info("can not find image files, failed to draw gif")
        return
    # 使用正则表达式提取文件名中的数字，并进行排序
    image_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # 用于保存所有图片的列表
    images = []

    # 将每个图片文件加载到列表中
    for file_name in image_files:
        file_path = os.path.join(folder_path, file_name)
        img = Image.open(file_path)
        images.append(img)

    # 保存为GIF
    images.extend([images[-1]]*40)
    images[0].save(gif_path,
                save_all=True,
                append_images=images[1:],
                optimize=False,
                duration=50,  # 每帧显示时间（以毫秒为单位）
                loop=0)        # 循环次数（0表示无限循环）

    logging.info("Success to create GIF: %s" % (gif_path))

# if __name__ == "__main__":
def placer3d(config_path:str,log_path=None):
    """
    @brief main function to invoke the entire placement flow.
    """
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.ERROR, # logging.ERROR,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    params = Params.Params()
    # params.printWelcome()

    # load parameters
    params.load(config_path)
    logging.info("parameters = %s" % (params))
    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # run placement
    tt = time.time()
    metrics,density_weight_list,gamma_list = place(params)
    logging.info("placement takes %.3f seconds" % (time.time() - tt))
    

    ################################################################
    # 绘制 HPWL 与 overflow 变化曲线
    if params.plot_flag:
        metricslen = len(metrics)
        #因为进行了合法化与细节放置操作的metric不一致，里面直接放的是EvalMetrics类型，因而会导致下面的取法错误
        if params.legalize_flag == 1:
            metricslen -= 1
        if params.detailed_place_flag == 1:
            metricslen -= 1
        x = np.arange(1, metricslen) #不要第一个初始值
        hpwl = []
        overflow = []
        energyList = []
        

        for i in range(1, metricslen):
            hpwl.append(metrics[i][0][0].hpwl.cpu())
            overflow.append(metrics[i][0][0].overflow[0].cpu())
            energyList.append(metrics[i][0][0].density.cpu())
        
        hpwl = np.array(hpwl)
        overflow = np.array(overflow)
        density_weight_list = np.array(density_weight_list)
        gamma_list = np.array(gamma_list)
        #路径和文件名
        path = "%s/%s" % (params.result_dir, params.design_name())
        figname = "%s/%s_hpwl_overflow.png" % (path, params.design_name())
        density_figname = "%s/%s_density_weight.png" % (path, params.design_name())
        gamma_figname = "%s/%s_gamma.png" % (path, params.design_name())
        energy_figname = "%s/%s_energy.png" % (path, params.design_name())
        
        ###############################
        #画hpwl与重叠率变化图
        fig = plt.figure()

        ax_hpwl = fig.add_subplot(111)
        ax_hpwl.plot(x,hpwl,"b")
        ax_hpwl.set_ylabel("hpwl")

        ax_hpwl.set_xlabel("iteration")
        ax_hpwl.set_title("%s GP variation curve" % (params.design_name()))
        # ax_hpwl.legend() # 显示图例

        ax_overflow = ax_hpwl.twinx()
        ax_overflow.plot(x,overflow,"r",label="overflow")
        ax_overflow.set_ylabel("overflow")
        ax_overflow.legend()

        #保存图片
        fig.savefig(figname)
        plt.close('all')
        logging.info("plotting to %s " % (figname))

        
        ###############################
        # 绘制density_weight变化图，即lambda
        fig = plt.figure()
        # 在图像对象上创建子图
        ax = fig.add_subplot(1, 1, 1)
        # 绘制带线的散点图
        x = np.arange(len(density_weight_list))
        ax.plot(x, density_weight_list,'b')  # marker表示点的样式，linestyle表示线的样式，color表示颜色
        ax.set_xlabel('iter')
        ax.set_ylabel('density_weight')
        ax.set_title("%s density_weight variation curve" % (params.design_name()))
        plt.savefig(density_figname)
        plt.close('all')
        logging.info("plotting to %s " % (density_figname))


        ###############################
        # 绘制gamma变化图，即线长的gamma
        fig = plt.figure()
        # 在图像对象上创建子图
        ax = fig.add_subplot(1, 1, 1)
        # 绘制带线的散点图
        x = np.arange(len(gamma_list)) 
        ax.plot(x, gamma_list,'b')  # marker表示点的样式，linestyle表示线的样式，color表示颜色
        ax.set_xlabel('iter')
        ax.set_ylabel('gamma')
        ax.set_title("%s gamma variation curve" % (params.design_name()))
        plt.savefig(gamma_figname)
        plt.close('all')
        logging.info("plotting to %s " % (gamma_figname))

        ###############################
        # 绘制density(energy)变化图
        fig = plt.figure()
        # 在图像对象上创建子图
        ax = fig.add_subplot(1, 1, 1)
        # 绘制带线的散点图
        x = np.arange(len(energyList)) 
        ax.plot(x, energyList,'b')  # marker表示点的样式，linestyle表示线的样式，color表示颜色
        ax.set_xlabel('iter')
        ax.set_ylabel('energy (density)')
        ax.set_title("%s energy variation curve" % (params.design_name()))
        plt.savefig(energy_figname)
        plt.close('all')
        logging.info("plotting to %s " % (energy_figname))

        
        # 制作gif图
        folder_path = "%s/plot" % (path) 
        gif_path = "%s/%s_iter_plot.gif" % (path,params.design_name())
        if os.path.exists(folder_path):
            picture2gif(folder_path, gif_path)
        plt.close('all')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Placer.py benchmark/xx.json")
        exit(1)
    placer3d(sys.argv[1])

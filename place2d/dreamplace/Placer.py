##
# @file   Placer.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  Main file to run the entire placement flow.
#

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
import NonLinearPlace
import pdb



def place(params, legal, gp_file):
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

    # solve placement
    tt = time.time()
    placer = NonLinearPlace.NonLinearPlace(params, placedb)
    logging.info("non-linear placement initialization takes %.2f seconds" %
                 (time.time() - tt))
    metrics = placer(params, placedb, legal)
    logging.info("non-linear placement takes %.2f seconds" %
                 (time.time() - tt))

    # write placement solution
    path = "%s/%s/layer_%s" % (params.result_dir, params.design_name(), str(params.curLayer))
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    gp_out_file = os.path.join(
        path,
        "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
    placedb.write(params, gp_out_file)
    gp_file[0] = gp_out_file

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

    return metrics

def placer2d(config_path:str, curLayer, legal, log_path=None):
    """
    @brief main function to invoke the entire placement flow.
    """
    logging.root.name = "Place3D"
    logging.basicConfig(stream=sys.stdout,
                        level=logging.ERROR,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        )
    params = Params.Params()
    params.printWelcome()

    # load parameters
    params.load(config_path)
    params.curLayer = curLayer
    logging.info("parameters = %s" % (params))
    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # run placement
    tt = time.time()
    gp_file = [""] # pl结果路径
    metrics = place(params, legal, gp_file)
    logging.info("placement takes %.3f seconds" % (time.time() - tt))
    
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

        for i in range(1, metricslen):
            hpwl.append(metrics[i][0][0].hpwl.cpu())
            overflow.append(metrics[i][0][0].overflow[0].cpu())
        # 取出最后两个
        # if params.legalize_flag == 1 and params.detailed_place_flag == 1:
        #     hpwl.append(metrics[len(metrics)-2].hpwl.cpu())
        #     overflow.append(metrics[len(metrics)-3][0][0].overflow[0].cpu()) #保持上一个

        #     hpwl.append(metrics[len(metrics)-1].hpwl.cpu())
        #     overflow.append(metrics[len(metrics)-3][0][0].overflow[0].cpu()) #保持上一个

        # elif params.legalize_flag == 1 or params.detailed_place_flag == 1:
        #     hpwl.append(metrics[len(metrics)-1].hpwl.cpu())
        #     overflow.append(metrics[len(metrics)-2][0][0].overflow[0].cpu()) #保持上一个

        hpwl = np.array(hpwl)
        overflow = np.array(overflow)
        #路径和文件名
        path = "%s/%s/layer_%s" % (params.result_dir, params.design_name(), str(params.curLayer))
        figname = "%s/plot/_hpwl_overflow.png" % (path)

        #画图
        fig = plt.figure()

        ax_hpwl = fig.add_subplot(111)
        ax_hpwl.plot(x,hpwl,"b")
        ax_hpwl.set_ylabel("hpwl")
        ax_hpwl.set_xlabel("iteration")
        ax_hpwl.set_title("GP variation curve")

        ax_overflow = ax_hpwl.twinx()
        ax_overflow.plot(x,overflow,"r",label="overflow")
        ax_overflow.set_ylabel("overflow")
        ax_overflow.legend()

        #保存图片
        fig.savefig(figname)
        logging.info("plotting to %s " % (figname))
        # 关闭所有打开的图形窗口
        plt.close('all')
    return gp_file[0]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: Placer.py benchmark/xx.json curLayer")
        exit(1)
    legal = [True, True]
    placer2d(sys.argv[1], int(sys.argv[2]), legal)
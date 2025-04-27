import sys
import os.path as path
import os
import time
import random

# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import database.DataBase as DataBase
import database.Logger as Logger
# import place3d.build.dreamplace.Placer as Placer3d #3D 直接外部调用
import place2d.build.dreamplace.Placer as Placer2d # 2D 

def create_nested_folders(root, folders):
    """
    递归地创建嵌套的文件夹结构
    Args:
        root: 根文件夹路径
        folders: 文件夹结构字典 每个键表示文件夹名称 值是子文件夹的结构字典或None
    """
    for folder_name, sub_folders in folders.items():
        folder_path = os.path.join(root, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder {folder_path} was created")
        # else:
        #     print(f"Folder '{folder_path}' already exists")
        
        if sub_folders is not None:
            create_nested_folders(folder_path, sub_folders)

# 文件夹结构字典
folder_structure = {
    'temp': {
        'a_mix3d': {
            'bookshelf': None,
            'json': None,
            'result':None
        },
        'b_mix2d': {
            'bookshelf': None,
            'json': None,
            'result':None
        },
        'c_cell3d': {
            'bookshelf': None,
            'json': None,
            'result':None
        },
        'd_cell2d': {
            'bookshelf': None,
            'json': None,
            'result':None
        },
    }
}

# 程序路径
program3d = "eDensity3d/build/dreamplace/Placer.py"
# 运行 python program3d json
program2d = "place2d/build/dreamplace/Placer.py" 
# program2d = "autoDMP/build/dreamplace/Placer.py"
# 运行 python program2d json curlayer

# 阶段
phase = ["Mixed_3DGP", "Mixed_2DGP", "Cell_3DGP", "Cell_2DLG"]

# 一级目录
firstDir = "temp"

# 二级目录
secondDir = ["a_mix3d", "b_mix2d", "c_cell3d", "d_cell2d"]

# 三级目录
thirdDir = ["bookshelf", "json", "result"]

# 每一个方法都要先生成bookshelf文件然后再生成json文件
# 指定哪些阶段要做
stages = [1, 1, 1, 1] # 对应"a_mix3d", "b_mix2d", "c_cell3d", "d_cell2d"
runProgram = [1, 1, 1, 1] # 对应"a_mix3d", "b_mix2d", "c_cell3d", "d_cell2d" 中的程序是否运行
addV1 = True # 是否添加巨大虚拟宏块


def main():
    #######################
    ### 0 pre operation ###
    #######################
    # 预操作
    # 0.1 读入外部参数 需要原始ispd2005文件路径下的aux路径作为输入
    stratTime = time.time()
    auxInputFile = "benchmarks/ispd2005/adaptec1/adaptec1.aux"
    numLayer = 2
    zAlpha = 0.25  # 默认值
    if len(sys.argv) == 3 or len(sys.argv) == 4:
        auxInputFile = sys.argv[1]
        numLayer = int(sys.argv[2])
        if len(sys.argv) == 4:
            zAlpha = float(sys.argv[3])
    else:
        print(f"Usage: python {sys.argv[0]} aux_input_file_path 2")
        exit(1)
    Logger.printPlace(f"zAlpha:{zAlpha}")    
    auxFileName = os.path.basename(auxInputFile) # 获取最后的文件名
    auxFileName = os.path.splitext(auxFileName)[0] # 去掉后缀名
    dataBase = DataBase.DataBase()
    dataBase.auxFileName = auxFileName
    dataBase.fileName = auxFileName+"_"+str(numLayer)  # 加上分层的后缀
    dataBase.numLayer = numLayer
    # if len(sys.argv) == 3:
    #     if auxFileName in bata_z_map:
    #         if numLayer in bata_z_map[auxFileName]:
    #             beta_z = bata_z_map[auxFileName][numLayer]
    #         else:
    #             beta_z = -1
    #     else:
    #         beta_z = -1
    
    # 0.2 创建temp文件存放 文件夹
    create_nested_folders(path.dirname(__file__), folder_structure)
    
    # 0.3 读取文件
    Logger.printPlace("Reading inputFile " + auxInputFile)
    dataBase.readBookshelf(auxInputFile)
    # dataBase.stats2layer()
    # 设置3D numbin数量
    numbins_x = 128
    numbins_y = 128
    numbins_z = 128
    
    dataBase.updateData(numbins_x, numbins_y, numbins_z) # 更新database的一些数据，如pin相对左下角的位置  更新芯片大小
    dataBase.updateMacroConnectNets() # 更新macro连接的nets
    
    # dataBase.statsInfo() # 统计输入样例的一些数据
    
    # 在中心生成一个巨大固定宏块 只在第一第二阶段起作用
    if addV1:
        dataBase.addVirtualMacro()

    # 加入随机变量
    deterministic_flag = 1
    random_seed = 1000
    iteration = 1000  # 换成1000 加快结束
    maxIt = 5
    while(True):
        ######################
        #### 1 Macro 3DGP ####
        ######################
        maxIt -= 1
        # 进行混合的macro可移动的放置
        phaseNum = 0
        if stages[phaseNum]:      
            bookshelfDir = os.path.join(path.dirname(__file__), firstDir, secondDir[phaseNum], thirdDir[0])
            jsonDir= os.path.join(path.dirname(__file__), firstDir, secondDir[phaseNum], thirdDir[1])
            resultDir= os.path.join(path.dirname(__file__), firstDir, secondDir[phaseNum], thirdDir[2])
            print(f" ###################### Phase {phaseNum + 1} {phase[phaseNum]} ###################### ")
            # 1.1 生成bookshelf文件
            auxFile,nodeInThisLayer = dataBase.generateMacroConnectBookShelf(bookshelfDir, dataBase.fileName, curlayer = -1,  moveMacro = True)
            jsonFile = dataBase.generateJson(jsonDir, dataBase.fileName, auxFile, resultDir, dataBase.numLayer,
                                            curlayer = -1, gpu = 0, gp = 1, lg = 0, dp = 0, plotF = 0, _stopOverflow = 0.06, _enable_fillers = 1,
                                            _numBins = numbins_x, _num_bins_z = numbins_z, _deterministic_flag = deterministic_flag, _random_seed = random_seed,
                                            _iteration = iteration, _zAlpha = zAlpha) 
            #layer = -1 表示不分层  3D plotF不能打开, 画不了图
            Logger.printPlace("Generate bookshelf file " + auxFile)
            Logger.printPlace("Generate json file " + jsonFile)
            # 1.2 运行3DGP
            if runProgram[phaseNum]:
                Logger.printPlace(f"{phase[phaseNum]} Placing ...")
                program = f"python {program3d} {jsonFile}"
                status = os.system(program)
                if status != 0: #运行出错
                    print(f"running {program} EEROR!")
                    # 进行随机GP
                    deterministic_flag = 0
                    random_seed = random.randint(1, 1000)
                    iteration = 1000 # 可能由于过多迭代导致运行失败
                    continue
            
            # 1.3 读取结果文件
            plFile = os.path.join(resultDir, dataBase.fileName, dataBase.fileName + ".gp.pl")
            layerFile = os.path.join(resultDir, dataBase.fileName, dataBase.fileName + ".layer.txt")  
            dataBase.readResultFile(plFile, layerFile, phase[phaseNum])
        
            dataBase.calculHPWL()
            Logger.printPlace(f"HPWL {dataBase.HPWL} HPWL3d {dataBase.HPWL3d} VI {dataBase.VI}")
        
        # dataBase.drawMacroConnect()
        ######################
        #### 2 Macro 2DLG ####  # 只有宏块合法化 
        ######################
        phaseNum = 1
        if stages[phaseNum]:
            bookshelfDir = os.path.join(path.dirname(__file__), firstDir, secondDir[phaseNum], thirdDir[0])
            jsonDir= os.path.join(path.dirname(__file__), firstDir, secondDir[phaseNum], thirdDir[1])
            resultDir= os.path.join(path.dirname(__file__), firstDir, secondDir[phaseNum], thirdDir[2])
            print(f" ###################### Phase {phaseNum + 1} {phase[phaseNum]} ###################### ")
            
            legal = [True,True]
            for curlayer in range(dataBase.numLayer):
                print(f" ------------ layer {curlayer} ------------ ")
                legal = [True,True] # 对应每层的Macro、cell是否合法，得两个都是true才能进入下一阶段
                #################################
                # 合法化一层分为三个层次
                # 0. 直接合法化 不进行GP
                # 1. 去除巨大宏块再合法化
                # 2. 有巨大宏块 进行GP-合法化
                # 3. 去除巨大宏块再合法化
                # 如果addV1为False则只会执行 0 2
                #################################
                
                # 单独处理bigblue2 bigbule4中宏块过多的情况
                tooManyMacro = False
                numLayerMacro = 0
                for _,macro in dataBase.macros.items():
                    if macro.layer == curlayer:
                        numLayerMacro += 1
                if numLayerMacro > 5000:
                    # 五千以上则将最小的宏块看成标准单元处理
                    tooManyMacro = True
                # 循环中会修改的参数
                gp = 1
                enable_fillers = 1
                moveMacro = True # 在 tooManyMacro 时使用
                stopOverflow = 0.07
                for i in range(4):
                    # 调整参数
                    if i == 0:
                        gp = 0
                        enable_fillers = 0
                        if addV1:
                            dataBase.macros["v1"].layer = curlayer # 修改一下最大宏块所属层，他应该存在于每一层中央
                    elif i == 1:
                        gp = 0
                        enable_fillers = 0
                        if addV1:
                            dataBase.macros["v1"].layer = 100 # 让巨大宏块不可见
                        else:
                            continue
                    elif i == 2:
                        gp = 1
                        enable_fillers = 1
                        if addV1:
                            dataBase.macros["v1"].layer = curlayer # 修改一下最大宏块所属层，他应该存在于每一层中央
                    elif i == 3:
                        gp = 0
                        enable_fillers = 0
                        if addV1:
                            dataBase.macros["v1"].layer = 100 # 让巨大宏块不可见
                        else:
                            continue
                    print(f" -------  {i}th attempt at macro legalization ------- ")
                    # 2.1 生成bookshelf文件
                    auxFile,nodeInThisLayer = dataBase.generateMacroConnectBookShelf(bookshelfDir, dataBase.fileName, curlayer = curlayer,
                                                                                    moveMacro = moveMacro, macrolegal = i, tooManyMacro = tooManyMacro)
                    Logger.printPlace("Generate bookshelf file " + auxFile)
                    if nodeInThisLayer is False:
                        # 意味着该层没有可移动节点，都是terminal_NI，直接跳过
                        Logger.printPlace(f"There are no movable nodes in {curlayer} layer")
                        break
                    jsonFile = dataBase.generateJson(jsonDir, dataBase.fileName, auxFile, resultDir, dataBase.numLayer,
                                                    curlayer = curlayer, gpu = 1, gp = gp, lg = 1, dp = 0, plotF = 0, 
                                                    _stopOverflow = stopOverflow, _enable_fillers = enable_fillers, 
                                                    onlyMacroLgFlag = 1, _numBins = 1024) 
                    Logger.printPlace("Generate json file " + jsonFile)
                    # 2.2 分层运行2DGP
                    if runProgram[phaseNum]:
                        Logger.printPlace(f"{phase[phaseNum]} Placing ...")
                        plFile = Placer2d.placer2d(jsonFile, curlayer, legal)
                    # 2.3 读取结果文件
                    # plFile = os.path.join(resultDir, dataBase.fileName, "layer_"+str(curlayer), dataBase.fileName + ".gp.pl") 
                    dataBase.readResultFile(plFile, None, phase[phaseNum])
                    
                    #################
                    if tooManyMacro: #还需要固定大的宏块，再对小的看做标准单元的宏块进行合法化
                        # 2.4 生成bookshelf文件
                        print(" -------  tooManyMacro at macro legalization ------- ")
                        Logger.printPlace(f"Fix large macros and move smallest macros")
                        auxFile,nodeInThisLayer = dataBase.generateMacroConnectBookShelf(bookshelfDir, dataBase.fileName, curlayer = curlayer,
                                                                                        moveMacro = False, macrolegal = i, tooManyMacro = tooManyMacro)
                        Logger.printPlace("Generate bookshelf file " + auxFile)
                        if nodeInThisLayer is False:
                            # 意味着该层没有可移动节点，都是terminal_NI，直接跳过
                            Logger.printPlace(f"There are no movable nodes in {curlayer} layer")
                            break
                        jsonFile = dataBase.generateJson(jsonDir, dataBase.fileName, auxFile, resultDir, dataBase.numLayer,
                                                        curlayer = curlayer, gpu = 1, gp = gp, lg = 1, dp = 0, plotF = 0, 
                                                        _stopOverflow = stopOverflow, _enable_fillers = enable_fillers, 
                                                        onlyMacroLgFlag = 0, _numBins = 1024) 
                        Logger.printPlace("Generate json file " + jsonFile)
                        # 2.5 分层运行2DGP
                        if runProgram[phaseNum]:
                            Logger.printPlace(f"{phase[phaseNum]} Placing ...")
                            plFile = Placer2d.placer2d(jsonFile, curlayer, legal)
                        # 2.6 读取结果文件
                        # plFile = os.path.join(resultDir, dataBase.fileName, "layer_"+str(curlayer), dataBase.fileName + ".gp.pl") 
                        dataBase.readResultFile(plFile, None, phase[phaseNum])
                    #################
                    
                    if legal[0] : # macro 已经合法
                        if tooManyMacro: # 如果是过多宏块下，宏块当成标准单元，则还需要判断cell是否合法
                            if legal[1]:
                                Logger.printPlace(f"layer {curlayer} is legal")
                                break
                        else:
                            Logger.printPlace(f"layer {curlayer} is legal")
                            break
                    
                if not legal[0] or (tooManyMacro and not legal[1]): # 宏块不合法 
                    # 如果是过多宏块下，宏块当成标准单元，则还需要判断cell是否合法
                    Logger.printPlace(f"layer {curlayer} macro not legal")
                    # 进行随机GP
                    deterministic_flag = 0
                    random_seed = random.randint(1, 1000)
                    break
                
            if legal[0] and legal[1]:
                if addV1:
                    # 阶段2结束移除掉固定的巨大宏块
                    dataBase.removeNode() 
                dataBase.calculHPWL()
                Logger.printPlace(f"HPWL {dataBase.HPWL} HPWL3d {dataBase.HPWL3d} VI {dataBase.VI}")
                Logger.printPlace(f"every layer is legal and go to next stage")
                break
            # 否则继续回到第一步GP
            else:
                Logger.printPlace(f"back to the macro 3DGP stage")

        if maxIt <= 0:
            Logger.printPlace("3D Macro Placement run more than 10 times, fail to leagl")
            exit(1)

    ######################
    #### 3 cell 3DGP  ####
    ######################
    phaseNum = 2
    zPosFile = None # 为第四阶段使用做准备
    if stages[phaseNum]:
        bookshelfDir = os.path.join(path.dirname(__file__), firstDir, secondDir[phaseNum], thirdDir[0])
        jsonDir= os.path.join(path.dirname(__file__), firstDir, secondDir[phaseNum], thirdDir[1])
        resultDir= os.path.join(path.dirname(__file__), firstDir, secondDir[phaseNum], thirdDir[2])
        print(f" ###################### Phase {phaseNum + 1} {phase[phaseNum]} ###################### ")
        # 3.1 生成bookshelf文件
        # 不移动宏块
        auxFile,nodeInThisLayer = dataBase.generateBookShelf(bookshelfDir, dataBase.fileName, curlayer = -1, moveMacro = False)
        jsonFile = dataBase.generateJson(jsonDir, dataBase.fileName, auxFile, resultDir, dataBase.numLayer,
                                        curlayer = -1, gpu = 0, gp = 1, lg = 0, dp = 0, plotF = 0, _stopOverflow = 0.06, 
                                        _enable_fillers = 1, _iteration = 1000, _numBins = numbins_x, _num_bins_z = numbins_z, _zAlpha = zAlpha) # _iteration = 1000
        #layer = -1 表示不分层  3D plotF不能打开, 画不了图
        Logger.printPlace("Generate bookshelf file " + auxFile)
        Logger.printPlace("Generate json file " + jsonFile)
        # 3.2 运行Cell 3DGP
        if runProgram[phaseNum]:
            Logger.printPlace(f"{phase[phaseNum]} Placing ...")
            program = f"python {program3d} {jsonFile}"
            status = os.system(program)
            if status != 0: #运行出错
                print(f"running {program} EEROR!")
                Logger.printPlace("Placement total takes %.2f seconds" % (time.time() - stratTime))
                exit(1)
    
        # 3.3 读取结果文件
        plFile = os.path.join(resultDir, dataBase.fileName, dataBase.fileName + ".gp.pl")
        layerFile = os.path.join(resultDir, dataBase.fileName, dataBase.fileName + ".layer.txt")
        zPosFile = os.path.join(resultDir, dataBase.fileName, dataBase.fileName + ".zPos.txt")
        dataBase.readResultFile(plFile, layerFile, phase[phaseNum])
    
        dataBase.calculHPWL()
        Logger.printPlace(f"HPWL {dataBase.HPWL} HPWL3d {dataBase.HPWL3d} VI {dataBase.VI}")
    #########################
    #### 4 cell GP&LG&DP ####
    #########################
    while(True):
        phaseNum = 3
        if stages[phaseNum]:
            bookshelfDir = os.path.join(path.dirname(__file__), firstDir, secondDir[phaseNum], thirdDir[0])
            jsonDir= os.path.join(path.dirname(__file__), firstDir, secondDir[phaseNum], thirdDir[1])
            resultDir= os.path.join(path.dirname(__file__), firstDir, secondDir[phaseNum], thirdDir[2])
            print(f" ###################### Phase {phaseNum + 1} {phase[phaseNum]} ###################### ")
            
            legal = [True,True]
            for curlayer in range(dataBase.numLayer):
                # 4.1 生成bookshelf文件
                print(f" ------------ layer {curlayer} ------------ ")
                legal = [True,True] 
                auxFile,nodeInThisLayer = dataBase.generateBookShelf(bookshelfDir, dataBase.fileName, curlayer = curlayer, moveMacro = False)
                if nodeInThisLayer is False:
                    #该层没有可移动节点，都是terminal_NI
                    Logger.printPlace(f"There are no movable nodes in {curlayer} layer")
                    continue
                jsonFile = dataBase.generateJson(jsonDir, dataBase.fileName, auxFile, resultDir, dataBase.numLayer,
                                                curlayer = curlayer, gpu = 1, gp = 1, lg = 1, dp = 1, plotF = 0, 
                                                _stopOverflow = 0.07, _enable_fillers = 1, _numBins = 1024) 
                Logger.printPlace("Generate bookshelf file " + auxFile)
                Logger.printPlace("Generate json file " + jsonFile)
                # 4.2 分层运行 2D LG & DP
                if runProgram[phaseNum]:
                    Logger.printPlace(f"{phase[phaseNum]} Placing ...")
                    Placer2d.placer2d(jsonFile, curlayer, legal)
                # 4.3 读取结果文件
                plFile = os.path.join(resultDir, dataBase.fileName, "layer_"+str(curlayer), dataBase.fileName + ".gp.pl") 
                dataBase.readResultFile(plFile, None, phase[phaseNum])
                if not legal[1]: # 标准单元不合法
                    Logger.printPlace(f"layer {curlayer} cell not legal")
                    
                    dataBase.parsezPosFile(zPosFile) # 修改单元所属层
                    break
            if legal[0] and legal[1]:
                Logger.printPlace(f"every layer is legal and go to next stage")
                break
            # 否则继续回到循环开始
            else:
                Logger.printPlace(f"back to the cell 2d stage")
                
            
            
    ######################
    #### 5 calculHPWL ####
    ######################
    dataBase.calculHPWL()
    Logger.printPlace(f"HPWL {dataBase.HPWL} HPWL3d {dataBase.HPWL3d} VI {dataBase.VI}")
    
    
    endTime = time.time()
    Logger.printPlace("Placement total takes %.2f seconds" % (endTime - stratTime))
    # print(f"{dataBase.HPWL}, {dataBase.VI}, {endTime - stratTime}")
    with open("record/record250419debug.csv","a",encoding="utf-8") as f:
    # with open("log/record250118tune_5-10.csv","a",encoding="utf-8") as f:
    # with open("log/record250118_volum.csv","a",encoding="utf-8") as f:
        f.write(f"{dataBase.auxFileName},{dataBase.numLayer},{dataBase.HPWL},{dataBase.VI},{endTime - stratTime},{zAlpha}\n")
        
    
    return dataBase.HPWL

if __name__ == "__main__":
    main()
    

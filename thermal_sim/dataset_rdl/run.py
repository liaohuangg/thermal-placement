import os
import time

# 获取脚本所在目录，确保路径相对于脚本位置
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
parent_dir = os.path.dirname(script_dir)



for i in range(400):
    # ChipletCoreFile = "Chiplet_Core"+str(i)+".flp"
    ChipletCoreFile = os.path.join(script_dir, "Chiplet_Core"+str(i)+".flp")
    os.rename(ChipletCoreFile, "Chiplet_Core.flp")
    for j in range(20):
        hotspot_bin = os.path.join(parent_dir, "hotspot")
        config_file = os.path.join(script_dir, "hotspot.config")
        layer_file = os.path.join(script_dir, "Chiplet.lcf")
        ptrace_file = os.path.join(script_dir, "Chiplet_Core"+str(i)+"_Power"+str(j)+".ptrace")
        steady_file = os.path.join(script_dir, "Chiplet.steady")
        grid_steady_file = os.path.join(script_dir, "Chiplet.grid.steady")
        cmd = f"{hotspot_bin} -c {config_file} -f Chiplet_Core.flp -p {ptrace_file} -steady_file {steady_file}  -model_type grid -detailed_3D on -grid_layer_file {layer_file} -grid_steady_file {grid_steady_file}"
        tmr_start = time.time()
        os.system(cmd)
        tmr_end = time.time()
        print(tmr_end - tmr_start)

       # cmd = "../grid_thermal_map.pl Chiplet_Core.flp Chiplet.grid.steady 64 64 > Chiplet" + str(i) + str(j) + ".svg"
        
       # os.system(cmd)

        print(i,j)

        
        EdgeFile = os.path.join(data_dir, "Edge"+"_"+str(i)+"_"+str(j)+".csv")
        os.rename(os.path.join(data_dir, "Edge.csv"), EdgeFile)
        TempFile = os.path.join(data_dir, "Temperature"+"_"+str(i)+"_"+str(j)+".csv")
        os.rename(os.path.join(data_dir, "Temperature.csv"), TempFile)
        PowerFile = os.path.join(data_dir, "Power"+"_"+str(i)+"_"+str(j)+".csv")
        os.rename(os.path.join(data_dir, "Power.csv"), PowerFile)


    os.rename("Chiplet_Core.flp", ChipletCoreFile)
    
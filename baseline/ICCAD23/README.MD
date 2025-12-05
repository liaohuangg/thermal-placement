
## src/ :
chiplet_model.py : 基础数据结构，芯片 建模，问题建模。
TCG.py :TCG建模 ，判断合法性
Generate_initial_TCG.py ：从TCG生成布局layout
Bridge_Overlap_Adjustment.py ：判断硅桥重叠
legalize_tcg.py ：基础layout优化
unit.py ：一些工具函数，IO,可视化等
Legality_optimized_SA.py ：核心方法SA



## test/ :（调试，调参，主要用这里）
## 目前来说，对于10以内的芯粒求解，效果不错，若是数量再增加，找解困难
test_SA_1.py :单次运行SA
test_multi.py :多次运行SA


## test_input/ :
例：{
  "chiplets": [
    {"name": "A", "width": 10.0, "height": 10.0},
    {"name": "B", "width": 10.0, "height": 10.0},
    {"name": "C", "width": 10.0, "height": 10.0}
  ],
  "connections": [
    ["A", "B"],
    ["B", "C"]

  ]
}


包含一个chiplet集合和一个连接集合，注意：连接集合要合法，否则无法求解出布局


## out_put/ :
输出的图片和json格式layout都在这



## how to run ？ ：
pip install -r requirements.txt
cd test 
python test_multi.py 或 python_SA_1.py


## to do：
1.未生成相似TCG群组 ，即文章Group Construction部分
2.目前评价指标为”面积利用率“
3.算法效率较低，部分硬编码


# MAPPO Chiplet Placement Implementation - Summary

## ✅ 实现完成

已成功实现基于MAPPO算法的同构多智能体chiplet布局优化系统。

## 📁 文件结构

```
src/MARL/
├── config.py                 # MAPPO配置参数
├── train_mappo.py            # 主训练脚本
├── test_mappo.py             # 单元测试
├── run.py                    # 快速启动脚本
├── README.md                 # 详细文档
├── agents/
│   ├── __init__.py
│   ├── policy_net.py         # Actor-Critic网络
│   └── marl_solver.py        # MAPPO求解器
├── envs/
│   ├── __init__.py
│   └── chiplet_env.py        # 多智能体环境
└── utils/
    ├── __init__.py
    └── visualize.py          # 可视化工具
```

## 🚀 快速开始

### 方式1: 使用快速启动脚本

```powershell
cd z:\work\placement\thermal-placement\src\MARL
python run.py
```

然后按提示选择:
- 选项1: 运行测试 (首次推荐)
- 选项2: 训练所有测试用例

### 方式2: 直接运行

```powershell
# 测试实现
cd z:\work\placement\thermal-placement\src
python -m MARL.test_mappo

# 训练模型
python -m MARL.train_mappo
```

## 🎯 核心特性

### 1. 多智能体架构
- **每个chiplet = 一个智能体**
- 独立决策放置位置和旋转
- 参数共享的Actor网络
- 中心化的Critic网络(CTDE)

### 2. 约束满足
✅ **强制约束**:
- 无重叠: chiplet不能重叠
- 边界约束: 必须在估算边界内

✅ **软约束**(通过奖励函数):
- **相邻约束**: 有连接的chiplet必须满足重叠长度 ≥ `min_overlap_length`
- 面积最小化: 最小化外接框面积
- 间隙惩罚: 减少chiplet之间的间隙

### 3. 动作空间
- 离散化网格: 50×50 (可配置)
- 动作: `(grid_x, grid_y, rotation)`
- 自动过滤无效动作(重叠检测)

## ⚙️ 配置参数

在 `config.py` 中调整:

```python
class MAPPOConfig:
    # 环境设置
    grid_resolution = 50          # 网格分辨率
    min_overlap_length = 0.5      # 最小重叠长度(连接约束)
    
    # MAPPO超参数
    lr_actor = 3e-4              # Actor学习率
    lr_critic = 3e-4             # Critic学习率
    gamma = 0.99                 # 折扣因子
    clip_ratio = 0.2             # PPO clip参数
    ppo_epochs = 4               # PPO更新轮数
    
    # 训练设置
    num_episodes = 2000          # 训练轮数
    max_steps_per_episode = 200  # 每轮最大步数
    
    # 奖励权重
    overlap_penalty = 1000.0      # 重叠惩罚
    adjacency_penalty = 500.0     # 相邻约束违反惩罚
    adjacency_reward = 100.0      # 相邻约束满足奖励
    area_penalty = 1.0            # 面积惩罚
    gap_penalty = 50.0            # 间隙惩罚
```

## 📊 输入格式

测试用例位于: `baseline/ICCAD23/test_input/`

JSON格式:
```json
{
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
```

## 📈 输出结果

### 1. 控制台输出
```
Episode 50/2000
  Avg Reward: 245.67
  Avg Length: 12.3
  Best Reward: 389.45
  Placed: 3/3
  Adjacency: 2/2 (100%)

Final Statistics:
  Best reward: 389.45
  Placed chiplets: 3/3
  Adjacency constraints: 2/2 (100%)
  Bounding box area: 324.5
```

### 2. 可视化图片
保存位置: `output/MARL/`

图中显示:
- 彩色矩形: 各个chiplet
- 绿色实线: 满足的连接约束
- 红色虚线: 违反的连接约束
- (R)标记: 旋转的chiplet

## 🔧 算法细节

### MAPPO流程

1. **初始化**: 创建Actor(每个智能体共享)和Critic(中心化)
2. **收集轨迹**: 
   - 每个智能体根据局部观察选择动作
   - 环境执行联合动作
   - 检测冲突并计算奖励
3. **计算优势**: 使用GAE(Generalized Advantage Estimation)
4. **策略更新**: 
   - PPO clipped objective更新Actor
   - MSE损失更新Critic
   - 多epoch重用数据

### 状态表示

每个智能体的观察包含:
- Grid占用情况 (2500维, 50×50)
- 自身尺寸信息 (3维)
- 其他智能体位置和连接信息 (4×(n-1)维)

### 奖励函数

```python
global_reward = (
    adjacency_reward × 满足的连接数
    - adjacency_penalty × 违反的连接数
    - area_penalty × 外接框面积
    - gap_penalty × 总间隙
)

agent_reward = global_reward / 已放置智能体数 (冲突则惩罚)
```

## 🎓 关键技术点

1. **中心化训练,去中心化执行(CTDE)**
   - 训练时Critic看到全局状态
   - 执行时Actor只基于局部观察

2. **参数共享**
   - 所有智能体共享同一个Actor网络
   - 减少参数量,提高样本效率

3. **有效动作过滤**
   - 实时计算每个智能体的有效动作
   - mask掉无效动作(会重叠的位置)

4. **冲突检测**
   - 同一步多个智能体可能选择冲突位置
   - 只执行无冲突的放置
   - 冲突智能体获得负奖励

## 📝 测试用例

当前支持:
- `3core.json`: 3个chiplet, 2条连接
- `5core.json`: 5个chiplet
- `6core.json`: 6个chiplet
- `8core.json`: 8个chiplet
- `10core.json`: 10个chiplet

## 🐛 故障排除

### GPU不可用
- 自动回退到CPU
- RTX 5090等新架构可能需要PyTorch nightly版本

### 内存不足
```python
# 在config.py中降低:
grid_resolution = 30  # 降低网格分辨率
num_episodes = 1000   # 减少训练轮数
```

### 训练太慢
```python
# 优化建议:
ppo_epochs = 2        # 减少PPO更新轮数
max_steps_per_episode = 100  # 减少最大步数
```

### 约束不满足
```python
# 增加惩罚权重:
adjacency_penalty = 1000.0  # 提高相邻约束惩罚
overlap_penalty = 2000.0    # 提高重叠惩罚
```

## 📚 下一步扩展

已实现的基础版本支持同构智能体。如需异构智能体支持:

1. 在 `ChipletNode` 中添加类型字段
2. 创建类型特定的策略网络
3. 实现类型感知的奖励函数
4. 添加亲和性约束

参考前面讨论的异构MARL设计。

## 🎉 完成清单

- ✅ 多智能体环境实现
- ✅ Actor-Critic网络
- ✅ MAPPO训练算法
- ✅ 相邻约束(min_overlap_length)
- ✅ 冲突检测和处理
- ✅ 有效动作过滤
- ✅ 可视化工具
- ✅ 测试脚本
- ✅ 文档和README

## 运行示例

```powershell
# 在PowerShell中
cd z:\work\placement\thermal-placement\src\MARL
python test_mappo.py  # 先测试

# 测试通过后训练
python train_mappo.py
```

预期输出将保存在 `output/MARL/` 目录下,包含每个测试用例的可视化结果。

祝训练顺利! 🚀

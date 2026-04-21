# HazeFlow: Revisit Haze Physical Model as ODE and Non-Homogeneous Haze Generation for Real-World Dehazing (ICCV2025)

> 📢 **本地复刻说明 (Local Fork Notice):**
> 本仓库为 [cloor/HazeFlow](https://github.com/cloor/HazeFlow) 的本地复刻版本。
> 
> 此项目**仅用于个人在本地进行代码修改、实验调优与版本管理**。
> 算法核心、研究成果及原始代码均归属于原作者团队。如需获取最新官方更新或提交 Issue，请前往[官方原始仓库](https://github.com/cloor/HazeFlow)。

<div style="display: flex; justify-content: space-between; align-items: baseline;">
  <h2 style="color: gray; margin: 0;">原作者团队 (Original Authors)</h2> 
</div>

<h3 style="margin-top: 0;">
  <a href="https://junsung6140.github.io/">Junseong Shin*</a>, <a href="https://cloor.github.io/">Seungwoo Chung*</a>, Yunjeong Yang, <a href="https://sites.google.com/view/lliger9">Tae Hyun Kim<sup>&#8224;</sup></a>
</h3>
<h4><sub><sup>(* 代表共同第一作者。  <sup>&#8224;</sup> 代表通讯作者。)</sup></sub></h4>

<p align="center">
  <img src="assets/ASM5.png" alt="hazeflow" width="800"/>
</p>

这是 ICCV2025 论文 "**HazeFlow**: Revisit Haze Physical Model as ODE and Non-Homogeneous Haze Generation for Real-World Dehazing" 的官方实现 [[论文链接]](https://arxiv.org/abs/2509.18190) / [[项目主页]](https://junsung6140.github.io/hazeflow/)。

## 📊 效果展示 (Results)
<p align="center">
  <img src="assets/result.png" alt="result" width="800"/>
</p>

更多定性（Qualitative）和定量（Quantitative）对比结果，请前往原作者的 [[项目主页]](https://junsung6140.github.io/hazeflow/) 查看。

## 📦 环境安装 (Installation)

你可以选择使用 pip 或 conda 来配置本地环境：

**使用 pip:**
```bash
git clone <你的本地/远程仓库地址>
cd HazeFlow
pip install -r requirements.txt

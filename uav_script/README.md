# 说明
改代码部署：妙算3 | Jetson Orin NX 16GB | JetPack 5.1.3 | Python 3.8
---
# 拉取代码
```bash
git clone -b uav-flow/orinNX/jpack5/py3.8 https://github.com/yirongjie/UAV-Isaac-GR00T.git
cd UAV-Isaac-GR00T
```

# 安装环境
安装pytorch 2.1.0
```bash
wget https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
pip3.8 install ./torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```
---
安装torchvision 0.16.1
```bash
git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision
vim torchvision/transforms/v2/functional/_geometry.py
```
修改`if (interpolation == InterpolationMode.BILINEAR and "AVX2" in torch.backends.cpu.get_cpu_capability()) or (`为`if (interpolation == InterpolationMode.BILINEAR and hasattr(torch.backends, "cpu") and "AVX2" in torch.backends.cpu.get_cpu_capability()) or (`
```bash
cd torchvision
export BUILD_VERSION=0.16.1
python3.8 setup.py install --user
```
---
安装flash-attn 1.0.9
```bash
# 指定 Orin 的计算能力 (SM 8.7)
export CMAKE_CUDA_ARCHITECTURES="87"
# 限制编译作业数，防止 OOM。
# 如果您是 16GB 内存版本，用 4；如果是 8GB 内存版本，请用 2
export MAX_JOBS=4
# 先安装一些编译依赖
pip3.8 install packaging ninja
pip3.8 install flash-attn==1.0.9
```
---
安装decord
```bash
git clone --recursive https://github.com/dmlc/decord
cd decord
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HOME/ffmpeg_build
make -j$(nproc)
cd ../python
python3 setup.py install --user
```
---
安装pytorch3d
```bash
export TORCH_CUDA_ARCH_LIST="8.7"
export CUB_HOME=/usr/local/cuda
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
python3 setup.py install --user
pip3.8 install flash-attn==1.0.9
```
---
安装其他python包
```bash
pip3 install -r ./requirement
pip3 install -e ./gr00t
```
---
# 运行

确保训练后的代码存放在`/open_app/UAV-Gr00t-004`中
```bash
python3.8 ./uav_script/main_uav_flow.py --model gr00t --local-gr00t --horizon 4
```
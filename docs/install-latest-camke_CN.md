# 安装最新版CMake

## Windows

直接前往[CMake官网](https://cmake.org/download/)下载安装器（.msi），并执行即可。安装器会自动安装CMake GUI

## Linux

1. 更新软件包并安装必要工具（如果没有安装）

```bash
sudo apt update
sudo apt install -y software-properties-common wget apt-transport-https ca-certificates gnupg
```

2. 下载并导入 Kitware 的 GPG 密钥，将密钥保存到 /usr/share/keyrings/ 目录

```bash
wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
```

3. 添加 Kitware APT 仓库

```bash
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
```

- 对于Ubuntu 22.04版本，请将以上命令中的“noble”替换为22.04的版本代号“jammy”

4. 安装CMake

```bash
sudo apt update && sudo apt install cmake cmake-qt-gui
```

## [撤销仓库添加的更改]

1. 删除 Kitware APT 仓库配置

```bash
sudo rm /etc/apt/sources.list.d/kitware.list
```

2. 删除 Kitware GPG 密钥

```bash
sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg
```

3. 恢复软件包列表

```bash
sudo apt update
```
#!/usr/bin/env python
"""
GIUSW GUI项目一键启动脚本
"""

import os
import subprocess
import sys

def main():
    """主函数"""
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 虚拟环境路径
    conda_env = r"E:\CondaEnvs\tf-gpu-env"
    
    # 构建命令
    command = f'conda activate "{conda_env}" && python GiUSW_CRC_V02.py'
    
    print(f"正在启动GIUSW GUI...")
    print(f"激活环境: {conda_env}")
    print(f"执行命令: {command}")
    print("=" * 60)
    
    # 执行命令并实时显示输出
    try:
        # 使用Popen执行命令，实时捕获输出
        process = subprocess.Popen(
            command,
            cwd=project_root,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
            text=True,
            bufsize=1  # 行缓冲
        )
        
        # 实时读取并打印输出
        for line in process.stdout:
            print(line, end='')
        
        # 等待进程完成
        process.wait()
        
        # 检查返回码
        if process.returncode != 0:
            print(f"\n启动失败 (返回码: {process.returncode})")
            sys.exit(1)
        
    except Exception as e:
        print(f"启动时发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
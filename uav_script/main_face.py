import time
import sys

# 假设您的无人机封装类保存在名为 'DJI_M4D_smol_wrapper.py' 的文件中
try:
    from DJI_M4D_smol_wrapper import DjiM4DDrone
except ImportError:
    print("错误: 未找到 'DJI_M4D_smol_wrapper.py' 文件。")
    print("请确保 main.py 与 DJI_M4D_smol_wrapper.py 在同一目录下。")
    sys.exit(1)

def main():
    """
    主执行函数：
    1. 初始化无人机。
    2. 起飞。
    3. 执行 move_to_person。
    4. 降落并关闭。
    """

    
    
    drone = None

    try:
        print("[Main] 正在初始化无人机连接...")
        drone = DjiM4DDrone()
        print("[Main] 连接成功。")

        # --- 1. 起飞 ---
        print("[Main] 正在起飞...")
        drone.talk("正在准备起飞")
        drone.take_off()
        print("[Main] 起飞完成。悬停 3 秒以稳定...")
        time.sleep(3) # 等待无人机稳定

        # --- 2. 执行 move_to_person ---
        # 目标：移动到 person_A 面前 2.5 米 (250 厘米) 处
        target_dist_cm = 150
        print(f"[Main] 开始执行 move_to_person，目标距离 {target_dist_cm} 厘米...")
        drone.talk("开始搜索目标人物")
        
        success = drone.move_to_person(
            target_distance_cm=target_dist_cm, 
            target_height_cm=0  # 保持与目标人物中心等高
        )

        if success:
            print(f"[Main] 成功移动到 person_A 面前。")
            drone.talk("已找到目标人物，任务完成。")
        else:
            print(f"[Main] 未能找到或移动到 person_A。")
            drone.talk("未找到目标人物，任务结束。")

        print("[Main] 任务执行完毕，准备降落。")
        time.sleep(2)

    except KeyboardInterrupt:
        print("\n[Main] 检测到用户中断 (Ctrl+C)。")
        if drone:
            drone.talk("任务已中断，正在紧急降落。")
            
    except Exception as e:
        print(f"\n[Main] 发生意外错误: {e}")
        if drone:
            drone.talk("系统发生错误，正在紧急降落。")
            
    finally:
        # --- 3. 降落并关闭 ---
        if drone:
            print("[Main] 正在执行降落程序...")
            drone.land()
            print("[Main] 正在关闭无人机连接...")
            drone.shutdown()
            print("[Main] 脚本执行完毕。")
        else:
            print("[Main] 无人机未成功初始化，退出。")

if __name__ == "__main__":
    main()
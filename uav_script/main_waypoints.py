import time
import sys
import os
from DJI_M4D_smol_wrapper import DjiM4DDrone
# 导入 KMZ 库中的 GPS 计算函数
from dji_kmz_mission_generator import calculate_new_gps

def main():
    """
    使用 DjiM4DDrone wrapper 执行一个完整的动态航线任务。
    
    流程:
    1. 初始化 Wrapper (连接 C++ PSDK Server)
    2. (N 挡) 调用 take_off()
    3. (N 挡) 获取当前 GPS，计算 3 个绝对坐标点
    4. (N 挡) 调用 fly_dynamic_kmz_mission_gps() (航线设置为 'hover')
    5. 等待用户按下 Ctrl+C
    6. (N 挡) 在 finally 块中调用 land()
    7. 关闭连接
    """
    
    print(f"--- 动态航线任务 (悬停 + 手动降落版) ---")
    print(f"!!! [重要] 请确保 C++ Server 正在运行 !!!")
    print(f"!!! [重要] 请确保遥控器 [全程] 保持在 [N 挡] !!!")
    
    drone = None
    try:
        # 1. 初始化 Wrapper
        print("正在初始化 DjiM4DDrone Wrapper...")
        drone = DjiM4DDrone()
        print("  -> Wrapper 初始化成功，已连接到 C++ Server。")

        # 2. 起飞
        print("\n[步骤 1/4] 正在发送 'take_off' 指令...")
        drone.take_off() # 此函数内部已包含原点设置
        
        print("  -> 起飞指令已接受。等待 2 秒让飞机升空并稳定...")
        time.sleep(2) 

        # 3. 获取当前 GPS 并规划 3 个“非正交”航点
        print("\n[步骤 2/4] 正在获取当前 GPS 位置以规划航线...")
        
        if not drone.origin_lat or not drone.origin_lon:
             pose = drone._get_realtime_pose()
             if not pose or pose.get('lat') == 0.0:
                 raise Exception("无法获取有效的 GPS 起始点。")
             base_lat = pose['lat']
             base_lon = pose['lon']
             base_alt = pose.get('alt_m', 50.0)
        else:
             base_lat = drone.origin_lat
             base_lon = drone.origin_lon
             base_alt = drone.origin_alt_m

        print(f"  -> 航线基准点: Lat={base_lat:.6f}, Lon={base_lon:.6f}, Alt={base_alt:.1f}m")

        # --- 规划 3 个绝对航点 ---
        lat_1, lon_1 = calculate_new_gps(base_lat, base_lon, 20.0, 20.0)
        lat_2, lon_2 = calculate_new_gps(lat_1, lon_1, -10.0, 30.0)
        lat_3, lon_3 = calculate_new_gps(lat_2, lon_2, 10.0, -10.0)

        mission_points_lla = [
            [lat_1, lon_1, base_alt], # [纬度, 经度, 高度]
            [lat_2, lon_2, base_alt],
            [lat_3, lon_3, base_alt]
        ]
        
        print(f"  -> 航点 1 (NE): Lat={lat_1:.6f}, Lon={lon_1:.6f}")
        print(f"  -> 航点 2 (SE): Lat={lat_2:.6f}, Lon={lon_2:.6f}")
        print(f"  -> 航点 3 (W) : Lat={lat_3:.6f}, Lon={lon_3:.6f}")

        # 4. 执行动态 KMZ 航线
        print("\n[步骤 3/4] 正在发送 'fly_dynamic_kmz_mission_gps' 指令...")
        
        success = drone.fly_dynamic_kmz_mission_gps(
            absolute_points_lla=mission_points_lla,
            use_current_pos_as_start=True # 自动将当前位置作为航点 0
        )
        
        if not success:
            raise Exception("发送 KMZ 任务失败，请检查 C++ Server 日志。")
        
        print("  -> 动态航线任务已成功发送。")
        
        # 5. [修改] 等待任务执行
        print("\n[步骤 4/4] --- 任务执行中 ---")
        print("--- 飞机将飞行 [起点 -> 1 -> 2 -> 3] ---")
        # print("--- 并在 航点 3 处 [原地悬停] (因为 finishAction=hover) ---")
        # print("--- 按下 [Ctrl+C] 来中断等待并执行降落。 ---")
        print("--- 飞机现在应该在最后一个航点 [原地悬停] ---")
        print("--- [重要] 程序将自动进入 'finally' 块执行降落并退出 ---")
        
        # # [修改] 恢复为 Ctrl+C 等待循环
        # while True:
        #     # drone._send_command("fc_vel 0 0 -100 0")
        #     time.sleep(1)
        time.sleep(5)

    except KeyboardInterrupt:
        print("\n[中断] 检测到 Ctrl+C。正在准备降落...")
    
    except Exception as e:
        print(f"\n[错误] 发生了一个错误: {e}")
        print("正在准备降落...")
        
    finally:
        # 6. [修改] 降落和清理
        if drone:
            print("\n[最后一步] 正在请求 C++ Server 降落 (N 挡)...")
            
            # [修改] 无论如何都必须调用 land()，因为飞机在悬停
            drone.land()
            # time.sleep(200)
            
            print("正在关闭 Wrapper 连接...")
            drone.shutdown()
            print("程序已退出。")
        else:
            print("\n[错误] Drone 对象未成功初始化，无法降落。")

if __name__ == "__main__":
    # 确保 dji_kmz_mission_generator.py 在路径中
    sys.path.append(os.path.dirname(__file__))
    main()
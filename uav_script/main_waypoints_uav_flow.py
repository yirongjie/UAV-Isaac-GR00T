# -*- coding: utf-8 -*-
# --- 全局配置 ---
# 设置为 True 以启用中文语音输入, 设置为 False 以使用键盘输入
ENABLE_SPEECH = False
# ------------------

import time
import argparse
import numpy as np
from PIL import Image
import threading
import cv2
import os
import datetime
import queue 
import typing 
import signal 
import sys    
import tty    
import termios 
from DJI_M4D_smol_wrapper import DjiM4DDrone

# 仅在启用语音时才需要这些模块
# try:
#     import speech
#     import instruction
# except ImportError:
#     if ENABLE_SPEECH:
#         print("错误: 启用语音失败。")
#         print("未找到 'speech' 或 'instruction' 模块。")
#         print("请确保 speech.py 和 instruction.py 与此脚本位于同一目录,")
#         print("或者将 ENABLE_SPEECH 设置为 False。")
#         exit(1)

# --- 导入 VLA 客户端 ---
from policy_client import OpenVLAClient, Gr00tClient, Gr00tLocalClient
# --- 导入航点 GPS 计算 ---
try:
    from dji_kmz_mission_generator import calculate_new_gps
except ImportError:
    print("错误: 未找到 'dji_kmz_mission_generator' 模块。")
    print("请确保 dji_kmz_mission_generator.py 与此脚本位于同一目录。")
    exit(1)


# --- 线程协调事件 ---
program_stop_event = threading.Event() # 用于 Ctrl+C 或 'q' 退出整个程序
round_interrupt_event = threading.Event() # 用于 'i' 中断当前轮次


# ==============================================================================
# 
# VLA (main_uav_flow.py) 的辅助函数
# 
# ==============================================================================

def get_vla_proprio(drone):
    """
    获取无人机当前姿态,并转换为VLA模型所需的proprio格式。
    (此函数未更改)
    """
    pose = drone.get_current_pose()
    
    proprio_x_m = pose['x']
    proprio_y_m = -pose['y'] # Tello y(左+) -> VLA y(Right+)
    proprio_z_m = pose['z']   # Tello z(上+) -> VLA z(Up+)
    
    yaw_deg_ccw = pose['yaw']
    proprio_yaw_deg = (yaw_deg_ccw + 180) % 360 - 180
    
    return np.array([proprio_x_m, proprio_y_m, proprio_z_m, proprio_yaw_deg])

def convert_openvla_poses_to_deltas(poses_frd_ccw_rad):
    """
    将 OpenVLA 的姿态序列 (相对于起点) 转换为 Tello 需要的增量序列。
    (此函数未更改)
    """
    deltas_frd_cw_deg = []
    last_pose_frd_ccw_deg = np.array([0.0, 0.0, 0.0, 0.0])
    
    for pose in poses_frd_ccw_rad:
        current_pose_frd_ccw_deg = np.array([
            pose[0], 
            pose[1], 
            pose[2], 
            np.degrees(pose[3])
        ])
        
        delta_frd_ccw_deg = current_pose_frd_ccw_deg - last_pose_frd_ccw_deg
        # VLA (F, R, D, CCW+) -> C++ fc_seq (F, R, D, CW+)
        delta_frd_cw_deg = delta_frd_ccw_deg
        delta_frd_cw_deg[3] = -delta_frd_ccw_deg[3] # Yaw CCW+ -> CW+
        
        deltas_frd_cw_deg.append(delta_frd_cw_deg)
        last_pose_frd_ccw_deg = current_pose_frd_ccw_deg
        
    return np.array(deltas_frd_cw_deg)


def llm_inference_wrapper(drone, client, instruction, args, 
                          first_image, 
                          out_data):
    """
    在一个单独的线程中运行 VLA 推理。
    (此函数未更改)
    """
    try:
        # 1. 获取当前状态 (图像 + 姿态)
        current_image_rgb = drone.get_frame()
        if current_image_rgb is None:
            print("[VLA-Thread] 警告: 无法获取当前图像。")
            out_data['error'] = "Failed to get frame"
            return
            
        current_image = Image.fromarray(current_image_rgb)
        proprio = get_vla_proprio(drone)
        
        # 2. 从 VLA 模型获取动作
        obs = {
            'first_image': first_image,
            'image': current_image,
            'proprio': proprio,
            'instr': instruction
        }
        
        print("[VLA-Thread] 请求 VLA 模型... (proprio: {})".format(proprio))
        t_start = time.time()
        response = client.get_action(obs)
        t_end = time.time()
        print("[VLA-Thread] VLA 模型响应时间: {:.2f}s".format(t_end - t_start))
        
        # 3. 解析动作
        deltas_frd_cw_deg = None
        if args.model == 'gr00t':
            deltas_frd_cw_deg = response.get('action_ori')
            if not deltas_frd_cw_deg:
                print("[VLA-Thread] 警告: GR00T 未返回 'action_ori'。")
                out_data['error'] = "No action_ori"
                return
        
        elif args.model == 'openvla':
            poses_frd_ccw_rad = response.get('action_ori')
            if not poses_frd_ccw_rad:
                print("[VLA-Thread] 警告: OpenVLA 未返回 'action_ori'。")
                out_data['error'] = "No action_ori"
                return
            deltas_frd_cw_deg = convert_openvla_poses_to_deltas(poses_frd_ccw_rad)

        # 4. 存储结果
        out_data['batch'] = np.array(deltas_frd_cw_deg)
        out_data['done'] = response.get('done', False)

    except Exception as e:
        print("[VLA-Thread] LLM 推理线程出错: {}".format(e))
        out_data['error'] = str(e)


# [!!] 关键修改 [!!]
# 添加了 'skip_takeoff' 参数
def main_vla_logic(drone, client, instruction, args, program_stop_event, round_interrupt_event, skip_takeoff=False):
    """
    VLA 控制的主逻辑 (流水线版本)。
    """
    
    llm_thread = None
    next_batch_data = {}
    current_batch = None
    first_image = None
    step_count = 0

    class RoundInterruptedError(Exception):
        pass

    try:
        # [!!] 关键修改 [!!]
        if not skip_takeoff:
            # --- 自动起飞 ---
            print("[VLA] 正在起飞...")
            # drone.talk("收到指令: {}。准备起飞。".format(instruction))
            drone.take_off()
            
            # 使睡眠可中断
            start_sleep = time.time()
            while time.time() - start_sleep < 5:
                if program_stop_event.is_set() or round_interrupt_event.is_set():
                    raise RoundInterruptedError("在起飞等待时中断")
                time.sleep(0.1) # 等待稳定
                
            print("[VLA] 无人机已起飞。")
        else:
            print("[VLA] 无人机已在空中，跳过起飞步骤。")
        
        
        # --- VLA 持续控制循环 ---
        print("[VLA] 开始执行 VLA 指令: {}".format(instruction))

        # 1. 获取第一帧图像 (用于 VLA)
        first_image_rgb = drone.get_frame()
        while first_image_rgb is None and not (program_stop_event.is_set() or round_interrupt_event.is_set()):
                print("[VLA] 警告: 无法获取第一帧图像，正在重试...")
                if program_stop_event.wait(0.5): break 
                if round_interrupt_event.wait(0.01): break 
                first_image_rgb = drone.get_frame()
        
        if program_stop_event.is_set() or round_interrupt_event.is_set():
                raise RoundInterruptedError("在获取第一帧图像时程序被终止")

        first_image = Image.fromarray(first_image_rgb)
        
        # 2. *第一次* LLM 推理 (阻塞)
        print("[VLA] 正在执行*第一次* LLM 推理 (阻塞)...")
        llm_inference_wrapper(drone, client, instruction, args, first_image, next_batch_data)
        
        if program_stop_event.is_set() or round_interrupt_event.is_set():
                raise RoundInterruptedError("在第一次推理后中断")
                
        current_batch = next_batch_data.get('batch')
        if current_batch is None:
            print("[VLA] 第一次 LLM 推理失败: {}".format(next_batch_data.get('error')))
            raise Exception("Initial LLM inference failed")
        
        while not (program_stop_event.is_set() or round_interrupt_event.is_set()):
            if current_batch is None or len(current_batch) == 0:
                print("[VLA] 未收到有效动作，任务终止。")
                break

            # 3. 拆分批次
            if args.extra_horizon == 0:
                main_steps = current_batch
                buffer_steps = current_batch
            else:
                main_steps = current_batch[:-args.extra_horizon]
                buffer_steps = current_batch[-args.extra_horizon:]
            
            print("[VLA] 收到 {} 步. 拆分为 {} (main) + {} (buffer)。".format(len(current_batch), len(main_steps), len(buffer_steps)))

            # 4. 执行主要步骤
            if len(main_steps) > 0:
                print("[VLA] 正在执行 {} 步主要动作...".format(len(main_steps)))
                drone.move_by_delta_pose_sequence(
                    main_steps, 
                    speed=args.speed
                )
            
            if program_stop_event.is_set() or round_interrupt_event.is_set():
                raise RoundInterruptedError("在主要步骤执行期间中断")

            # 5. 等待上一个 LLM 线程 (如果存在)
            if llm_thread is not None:
                print("[VLA] 等待上一个 LLM 线程完成...")
                llm_thread.join() 
                
                if program_stop_event.is_set() or round_interrupt_event.is_set():
                    raise RoundInterruptedError("在等待 LLM 线程时中断")
                
                current_batch = next_batch_data.get('batch')
                
                if next_batch_data.get('done', False):
                    print("[VLA] LLM 报告任务完成。")
                    drone.talk("任务完成")
                    if len(buffer_steps) > 0:
                        print("[VLA] 正在执行最后的 {} 步缓冲区动作...".format(len(buffer_steps)))
                        drone.move_by_delta_pose_sequence(buffer_steps, speed=args.speed)
                    break 
                
                if current_batch is None:
                    print("[VLA] 下一批 LLM 推理失败: {}".format(next_batch_data.get('error')))
                    break

            # 6. 启动 *新的* LLM 推理线程 (并行)
            print("[VLA] 启动下一次 LLM 推理 (在后台)...")
            next_batch_data = {} 
            llm_thread = threading.Thread(
                target=llm_inference_wrapper, 
                args=(drone, client, instruction, args, first_image, next_batch_data),
                daemon=True
            )
            llm_thread.start()

            # 7. 执行缓冲区步骤 (在 LLM 思考时)
            if len(buffer_steps) > 0:
                print("[VLA] 正在执行 {} 步缓冲区动作 (LLM 正在并行计算)...".format(len(buffer_steps)))
                drone.move_by_delta_pose_sequence(
                    buffer_steps, 
                    speed=args.speed
                )

            if program_stop_event.is_set() or round_interrupt_event.is_set():
                raise RoundInterruptedError("在缓冲区步骤执行期间中断")
            
            step_count += 1
            if args.max_steps is not None and step_count >= args.max_steps:
                print("[VLA] 已达到 {} 步最大限制，任务终止。".format(args.max_steps))
                drone.talk("达到最大步数，任务停止")
                break
            
    except (KeyboardInterrupt, RoundInterruptedError, InterruptedError) as e:
        print("\n[VLA] 检测到中断。停止当前VLA任务... ({})".format(type(e).__name__))
        if not (program_stop_event.is_set() or round_interrupt_event.is_set()):
             drone.talk("任务已中断")
        drone._send_command("fc_vel 0 0 0 0") 
    
    except Exception as e:
        print("[VLA] VLA 控制循环出错: {}".format(e))
        
    finally:
        if llm_thread is not None and llm_thread.is_alive():
            print("[VLA] 正在等待最后的 LLM 线程退出...")
            llm_thread.join()
            
        print("[VLA] VLA 任务结束。")
        
        # [!!] 关键修改 [!!]
        # VLA 任务结束后不再自动降落，而是返回主循环等待新指令。
        # 只有在 'q' 或 'i' 或 Ctrl+C 时才会触发外部的降落。
        
        try:
            if program_stop_event.is_set(): 
               print("[VLA] 收到程序退出信号，将由主 'finally' 块处理降落。")
            elif round_interrupt_event.is_set():
               print("[VLA] 收到轮次中断信号 ('i')，将由主 'finally' 块处理降落。")
            else:
               print("[VLA] 任务正常完成，在空中悬停。")
            
        except Exception as e:
            print("[VLA] VLA 线程结束时出错: {}".format(e))
                
        print("[VLA] VLA 线程已完成。")


def get_instruction_from_user():
    """
    根据 ENABLE_SPEECH 标志获取用户指令。
    (此函数未更改)
    """
    instruction_text = ""
    # if ENABLE_SPEECH:
    #     print("\n语音输入已启用。请说话 (或按 Ctrl+C 退出)...")
    #     inst = speech.record_and_get_text()
    #     inst = instruction.get_inst(inst)
    #     instruction_text = inst
    #     print("识别到的指令: {}".format(instruction_text))
    # else:
    print("\n语音输入已禁用。")
    instruction_text = input("请输入VLA指令 (输入 'exit' 或空指令以取消): ")
    
    return instruction_text

def on_program_exit(sig, frame):
    """Ctrl+C 按下时的回调"""
    if not program_stop_event.is_set():
        print("\n[Main] 检测到 Ctrl+C！将退出所有进程...")
        program_stop_event.set()
        round_interrupt_event.set() # 同时也中断当前轮


# ==============================================================================
# 
# 航点 (main_waypoints_uav_flow.py) 的逻辑
# 
# ==============================================================================

def run_waypoint_mission(drone):
    """
    执行航点任务，起飞并飞行，结束后悬停。
    假定无人机已被初始化，但尚未起飞。
    """
    
    print(f"\n--- [阶段 1: 动态航线任务] ---")
    print(f"!!! [重要] 请确保遥控器 [全程] 保持在 [N 挡] !!!")
    
    try:
        # 1. 起飞
        print("\n[航点] [步骤 1/4] 正在发送 'take_off' 指令...")
        drone.take_off() # 此函数内部已包含原点设置
        
        print("  -> 起飞指令已接受。等待 2 秒让飞机升空并稳定...")
        time.sleep(2) 

        # 2. 获取当前 GPS 并规划 3 个“非正交”航点
        print("\n[航点] [步骤 2/4] 正在获取当前 GPS 位置以规划航线...")
        
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
        print("\n[航点] [步骤 3/4] 正在发送 'fly_dynamic_kmz_mission_gps' 指令...")
        
        success = drone.fly_dynamic_kmz_mission_gps(
            absolute_points_lla=mission_points_lla,
            use_current_pos_as_start=True # 自动将当前位置作为航点 0
        )
        
        if not success:
            raise Exception("发送 KMZ 任务失败，请检查 C++ Server 日志。")
        
        print("  -> 动态航线任务已成功发送。")
        
        # 5. 等待任务执行
        print("\n[航点] [步骤 4/4] --- 任务执行中 ---")
        print("--- 飞机将飞行 [起点 -> 1 -> 2 -> 3] ---")
        print("--- 飞机现在应该在最后一个航点 [原地悬停] ---")
        
        # [!!] 关键修改 [!!]
        # 移除 'finally' 块中的 land() 和 shutdown()
        # 增加一个短暂的等待，以确保任务已开始执行或完成
        # 注意：这只是一个简单的等待。在实际应用中，您可能需要
        # C++ Server 返回一个 "任务完成" 的状态。
        # 这里我们假设5秒后任务已完成或飞机在最后一个点悬停。
        time.sleep(5) 
        
        print(f"--- [阶段 1: 航点任务完成] ---")
        print(f"--- 飞机正在悬停，准备进入VLA模式 ---")


    except Exception as e:
        print(f"\n[航点错误] 航点任务阶段发生错误: {e}")
        print("无法继续 VLA 任务。将尝试降落...")
        if drone:
            # 尝试在航点阶段失败时降落
            try:
                drone.land()
            except Exception as e_land:
                print(f"航点阶段失败后尝试降落也失败: {e_land}")
        # 重新引发异常，以便被主 finally 块捕获
        raise e


# ==============================================================================
# 
# 主执行流程 (合并)
# 
# ==============================================================================

if __name__ == "__main__":
    
    # 确保 dji_kmz_mission_generator.py 在路径中
    sys.path.append(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description="使用 VLA 模型控制 Tello/M4D 无人机")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=['gr00t', 'openvla'], 
        required=True, 
        help="要使用的VLA模型 ('gr00t' 或 'openvla')"
    )
    parser.add_argument(
        "--ip", 
        type=str, 
        default="localhost", 
        help="VLA 模型服务器的 IP 地址"
    )
    parser.add_argument(
        "--horizon", 
        type=int, 
        default=4, 
        help="VLA 模型的总预测时域 (例如 8)"
    )
    parser.add_argument(
        "--speed", 
        type=int, 
        default=40,
        help="无人机执行 VLA 动作的速度 (cm/s)"
    )
    parser.add_argument(
        "-m", 
        '--max_steps', 
        type=int, 
        default=100, 
        help='最大VLA *批次* 数 (默认: 100)'
    )
    parser.add_argument(
        "-e", 
        '--extra_horizon', 
        type=int, 
        default=0, 
        help='额外飞行几个Horizon (默认: 0)'
    )
    parser.add_argument(
        "--local-gr00t",
        action="store_true",
        help="Muat dan jalankan model GR00T secara lokal (hanya berlaku jika --model=gr00t)"
    )
    
    args = parser.parse_args()
    
    drone = None
    client = None
    vla_thread = None 
    
    # 保存旧的终端设置
    old_settings = termios.tcgetattr(sys.stdin)

    # 注册 Ctrl+C (SIGINT) 处理器
    signal.signal(signal.SIGINT, on_program_exit)

    try:
        # 1. [!!] 初始化无人机 (只执行一次)
        print("正在初始化 无人机 (DjiM4DDrone)...")
        drone = DjiM4DDrone()
        print("  -> Wrapper 初始化成功，已连接到 C++ Server。")

        # 2. [!!] 阶段 1: 运行航点任务
        #     此函数将处理起飞、飞行航点，并在最后悬停
        run_waypoint_mission(drone)

        # 3. [!!] 阶段 2: 初始化 VLA 客户端
        print("\n--- [阶段 2: VLA 交互式控制] ---")
        client = None
        total_horizon = args.horizon + args.extra_horizon

        if args.model == 'gr00t':
            if args.local_gr00t:
                print("Memuat model GR00T secara LOKAL... (Horizon: {})".format(total_horizon))
                client = Gr00tLocalClient(horizon=total_horizon)
            else:
                port = 5555
                print("Menghubungkan ke server GR00T JARAK JAUH: {}:{} (Horizon: {})".format(args.ip, port, total_horizon))
                client = Gr00tClient(ip=args.ip, port=port, horizon=total_horizon)
        
        elif args.model == 'openvla':
            port = 5007
            print("Menghubungkan ke OpenVLA 客户端: {}:{}".format(args.ip, port))
            client = OpenVLAClient(ip=args.ip, port=port)
        
        if client is None:
            raise ValueError("Klien VLA tidak dapat diinisialisasi.")
        
        # 4. [!!] 阶段 2: 开始 VLA 主事件循环 (多轮次)
        
        # 设置终端为 cbreak 模式
        tty.setcbreak(sys.stdin.fileno())
        print("\n--- [VLA 事件循环已启动] ---")
        print("飞机已在空中，准备接收VLA指令。")
        print("按 'n' 开始新任务 (New Task)")
        print("按 'i' 中断当前任务 (Interrupt)")
        print("按 'q' 退出程序 (Quit)")
        print("按 'l' 强制降落 (Land)")
        print("--------------------------")
        
        while not program_stop_event.is_set():
            # 检查 VLA 线程是否已结束
            if vla_thread and not vla_thread.is_alive():
                vla_thread.join()
                vla_thread = None
                print("\n[Main] VLA 任务线程已结束。")
                print("按 'n' 开始新任务, 'i' 中断, 'q' 退出, 'l' 降落")

            if program_stop_event.wait(0.1): # 0.1秒的超时
                break # 程序被要求停止
            
            char = sys.stdin.read(1)
            
            if not char or program_stop_event.is_set():
                continue

            # (n) 开始新任务
            if char == 'n':
                if vla_thread and vla_thread.is_alive():
                    print("\n[Main] 错误: 任务已在运行。请先按 'i' 中断。")
                else:
                    # 关键: 运行 input() 前恢复终端
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    
                    instruction_text = ""
                    try:
                        instruction_text = get_instruction_from_user()
                    except (KeyboardInterrupt, EOFError):
                        program_stop_event.set()
                    
                    # 关键: 恢复 cbreak 模式
                    tty.setcbreak(sys.stdin.fileno())

                    if instruction_text and instruction_text.lower() != 'exit':
                        print("[Main] 收到新指令，启动VLA线程...")
                        round_interrupt_event.clear()
                        vla_thread = threading.Thread(
                            target=main_vla_logic,
                            args=(drone, client, instruction_text, args,
                                  program_stop_event, round_interrupt_event, 
                                  True), # [!!] 关键: 传递 skip_takeoff=True
                            daemon=True
                        )
                        vla_thread.start()
                    else:
                        print("[Main] 任务已取消。")
                        print("按 'n' 开始新任务, 'i' 中断, 'q' 退出, 'l' 降落")

            # (i) 中断当前任务
            elif char == 'i':
                if vla_thread and vla_thread.is_alive():
                    print("\n[Main] 检测到 'i'！将中断当前轮次并降落...")
                    round_interrupt_event.set()
                    # 线程将在其 finally 块中处理中断
                    # 主 finally 块将等待它 join 并执行降落
                else:
                    print("\n[Main] 没有正在运行的任务可以中断。")

            # (l) 强制降落
            elif char == 'l':
                print("\n[Main] 检测到 'l'！强制降落...")
                drone.land()
                # 降落后，飞机仍在连接状态，可以再次起飞
                # 如果希望 'l' 退出，可以添加 program_stop_event.set()

            # (q) 退出程序
            elif char == 'q':
                print("\n[Main] 检测到 'q'！退出程序...")
                program_stop_event.set()
                round_interrupt_event.set() # 确保VLA线程也停止
                break # 退出 while 循环

    except (KeyboardInterrupt, ValueError, Exception) as e:
        if isinstance(e, (KeyboardInterrupt)):
            print("\n[Main] 检测到初始化中断...")
        elif isinstance(e, (ValueError)):
             print(e)
        else:
            print("[Main] 程序主线程发生未捕獲异常: {}".format(e))
        
        program_stop_event.set() # 确保设置了停止标志
        
    finally:
        # 5. [!!] 统一的清理程序
        
        print("\n[Main] 正在关闭所有线程和连接...")
        program_stop_event.set() # 确保所有循环都停止
        
        # 关键: 恢复终端设置
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        if vla_thread and vla_thread.is_alive():
            print("[Main] 等待 VLA 线程结束...")
            vla_thread.join()
            
        if drone:
            print("[Main] 主线程安全检查：正在执行最后降落...")
            try:
                drone.land()
            except Exception as e:
                print("[Main] 主线程安全降落失败: {}".format(e))
            
            print("[Main] 正在关闭 Wrapper 连接...")
            drone.shutdown()
           
        print("[Main] 程序已退出。")
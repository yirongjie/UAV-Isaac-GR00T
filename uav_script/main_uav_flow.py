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
# import dotenv
import queue # [!!] 确保导入
import typing # Este import não é mais estritamente necessário, mas inofensivo

# 仅在启用语音时才需要这些模块
# (此部分保持不变)
try:
    import speech
    import instruction
except ImportError:
    if ENABLE_SPEECH:
        print("错误: 启用语音失败。")
        print("未找到 'speech' 或 'instruction' 模块。")
        print("请确保 speech.py 和 instruction.py 与此脚本位于同一目录,")
        print("或者将 ENABLE_SPEECH 设置为 False。")
        exit(1)

# dotenv.load_dotenv()

DRONE_TYPE = "djim4d"
# ssh -L 5555:localhost:5555 yrj@10.29.230.87
# ssh -L 5007:localhost:5007 yrj@10.109.246.210 -p 20212

# [!! PERUBAHAN !!] Impor Gr00tLocalClient yang baru
from policy_client import OpenVLAClient, Gr00tClient, Gr00tLocalClient
# from tello_smol_wrapper import Drone
from DJI_M4D_smol_wrapper import DjiM4DDrone

# --- 线程协调事件 ---
stop_event = threading.Event()

# [!!] (Fungsi 'keep_tello_alive', 'drone_live_feed', 'get_vla_proprio', 'convert_openvla_poses_to_deltas', dan 'llm_inference_wrapper' tetap TIDAK BERUBAH)
# ...
# (Menyalin fungsi-fungsi yang tidak berubah untuk kelengkapan)
# ...

# def keep_tello_alive(drone_instance: Drone):
#     """定期发送命令防止 Tello 自动降落。"""
#     if DRONE_TYPE != "tello":
#         print("[Keepalive] M4D 无人机不需要 Keepalive 线程。线程退出。")
#         return
        
#     while not stop_event.is_set():
#         try:
#             battery = drone_instance.get_battery()
#             print("[Keepalive] Battery: {}%".format(battery))
#         except Exception as e:
#             print("[Keepalive] 错误: {}".format(e))
#         stop_event.wait(8)
#     print("[Keepalive] 线程已停止。")


def drone_live_feed(drone_instance):
    """
    为实体无人机显示实时视频流的线程，并保存为 mp4 文件。
    """
    print("[LiveFeed] 正在启动视频流...")
    log_dir = "logs_vla"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 修改后的版本
    video_path = os.path.join(log_dir, "{}_VLA_Feed.mp4".format(ts))
    video_writer = None

    EXPECTED_W = 480
    EXPECTED_H = 360
    
    try:
        while not stop_event.is_set():
            frame_rgb = drone_instance.get_frame() 
            if frame_rgb is None or frame_rgb.size == 0:
                print("[LiveFeed] 未能获取到有效的视频帧，跳过...")
                stop_event.wait(0.1)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            h, w = frame_bgr.shape[:2]
            if w != EXPECTED_W or h != EXPECTED_H:
                frame_bgr = cv2.resize(frame_bgr, (EXPECTED_W, EXPECTED_H))

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, 30, (EXPECTED_W, EXPECTED_H))
            
            video_writer.write(frame_bgr)
            cv2.imshow("Drone Live Feed (VLA)", frame_bgr)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                print("[LiveFeed] 用户通过 'q' 键停止了视频流")
                stop_event.set()
                break

    except Exception as e:
        print(u"[LiveFeed] 视频流发生错误: {}".format(e))
    finally:
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            print(u"[LiveFeed] 视频已保存到: {}".format(video_path))
        print("[LiveFeed] 视频流线程已终止。")
        stop_event.set()


def get_vla_proprio(drone):
    """
    获取无人机当前姿态,并转换为VLA模型所需的proprio格式。
    """
    pose = drone.get_current_pose()
    
    proprio_x_m = pose['x']
    proprio_y_m = -pose['y'] # Tello y(左+) -> VLA y(Right+)
    proprio_z_m = pose['z']   # Tello z(上+) -> VLA z(Up+)
    
    yaw_deg_ccw = pose['yaw']
    proprio_yaw_deg = (yaw_deg_ccw + 180) % 360 - 180
    
    return np.array([proprio_x_m, proprio_y_m, proprio_z_m, proprio_yaw_deg])

# [!!] MODIFICADO: Anotações de tipo removidas
def convert_openvla_poses_to_deltas(poses_frd_ccw_rad):
    """
    将 OpenVLA 的姿态序列 (相对于起点) 转换为 Tello 需要的增量序列。
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


# [!!] MODIFICADO: Anotações de tipo removidas
def llm_inference_wrapper(drone, client, instruction, args, 
                          first_image, 
                          out_data):
    """
    在一个单独的线程中运行 VLA 推理。
    获取当前帧, 调用 client.get_action(), 并将结果放入 out_data 字典中。
    (Fungsi ini tidak perlu diubah karena 'client' adalah abstraksi
    yang berfungsi untuk Gr00tClient, Gr00tLocalClient, dan OpenVLAClient)
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
        
        # [!!] MODIFIED: Replaced f-string with .format()
        print("[VLA-Thread] 请求 VLA 模型... (proprio: {})".format(proprio))
        t_start = time.time()
        # [!!] PANGGILAN KLIEN ABSTRAK: Ini berfungsi untuk klien LOKAL dan JARAK JAUH
        response = client.get_action(obs)
        t_end = time.time()
        # [!!] MODIFIED: Replaced f-string with .format()
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
        # [!!] MODIFIED: Replaced f-string with .format()
        print("[VLA-Thread] LLM 推理线程出错: {}".format(e))
        out_data['error'] = str(e)


# [!!] MODIFICADO: Anotações de tipo removidas
def main_vla_logic(drone, client, instruction, args):
    """
    VLA 控制的主逻辑 (流水线版本)，在一个单独的线程中运行。
    (Fungsi ini juga não perlu diubah)
    """
    
    llm_thread = None
    next_batch_data = {}
    current_batch = None
    first_image = None
    step_count = 0

    try:
        # --- 自动起飞 ---
        print("[VLA] 正在起飞...")
        # [!!] MODIFIED: Replaced f-string with .format() (in commented line)
        # drone.talk("收到指令: {}。准备起飞。".format(instruction))
        drone.take_off()
        time.sleep(5) # 等待稳定
        print("[VLA] 无人机已起飞。")
        
        # --- VLA 持续控制循环 ---
        # [!!] MODIFIED: Replaced f-string with .format()
        print("[VLA] 开始执行 VLA 指令: {}".format(instruction))

        # 1. [!!] 获取第一帧图像 (用于 VLA)
        first_image_rgb = drone.get_frame()
        while first_image_rgb is None and not stop_event.is_set():
                print("[VLA] 警告: 无法获取第一帧图像，正在重试...")
                stop_event.wait(0.5)
                first_image_rgb = drone.get_frame()
        
        if stop_event.is_set():
                raise InterruptedError("在获取第一帧图像时程序被终止")

        first_image = Image.fromarray(first_image_rgb)
        
        # 2. [!!] *第一次* LLM 推理 (阻塞)
        print("[VLA] 正在执行*第一次* LLM 推理 (阻塞)...")
        llm_inference_wrapper(drone, client, instruction, args, first_image, next_batch_data)
        current_batch = next_batch_data.get('batch')
        if current_batch is None:
            # [!!] MODIFIED: Replaced f-string with .format()
            print("[VLA] 第一次 LLM 推理失败: {}".format(next_batch_data.get('error')))
            raise Exception("Initial LLM inference failed")
        
        while not stop_event.is_set():
            if current_batch is None or len(current_batch) == 0:
                print("[VLA] 未收到有效动作，任务终止。")
                break

            # 3. [!!] 拆分批次
            if args.extra_horizon == 0:
                main_steps = current_batch
                buffer_steps = current_batch
            else:
                main_steps = current_batch[:-args.extra_horizon]
                buffer_steps = current_batch[-args.extra_horizon:]
            
            # [!!] MODIFIED: Replaced f-string with .format()
            print("[VLA] 收到 {} 步. 拆分为 {} (main) + {} (buffer)。".format(len(current_batch), len(main_steps), len(buffer_steps)))

            # 4. [!!] 执行主要步骤
            if len(main_steps) > 0:
                # [!!] MODIFIED: Replaced f-string with .format()
                print("[VLA] 正在执行 {} 步主要动作...".format(len(main_steps)))
                drone.move_by_delta_pose_sequence(
                    main_steps, 
                    speed=args.speed
                )
            
            # 5. [!!] 等待上一个 LLM 线程 (如果存在)
            if llm_thread is not None:
                print("[VLA] 等待上一个 LLM 线程完成...")
                llm_thread.join()
                current_batch = next_batch_data.get('batch')
                
                if next_batch_data.get('done', False):
                    print("[VLA] LLM 报告任务完成。")
                    drone.talk("任务完成")
                    if len(buffer_steps) > 0:
                        # [!!] MODIFIED: Replaced f-string with .format()
                        print("[VLA] 正在执行最后的 {} 步缓冲区动作...".format(len(buffer_steps)))
                        drone.move_by_delta_pose_sequence(buffer_steps, speed=args.speed)
                    break 
                
                if current_batch is None:
                    # [!!] MODIFIED: Replaced f-string with .format()
                    print("[VLA] 下一批 LLM 推理失败: {}".format(next_batch_data.get('error')))
                    break

            # 6. [!!] 启动 *新的* LLM 推理线程 (并行)
            print("[VLA] 启动下一次 LLM 推理 (在后台)...")
            next_batch_data = {} # 清空数据
            llm_thread = threading.Thread(
                target=llm_inference_wrapper, 
                args=(drone, client, instruction, args, first_image, next_batch_data),
                daemon=True
            )
            llm_thread.start()

            # 7. [!!] 执行缓冲区步骤 (在 LLM 思考时)
            if len(buffer_steps) > 0:
                # [!!] MODIFIED: Replaced f-string with .format()
                print("[VLA] 正在执行 {} 步缓冲区动作 (LLM 正在并行计算)...".format(len(buffer_steps)))
                drone.move_by_delta_pose_sequence(
                    buffer_steps, 
                    speed=args.speed
                )
            
            step_count += 1
            if args.max_steps is not None and step_count >= args.max_steps:
                # [!!] MODIFIED: Replaced f-string with .format()
                print("[VLA] 已达到 {} 步最大限制，任务终止。".format(args.max_steps))
                drone.talk("达到最大步数，任务停止")
                break
            
    except (KeyboardInterrupt, InterruptedError):
        print("\n[VLA] 检测到中断。停止当前VLA任务...")
        drone.talk("任务已中断")
        drone._send_command("fc_vel 0 0 0 0") 
    
    except Exception as e:
        # [!!] MODIFIED: Replaced f-string with .format()
        print("[VLA] VLA 控制循环出错: {}".format(e))
        
    finally:
        if llm_thread is not None and llm_thread.is_alive():
            print("[VLA] 正在等待最后的 LLM 线程退出...")
            llm_thread.join()
            
        print("[VLA] VLA 任务结束，正在安全降落...")
        
        try:
            if drone.get_current_pose()['z'] > 10: 
               drone.land()
            
        except Exception as e:
            # [!!] MODIFIED: Replaced f-string with .format()
            print("[VLA] 降落时检查高度失败: {}，尝试强制降落...".format(e))
            try:
                drone.land()
            except:
                pass
                
        print("[VLA] VLA 线程已完成。")
        stop_event.set()


if __name__ == "__main__":
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
        help="VLA 模型服务器的 IP 地址 (jika menggunakan klien jarak jauh)"
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
        default=80,
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
    
    # [!! BARU !!] Argumen baru untuk mengontrol inferensi lokal
    parser.add_argument(
        "--local-gr00t",
        action="store_true",
        help="Muat dan jalankan model GR00T secara lokal alih-alih terhubung ke server (hanya berlaku jika --model=gr00t)"
    )
    
    args = parser.parse_args()
    
    vla_thread = None
    keepalive_thread = None
    drone = None

    try:
        # 1. 初始化无人机
        print("正在初始化 无人机...")
        drone = DjiM4DDrone()

        # 2. [!! DIPERBARUI !!] Inisialisasi klien berdasarkan model DAN mode (lokal/jarak jauh)
        client = None
        total_horizon = args.horizon + args.extra_horizon

        if args.model == 'gr00t':
            if args.local_gr00t:
                # Mode Lokal: Muat model secara langsung
                # [!!] MODIFIED: Replaced f-string with .format()
                print("Memuat model GR00T secara LOKAL... (Horizon: {})".format(total_horizon))
                client = Gr00tLocalClient(horizon=total_horizon)
            else:
                # Mode Jarak Jauh: Terhubung ke server
                port = 5555
                # [!!] MODIFIED: Replaced f-string with .format()
                print("Menghubungkan ke server GR00T JARAK JAUH: {}:{} (Horizon: {})".format(args.ip, port, total_horizon))
                client = Gr00tClient(ip=args.ip, port=port, horizon=total_horizon)
        
        elif args.model == 'openvla':
            port = 5007
            # [!!] MODIFIED: Replaced f-string with .format()
            print("Menghubungkan ke OpenVLA 客户端: {}:{}".format(args.ip, port))
            client = OpenVLAClient(ip=args.ip, port=port)
        
        if client is None:
            raise ValueError("Klien VLA tidak dapat diinisialisasi.")

        # 3. 根据 ENABLE_SPEECH 标志获取指令
        instruction_text = ""
        if ENABLE_SPEECH:
            print("语音输入已启用。请说话...")
            inst = speech.record_and_get_text()
            inst = instruction.get_inst(inst)
            instruction_text = inst
            # [!!] MODIFIED: Replaced f-string with .format()
            print("识别到的指令: {}".format(instruction_text))
        else:
            print("语音输入已禁用。")
            instruction_text = input("请输入要执行的VLA指令: ")

        if not instruction_text:
            raise ValueError("未输入指令，退出程序。")

        # 4. 启动后台线程
        # vla_thread = threading.Thread(target=main_vla_logic, args=(drone, client, instruction_text, args), daemon=True)
        
        # # if DRONE_TYPE == "tello":
        # #     keepalive_thread = threading.Thread(target=keep_tello_alive, args=(drone,), daemon=True)
        # #     print("启动 Keepalive 线程...")
        # #     keepalive_thread.start()
        
        # print("启动 VLA 主逻辑线程...")
        # vla_thread.start()

        # # 5. 在主线程中运行视频流
        # drone_live_feed(drone)
        main_vla_logic(drone, client, instruction_text, args)

    except (KeyboardInterrupt, ValueError) as e:
        if isinstance(e, ValueError):
            print(e)
        else:
            print("\n检测到用户中断 (Ctrl+C)，正在关闭程序...")
    
    except Exception as e:
        # [!!] MODIFIED: Replaced f-string with .format()
        print("程序主线程发生未捕獲异常: {}".format(e))
        
    finally:
        print("正在关闭所有线程...")
        stop_event.set()
        
        if keepalive_thread and keepalive_thread.is_alive():
            keepalive_thread.join()
            print("Keepalive 线程已加入。")
            
        if vla_thread and vla_thread.is_alive():
            vla_thread.join()
            print("VLA 线程已加入。")
            
        if drone:
            print("主线程安全检查：正在执行最后降落...")
            try:
                drone.land()
            except Exception as e:
                # [!!] MODIFIED: Replaced f-string with .format()
                print("主线程安全降落失败: {}".format(e))
           
        print("程序已退出。")
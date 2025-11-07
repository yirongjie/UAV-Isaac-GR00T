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

import signal # [!!] 用于 Ctrl+C
import sys    # [!!] 新增：用于 tty/termios
import tty    # [!!] 新增：用于终端控制
import termios # [!!] 新增：用于终端控制

# [!!] 移除: pynput (导致崩溃)
# from pynput import keyboard 

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

from policy_client import OpenVLAClient, Gr00tClient, Gr00tLocalClient
from DJI_M4D_smol_wrapper import DjiM4DDrone

# --- 线程协调事件 ---
program_stop_event = threading.Event() # 用于 Ctrl+C 或 'q' 退出整个程序
round_interrupt_event = threading.Event() # 用于 'i' 中断当前轮次


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

# [!!] MODIFICADO: Anotações de tipo removidas
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

# (在 main_uav_flow.py 文件中，例如在 convert_openvla_poses_to_deltas 函数之后)

def _is_action_batch_small(batch: np.ndarray, threshold: float = 1.0) -> bool:
    """
    检查整个动作批次中的所有增量是否都在 [-threshold, threshold] cm/deg 之间。
    
    Args:
        batch (np.ndarray): N x 4 的动作增量数组 (cm, deg)。
        threshold (float): 允许的最大绝对值 (厘米/度)。
        
    Returns:
        bool: 如果所有值都小于或等于阈值，则返回 True。
    """
    if batch.size == 0:
        return True # 空批次视为小动作
        
    # 检查批次中所有元素的绝对值是否都小于等于阈值
    # np.max(np.abs(batch)) 返回批次中绝对值最大的那个元素
    max_delta = np.max(np.abs(batch))
    
    if max_delta <= threshold:
        print(f"[VLA] 动作批次判断: 最大增量 {max_delta:.2f}cm/deg <= 阈值 {threshold:.1f}cm/deg。判定为小动作。")
        return True
    else:
        # print(f"[VLA] 动作批次判断: 最大增量 {max_delta:.2f}cm/deg > 阈值 {threshold:.1f}cm/deg。判定为大动作。")
        return False

# [!!] MODIFICADO: Anotações de tipo removidas
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


# [!!] MODIFICADO: 签名已更改以接受事件
def main_vla_logic(drone, client, instruction, args, program_stop_event, round_interrupt_event):
    """
    VLA 控制的主逻辑 (流水线版本)。
    [!!] 现在检查 program_stop_event (Ctrl+C / 'q') 和 round_interrupt_event ('i')。
    (此函数未更改)
    """
    
    llm_thread = None
    next_batch_data = {}
    current_batch = None
    first_image = None
    step_count = 0

    # [!!] 创建一个本地中断错误，以便我们可以优雅地捕获它
    class RoundInterruptedError(Exception):
        pass

    try:
        # --- 自动起飞 ---
        print("[VLA] 正在起飞...")
        # [!!] MODIFIED: Replaced f-string with .format() (in commented line)
        # drone.talk("收到指令: {}。准备起飞。".format(instruction))
        drone.take_off()
        
        # [!!] MODIFIED: 使睡眠可中断
        start_sleep = time.time()
        while time.time() - start_sleep < 5:
            if program_stop_event.is_set() or round_interrupt_event.is_set():
                raise RoundInterruptedError("在起飞等待时中断")
            time.sleep(0.1) # 等待稳定
            
        print("[VLA] 无人机已起飞。")
        
        # --- VLA 持续控制循环 ---
        # [!!] MODIFIED: Replaced f-string with .format()
        print("[VLA] 开始执行 VLA 指令: {}".format(instruction))

        # 1. [!!] 获取第一帧图像 (用于 VLA)
        first_image_rgb = drone.get_frame()
        # [!!] MODIFIED: 添加了中断检查
        while first_image_rgb is None and not (program_stop_event.is_set() or round_interrupt_event.is_set()):
                print("[VLA] 警告: 无法获取第一帧图像，正在重试...")
                # [!!] MODIFIED: 使用事件的 wait 方法
                if program_stop_event.wait(0.5): break # program_stop_event 被设置
                if round_interrupt_event.wait(0.01): break # round_interrupt_event 被设置
                first_image_rgb = drone.get_frame()
        
        if program_stop_event.is_set() or round_interrupt_event.is_set():
                raise RoundInterruptedError("在获取第一帧图像时程序被终止")

        first_image = Image.fromarray(first_image_rgb)
        
        # 2. [!!] *第一次* LLM 推理 (阻塞)
        print("[VLA] 正在执行*第一次* LLM 推理 (阻塞)...")
        llm_inference_wrapper(drone, client, instruction, args, first_image, next_batch_data)
        
        if program_stop_event.is_set() or round_interrupt_event.is_set():
                raise RoundInterruptedError("在第一次推理后中断")
                
        current_batch = next_batch_data.get('batch')
        if current_batch is None:
            # [!!] MODIFIED: Replaced f-string with .format()
            print("[VLA] 第一次 LLM 推理失败: {}".format(next_batch_data.get('error')))
            raise Exception("Initial LLM inference failed")
        
        # [!!] MODIFIED: 主循环添加了中断检查
        while not (program_stop_event.is_set() or round_interrupt_event.is_set()):
            if current_batch is None or len(current_batch) == 0:
                print("[VLA] 未收到有效动作，任务终止。")
                break

            # 检查当前批次是否是小增量，如果 max(|dx|, |dy|, |dz|, |dyaw|) < 1.0 cm/deg，则自动停止
            if _is_action_batch_small(current_batch, threshold=1.0):
                print("[VLA] **动作增量过小，判定任务完成，自动终止。**")
                drone.talk("动作增量过小，VLA任务自动完成")
                break 
            # --- ---

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
                
                # [!!] --- 唯一的代码修复在此 --- [!!]
                # 移除了不支持的 'stop_event' 参数
                drone.move_by_delta_pose_sequence(
                    main_steps, 
                    speed=args.speed
                )
                # [!!] --- 修复结束 --- [!!]
            
            if program_stop_event.is_set() or round_interrupt_event.is_set():
                raise RoundInterruptedError("在主要步骤执行期间中断")

            # 5. [!!] 等待上一个 LLM 线程 (如果存在)
            if llm_thread is not None:
                print("[VLA] 等待上一个 LLM 线程完成...")
                llm_thread.join() # 这是一个阻塞操作
                
                if program_stop_event.is_set() or round_interrupt_event.is_set():
                    raise RoundInterruptedError("在等待 LLM 线程时中断")
                
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

            if program_stop_event.is_set() or round_interrupt_event.is_set():
                raise RoundInterruptedError("在缓冲区步骤执行期间中断")
            
            step_count += 1
            if args.max_steps is not None and step_count >= args.max_steps:
                # [!!] MODIFIED: Replaced f-string with .format()
                print("[VLA] 已达到 {} 步最大限制，任务终止。".format(args.max_steps))
                drone.talk("达到最大步数，任务停止")
                break
            
    except (KeyboardInterrupt, RoundInterruptedError, InterruptedError) as e:
        print("\n[VLA] 检测到中断。停止当前VLA任务... ({})".format(type(e).__name__))
        if not (program_stop_event.is_set() or round_interrupt_event.is_set()):
             drone.talk("任务已中断")
        # [!!] 立即发送停止命令
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
            # [!!] 检查是 1. 程序退出 (Ctrl+C / 'q') 还是 2. 轮次中断 ('i')
            # 这两种情况都应该触发降落
            if program_stop_event.is_set(): 
               print("[VLA] 收到程序退出信号，强制降落...")
               drone.land()
            elif round_interrupt_event.is_set():
               print("[VLA] 收到轮次中断信号 ('i')，正在降落...")
               drone.land()
            # 否则，检查是否是正常结束且高度 > 10
            elif drone.get_current_pose()['z'] > 10: 
               print("[VLA] 任务正常完成，高度 > 10，正在降落。")
               drone.land()
            else:
               print("[VLA] 任务正常完成，高度 < 10，无需降落。")
            
        except Exception as e:
            # [!!] MODIFIED: Replaced f-string with .format()
            print("[VLA] 降落时检查高度失败: {}，尝试强制降落...".format(e))
            try:
                drone.land()
            except:
                pass
                
        print("[VLA] VLA 线程已完成。")


# [!!] 新增：用于获取指令的辅助函数
def get_instruction_from_user():
    """
    根据 ENABLE_SPEECH 标志获取用户指令。
    此函数会阻塞，但可以被 Ctrl+C (SIGINT) 中断。
    (此函数未更改)
    """
    instruction_text = ""
    if ENABLE_SPEECH:
        print("\n语音输入已启用。请说话 (或按 Ctrl+C 退出)...")
        inst = speech.record_and_get_text()
        inst = instruction.get_inst(inst)
        instruction_text = inst
        # [!!] MODIFIED: Replaced f-string with .format()
        print("识别到的指令: {}".format(instruction_text))
    else:
        print("\n语音输入已禁用。")
        instruction_text = input("请输入VLA指令 (输入 'exit' 或空指令以取消): ")
    
    return instruction_text

# [!!] MODIFIED: 'on_round_interrupt' 已被移除 (不再是回调)

def on_program_exit(sig, frame):
    """Ctrl+C 按下时的回调"""
    if not program_stop_event.is_set():
        print("\n[Main] 检测到 Ctrl+C！将退出所有进程...")
        program_stop_event.set()
        round_interrupt_event.set() # 同时也中断当前轮


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 VLA 模型控制 Tello/M4D 无人机")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=['gr00t', 'openvla'], 
        required=True, 
        help="要使用的VLA模型 ('gr00t' 或 'openvla')"
    )
    # ... (其余参数保持不变) ...
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
        help="Muat dan jalankan model GR00T secara lokal alih-alih terhubung ke server (hanya berlaku jika --model=gr00t)"
    )
    
    args = parser.parse_args()
    
    drone = None
    client = None
    vla_thread = None # [!!] 新增: 用于VLA逻辑的线程
    
    # [!!] 新增: 保存旧的终端设置
    old_settings = termios.tcgetattr(sys.stdin)

    # [!!] MODIFIED: 注册 Ctrl+C (SIGINT) 处理器
    signal.signal(signal.SIGINT, on_program_exit)

    try:
        # 1. [!!] 启动热键监听器 (已移除)

        # 2. [!!] 初始化无人机 (只执行一次)
        print("正在初始化 无人机...")
        drone = DjiM4DDrone()

        # 3. [!!] Inisialisasi klien (只执行一次)
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

        # 4. [!!] 开始主事件循环 (多轮次)
        
        # [!!] MODIFIED: 设置终端为 cbreak 模式 (立即读取单个字符)
        tty.setcbreak(sys.stdin.fileno())
        print("\n--- [主事件循环已启动] ---")
        print("按 'n' 开始新任务 (New Task)")
        print("按 'i' 中断当前任务 (Interrupt)")
        print("按 'q' "
" 退出程序 (Quit)")
        print("按 'l' 强制降落 (Land)")
        print("--------------------------")
        
        while not program_stop_event.is_set():
            # 检查 VLA 线程是否已结束
            if vla_thread and not vla_thread.is_alive():
                vla_thread.join()
                vla_thread = None
                print("\n[Main] VLA 任务线程已结束。")
                print("按 'n' 开始新任务, 'i' 中断, 'q' 退出, 'l' 降落")

            # 读取一个字符 (非阻塞)
            if program_stop_event.wait(0.1): # 0.1秒的超时
                break # 程序被要求停止
            
            # [!!] MODIFIED: 切换到阻塞读取, 这是 cbreak 模式的预期行为
            # 只有当有按键时, 循环才会继续
            char = sys.stdin.read(1)
            
            if not char or program_stop_event.is_set():
                continue

            # (n) 开始新任务
            if char == 'n':
                if vla_thread and vla_thread.is_alive():
                    print("\n[Main] 错误: 任务已在运行。请先按 'i' 中断。")
                else:
                    # [!!] 关键: 运行 input() 前恢复终端
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    
                    instruction_text = ""
                    try:
                        instruction_text = get_instruction_from_user()
                    except (KeyboardInterrupt, EOFError):
                        program_stop_event.set()
                    
                    # [!!] 关键: 恢复 cbreak 模式
                    tty.setcbreak(sys.stdin.fileno())

                    if instruction_text and instruction_text.lower() != 'exit':
                        print("[Main] 收到新指令，启动VLA线程...")
                        round_interrupt_event.clear()
                        vla_thread = threading.Thread(
                            target=main_vla_logic,
                            args=(drone, client, instruction_text, args,
                                  program_stop_event, round_interrupt_event),
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
                else:
                    print("\n[Main] 没有正在运行的任务可以中断。")

            # (l) 强制降落
            elif char == 'l':
                print("\n[Main] 检测到 'l'！强制降落...")
                drone.land()

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
        print("\n[Main] 正在关闭所有线程和连接...")
        program_stop_event.set() # 确保所有循环都停止
        
        # [!!] 关键: 恢复终端设置，否则您的终端会"坏掉"
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
           
        print("[Main] 程序已退出。")
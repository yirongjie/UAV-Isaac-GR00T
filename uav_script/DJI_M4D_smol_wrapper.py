import base64
import inspect
import json
import math
import os
import platform
import subprocess
import threading
import time
import re  
from functools import wraps

import cv2
import dotenv
import numpy as np
from openai import OpenAI

# --- M4D 特有的库 ---
import socket
import struct
import dji_kmz_mission_generator
# --- 结束 ---
from lvface_inferencer import LVFaceONNXInferencer

dotenv.load_dotenv()
try:
    from policy_client import Gr00tClient, OpenVLAClient
except ImportError:
    print("[Drone Wrapper] 警告: 未找到 'policy_client'。VLA 功能将不可用。")
    OpenVLAClient = None
    Gr00tClient = None
from PIL import Image

API_VL_MODEL = "qwen/qwen3-vl-30b-a3b-instruct"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ["OPENROUTER_API_KEY"]

class LocalSpeaker:
    """
    (LocalSpeaker 类保持不变)
    """
    def __init__(self):
        self.platform = platform.system()
        self.speaker_instance = None
        self.linux_speech_cmd = None
        
        if self.platform == "Windows":
            try:
                import win32com.client
                self.speaker_instance = win32com.client.Dispatch("SAPI.SpVoice")
                print("[Speaker] Initialized SAPI for speech.")
            except ImportError:
                print("[Speaker] Warning: win32com.client not found. Falling back to PowerShell for speech.")
                self.speaker_instance = "powershell"
        elif self.platform == "Darwin":
            print("[Speaker] Initialized 'say' command for speech on macOS.")
            pass
        elif self.platform == "Linux":
            if subprocess.call(["which", "spd-say"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                self.linux_speech_cmd = "spd-say"
                print("[Speaker] Initialized 'spd-say' command for speech on Linux.")
            elif subprocess.call(["which", "espeak"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                self.linux_speech_cmd = "espeak"
                print("[Speaker] Initialized 'espeak' command for speech on Linux.")
            else:
                print("[Speaker] Warning: 'spd-say' or 'espeak' not found on Linux. Speech will be disabled.")
        else:
            print(f"[Speaker] Warning: Unsupported platform '{self.platform}'. Speech will be disabled.")

    def speak(self, text: str):
        try:
            if self.platform == "Windows":
                if self.speaker_instance == "powershell":
                    command = f'powershell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\');"'
                    subprocess.run(command, shell=True, check=True)
                elif self.speaker_instance:
                    self.speaker_instance.Speak(text)
            elif self.platform == "Darwin":
                subprocess.run(["say", text], check=True)
            elif self.platform == "Linux" and self.linux_speech_cmd:
                if self.linux_speech_cmd == "spd-say":
                    subprocess.run(["spd-say", "-w", text], check=True)
                elif self.linux_speech_cmd == "espeak":
                    subprocess.run(["espeak", text], check=True)
            else:
                print(f"[Speech Disabled on {self.platform}] {text}")
        except subprocess.CalledProcessError as e:
            print(f"[Speaker] Error executing speech command: {e}")
        except Exception as e:
            print(f"[Speaker] An unexpected error occurred during speech: {e}")

def _cv2_to_base64(image, format=".png"):
    # (此函数保持不变)
    success, buffer = cv2.imencode(format, image)
    if not success:
        raise ValueError("图像编码失败")
    img_bytes = buffer.tobytes()
    return base64.b64encode(img_bytes).decode("utf-8")

# --- M4D 特有的辅助函数 ---
def recv_all(sock, count):
    """
    (此函数保持不变)
    """
    buf = b''
    while len(buf) < count:
        chunk = sock.recv(count - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf
# --- 结束 ---


class DjiM4DDrone:
    def __init__(self):
        self.llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        self.host = "localhost"#"192.168.42.120"
        self.port = 8899
        
        self.socket = None
        self.command_lock = threading.Lock()
        
        with self.command_lock:
            self._connect() 

        self.speaker = LocalSpeaker()
        
        self.x = 0.0
        self.y = 0.0
        
        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt_m = None
        self.origin_yaw_deg = None 

        self.world_map = {}

        print(f"DjiM4DDrone wrapper initialized for persistent connection to {self.host}:{self.port}")

        # --- VLA 客户端初始化 (保持不变) ---
        self.vla_client = None
        self.vla_model_type = 'gr00t'
        self.vla_gr00t_horizon = 4
        self.vla_ip="localhost"
        
        if self.vla_model_type == 'gr00t':
            if Gr00tClient is None:
                print("[Drone __init__] 警告: 'policy_client' 未加载。无法初始化 GR00T。")
                return
            try:
                port = 5555
                print(f"[Drone __init__] 正在尝试连接 GR00T 客户端: {self.vla_ip}:{port}")
                self.vla_client = Gr00tClient(ip=self.vla_ip, port=port, horizon=self.vla_gr00t_horizon)
                print("[Drone __init__] GR00T 客户端连接成功。")
            except Exception as e:
                print(f"[Drone __init__] 警告: 连接 GR00T 客户端失败: {e}。VLA 功能将不可用。")
        
        elif self.vla_model_type == 'openvla':
            if OpenVLAClient is None:
                print("[Drone __init__] 警告: 'policy_client' 未加载。无法初始化 OpenVLA。")
                return
            try:
                port = 5007
                print(f"[Drone __init__] 正在尝试连接 OpenVLA 客户端: {self.vla_ip}:{port}")
                self.vla_client = OpenVLAClient(ip=self.vla_ip, port=port)
                print("[Drone __init__] OpenVLA 客户端连接成功。")
            except Exception as e:
                print(f"[Drone __init__] 警告: 连接 OpenVLA 客户端失败: {e}。VLA 功能将不可用。")
        
        elif self.vla_model_type is not None:
            print(f"[Drone __init__] 警告: 未知的 VLA 模型类型 '{self.vla_model_type}'。VLA 功能将不可用。")
        
        else:
            print("[Drone __init__] 未指定 VLA 模型，VLA 功能将不可用。")
        # --- 结束 ---

    
    # 手动工具生成器
    # ----------------------------------------------------
    def _parse_docstring(self, docstring: str) -> dict:
        """
        从Numpy风格的文档字符串中解析描述、参数和返回类型。
        """
        if not docstring:
            return {"description": "", "inputs": {}, "output_type": "None"}

        # 1. 解析描述
        desc_match = re.search(r'^(.*?)(?=\n\s*(Args|Returns|Raises|Examples):|\Z)', docstring, re.DOTALL | re.IGNORECASE)
        description = desc_match.group(1).strip().replace('\n', ' ') if desc_match else ""

        # 2. 解析参数 (Args)
        inputs = {}
        args_match = re.search(r'\n\s*Args:\s*\n(.*?)(?=\n\s*(Returns|Raises|Examples):|\Z)', docstring, re.DOTALL | re.IGNORECASE)
        if args_match:
            args_block = args_match.group(1)
            # 匹配 'param_name (param_type): description'
            arg_pattern = re.compile(r'^\s*([a-zA-Z0-9_]+)\s*\((.*?)\):\s*(.*?)(?=\n\s*[a-zA-Z0-9_]+\s*\(|\Z)', re.DOTALL | re.MULTILINE)
            for match in arg_pattern.finditer(args_block):
                name, type_str, desc = match.groups()
                inputs[name.strip()] = {
                    "type": type_str.strip(),
                    "description": desc.strip().replace('\n', ' ')
                }

        # 3. 解析返回类型 (Returns)
        output_type = "None"
        returns_match = re.search(r'\n\s*Returns:\s*\n\s*(.*?):', docstring, re.DOTALL | re.IGNORECASE)
        if returns_match:
            output_type = returns_match.group(1).strip()

        return {"description": description, "inputs": inputs, "output_type": output_type}

    def get_tools(self) -> list:
        """
        手动生成工具列表，用于替换 smolagents.tool 装饰器。
        """
        # 这是基于你文件中注释掉的 @expose_methods_as_tools 'include' 列表
        tool_names = [
            "move_forward",
            "move_backward",
            "take_off",
            "move_up",
            "move_down",
            "move_left",
            "move_right",
            "turn_clockwise",
            "turn_counter_clockwise",
            "land",
            "get_height",
            "get_current_pose",
            "watch",
            "objects_vlm",
            "scan",
            "move_to_object",
            "take_picture",
            "talk",
        ]
        
        tools_list = []
        for name in tool_names:
            if not hasattr(self, name):
                print(f"[get_tools] 警告: 未找到名为 '{name}' 的方法。")
                continue
                
            method = getattr(self, name)
            docstring = inspect.getdoc(method)
            parsed_doc = self._parse_docstring(docstring)
            
            # 构建 smolagents 期望的格式 (也是 YAML 模板期望的格式)
            tool_def = {
                "name": name,
                "description": parsed_doc["description"],
                "inputs": parsed_doc["inputs"],
                "output_type": parsed_doc["output_type"],
                "callable": method # [!!] 保留对实际函数的引用
            }
            tools_list.append(tool_def)
            
        return tools_list

    def get_tool_by_name(self, name: str) -> callable:
        """
        根据名称获取可调用的工具方法。
        (SimpleToolCallingAgent 将使用此方法)
        """
        if not hasattr(self, name):
            raise ValueError(f"未找到名为 '{name}' 的工具。")
        
        method = getattr(self, name)
        if not callable(method) or name.startswith("_"):
             raise ValueError(f"'{name}' 不是一个有效的、可调用的工具。")
             
        return method
    # ----------------------------------------------------

    def _connect(self) -> bool:
        """
        (内部) 建立或重建到 C++ PSDKServer 的持久连接。
        此函数必须在持有 self.command_lock 时被调用。
        """
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass 
            self.socket = None
        
        try:
            print(f"[_connect] 正在连接到 {self.host}:{self.port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0) 
            self.socket.connect((self.host, self.port))
            print("[_connect] 连接已建立。")
            return True
        except (socket.error, socket.timeout) as e:
            print(f"[_connect] 连接失败: {e}")
            self.socket = None
            return False

    def _send_command(self, command: str) -> dict:
        """
        在持久套接字上发送单个命令，并获取响应。
        如果连接丢失，会自动尝试重新连接一次。
        此函数是线程安全的 (使用 self.command_lock)。
        """
        cmd_parts = command.strip().split()
        cmd_name = cmd_parts[0]
        cmd_args = " ".join(cmd_parts[1:])

        # print(f"[_send_command] Sending: '{cmd_name}' with args: '{cmd_args}'")
        
        with self.command_lock:
            LONG_RUNNING_CMDS = ["fc_pos", "fc_pos_gnd", "fc_seq", "fc_pos_gps", "fc_pos_wp"]

            is_long_task = False
            for cmd in LONG_RUNNING_CMDS:
                if cmd_name == cmd: # (确保 cmd_name 在此作用域内可见)
                    is_long_task = True
                    break
            try:
                if self.socket is None:
                    print(f"[_send_command] Socket is None for '{cmd_name}', 尝试重连...")
                    if not self._connect():
                        return {"status": "error", "message": "Reconnect failed"}
                
                self.socket.settimeout(10.0) 
                self.socket.sendall(command.encode('utf-8'))

                resp_data = recv_all(self.socket, 4)
                if not resp_data:
                    print(f"[_send_command] 错误: 服务器断开连接 (未收到响应) @ cmd: {cmd_name}")
                    self.socket.close()
                    self.socket = None
                    return {"status": "error", "message": "Connection closed prematurely"}

                response_code = struct.unpack('!I', resp_data)[0]

                if cmd_name == "tp":
                    img_size = response_code
                    if img_size == 0:
                        print(f"[_send_command] 错误: 服务器报告拍照失败 (收到 0 字节)。")
                        return {"status": "error", "message": "Take photo failed on server"}
                    self.socket.settimeout(10.0)
                    img_data = recv_all(self.socket, img_size)
                    if not img_data or len(img_data) != img_size:
                        print(f"[_send_command] 错误: 接收图像数据不完整。")
                        return {"status": "error", "message": "Incomplete image data"}
                    return {"status": "ok", "type": "image", "data": img_data}

                elif cmd_name == "get_gps" or cmd_name == "get_pose":
                    json_size = response_code
                    if json_size == 0:
                        print(f"[_send_command] 错误: 服务器报告 '{cmd_name}' 失败。")
                        return {"status": "error", "message": f"{cmd_name} failed on server"}
                    self.socket.settimeout(5.0)
                    json_data_raw = recv_all(self.socket, json_size)
                    if not json_data_raw:
                        print(f"[_send_command] 错误: 接收 JSON 数据不完整。")
                        return {"status": "error", "message": "Incomplete JSON data"}
                    try:
                        status_data = json.loads(json_data_raw.decode('utf-8'))
                        return {"status": "ok", "type": "json", "data": status_data}
                    except Exception as e:
                        print(f"[_send_command] 错误: 无法解析 JSON 响应: {e}")
                        return {"status": "error", "message": f"JSON parse error: {e}"}

                else:
                    if not is_long_task:
                        # --- 1. 标准短时任务 (fc_takeoff, fc_land, fc_vel) ---
                        if response_code == 0:
                            print(f"[_send_command] 错误: 服务器报告命令 '{cmd_name}' 失败。")
                            return {"status": "error", "message": f"Server reported command failed"}
                        else:
                            print(f"[_send_command] 命令 '{cmd_name}' 成功。")
                            return {"status": "ok", "type": "simple"}

                    else:
                        # --- 2. 新的长时任务 (fc_pos, fc_pos_wp, etc.) ---
                        if response_code == 1: # 1 = Accepted
                            print(f"[_send_command] 长时任务 '{cmd_name}' 已接受, 正在等待最终完成信号...")
                            self.socket.settimeout(300.0) 

                            final_resp_data = recv_all(self.socket, 4)
                            if not final_resp_data:
                                print(f"[_send_command] 错误: 等待 '{cmd_name}' 完成信号时服务器断开。")
                                self.socket.close()
                                self.socket = None
                                return {"status": "error", "message": "Connection closed while waiting for completion"}

                            final_response_code = struct.unpack('!I', final_resp_data)[0]

                            if final_response_code == 2: # 2 = Completed
                                print(f"[_send_command] 长时任务 '{cmd_name}' 已成功完成。")
                                return {"status": "ok", "type": "simple"}
                            else: # 0 = Failed
                                print(f"[_send_command] 错误: 长时任务 '{cmd_name}' 在执行期间失败 (code {final_response_code})。")
                                return {"status": "error", "message": f"Task failed during execution (code {final_response_code})"}

                        else: # 0 = Failed (任务被立即拒绝)
                            print(f"[_send_command] 错误: 长时任务 '{cmd_name}' 被服务器立即拒绝 (code {response_code})。")
                            return {"status": "error", "message": f"Task was rejected by server (code {response_code})"}

            except (socket.timeout, socket.error, BrokenPipeError, ConnectionResetError) as e:
                print(f"[_send_command] Socket 错误 @ cmd: {cmd_name}: {e}. 套接字将关闭。")
                if self.socket:
                    self.socket.close()
                self.socket = None
                return {"status": "error", "message": f"Socket error: {e}"}
            except Exception as e:
                print(f"[_send_command] 发生错误 @ cmd: {cmd_name}: {e}")
                return {"status": "error", "message": f"Unknown error: {e}"}
            

    # -----#

    def print_seq(self, sequence: str) -> None:
        """
        A simple tool to print a sequence of actions or observations.
        Args:
            sequence (str): The sequence to print.
        """
        print(sequence)

    def get_frame(self, sharpen: bool = True) -> np.ndarray:
        """Get the current frame from the drone. (For Agent/Live Feed)
        此函数返回一个 1/2 尺寸的 RGB 图像 (480x360)。

        Args:
            sharpen (bool, optional): Whether to apply sharpening and exposure adjustment. Defaults to True.

        Returns:
            np.ndarray: The processed frame as a NumPy array (RGB format, 480x360).
        """
        resp = self._send_command("tp 480") 
        if resp.get("status") != "ok" or resp.get("type") != "image":
            print("[get_frame] 错误: 未能从 M4D 获取图像。")
            return None
        img_data = resp["data"]
        try:
            np_arr = np.frombuffer(img_data, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                print("[get_frame] 错误: cv2.imdecode 失败。")
                return None
        except Exception as e:
            print(f"[get_frame] 图像解码时出错: {e}")
            return None
        frame_bgr_resized = cv2.resize(frame_bgr, (480, 360))
        frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)
        cv2.imwrite("current_frame.png", frame_bgr) 
        return frame_rgb

    def get_frame_vlm(self, sharpen: bool = True) -> np.ndarray:
        """Get the current frame from the drone. (For VLM models)
        此函数返回一个 *完整尺寸* 的 BGR 图像 (960x720)

        Args:
            sharpen (bool, optional): Whether to apply sharpening and exposure adjustment. Defaults to True.

        Returns:
            np.ndarray: The processed frame as a NumPy array (BGR format, 960x720).
        """
        resp = self._send_command("tp 720")
        if resp.get("status") != "ok" or resp.get("type") != "image":
            print("[get_frame_vlm] 错误: 未能从 M4D 获取图像。")
            return None
        img_data = resp["data"]
        try:
            np_arr = np.frombuffer(img_data, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                print("[get_frame_vlm] 错误: cv2.imdecode 失败。")
                return None
        except Exception as e:
            print(f"[get_frame_vlm] 图像解码时出错: {e}")
            return None
        cv2.imwrite("current_frame_vlm.png", frame_bgr) 
        return frame_bgr

    def _get_realtime_pose(self) -> dict:
        """
        (内部函数) 调用 C++ 'get_pose' 指令，获取并返回无人机的实时 3D 姿态。
        """
        resp = self._send_command("get_pose")
        if resp.get("status") == "ok" and resp.get("type") == "json":
            data = resp["data"]
            return {
                "lat": data.get("lat", 0.0),
                "lon": data.get("lon", 0.0),
                "alt_m": data.get("alt_m", 0.0),    
                "yaw_deg": data.get("yaw_deg", 0.0) 
            }
        print("警告: _get_realtime_pose() 失败，返回默认值 (0,0,0,0)")
        return {"lat": 0.0, "lon": 0.0, "alt_m": 0.0, "yaw_deg": 0.0}

    def _get_current_relative_yaw_ccw_deg(self) -> float:
        """
        (内部函数) 获取当前相对于起飞点的 *相对* 偏航角 (Tello 坐标系, CCW+)。
        """
        pose = self._get_realtime_pose()
        current_abs_yaw_cw_deg = pose.get("yaw_deg", 0.0)
        if self.origin_yaw_deg is None:
            print("警告: _get_current_relative_yaw_ccw_deg: 'origin_yaw_deg' 未设置。")
            origin_yaw_cw_deg = 0.0
        else:
            origin_yaw_cw_deg = self.origin_yaw_deg
        relative_yaw_cw_deg = (current_abs_yaw_cw_deg - origin_yaw_cw_deg) % 360
        relative_yaw_ccw_deg = (360 - relative_yaw_cw_deg) % 360 
        return relative_yaw_ccw_deg
    
    # def recalibrate_local_state(self):
    #     """
    #     重置本地航位推算状态 (x, y) 并重新设置原点 (GPS/Yaw)，
    #     假设无人机已在空中悬停。此过程不发送任何起飞/降落指令。
    #     """
    #     # [修改] 1. 调用新的 C++ 命令，执行 Release/Obtain/Hover 周期以强制解锁
    #     print("[校准] 正在执行 C++ 端 Release/Obtain 周期以强制解锁飞控...")
    #     resp = self._send_command("fc_regain_ctrl") 
        
    #     if resp.get("status") != "ok":
    #          # 如果解锁命令失败，后续 VLA 任务很可能会失败。
    #          print("[校准] 严重错误: C++ 端 'fc_regain_ctrl' 命令失败。请手动摇杆解锁。")
    #     else:
    #          print("[校准] C++ 端控制权重新获取成功。")

    #     # 2. 重置本地航位推算坐标
    #     self.x = 0.0
    #     self.y = 0.0
        
    #     # 3. 重新获取并设置原点姿态 (关键步骤)
    #     time.sleep(1) # 等待 PSDK 状态稳定
    #     pose = self._get_realtime_pose()
    #     print("[校准] 状态校准成功。")
    #     if pose.get("lat") != 0.0 or pose.get("lon") != 0.0:
    #         self.origin_lat = pose["lat"]
    #         self.origin_lon = pose["lon"]
    #         self.origin_alt_m = pose["alt_m"]
    #         self.origin_yaw_deg = pose["yaw_deg"] 
    #         print(f"[校准] 新原点已设置: Lat={self.origin_lat}, Lon={self.origin_lon}, Alt={self.origin_alt_m}m, Yaw={self.origin_yaw_deg}deg (CW North)")
    #     else:
    #         print("[校准] 警告: 状态校准后未能获取原点 GPS/Yaw。")
    #         self.origin_yaw_deg = 0.0

    def move_forward(self, distance: int) -> None:
        """
        Move the drone forward by a specified distance in centimeters.

        Args:
            distance (int): The distance to move forward in centimeters.必须是正数
        """
        dist_m = distance / 100.0
        yaw_rad_ccw = math.radians(self._get_current_relative_yaw_ccw_deg())
        self.x += distance * math.cos(yaw_rad_ccw)
        self.y += distance * math.sin(yaw_rad_ccw)
        self._send_command(f"fc_pos {dist_m} 0 0 0")

    def move_backward(self, distance: int) -> None:
        """
        Move the drone backward by a specified distance in centimeters.

        Args:
            distance (int): The distance to move backward in centimeters.必须是正数
        """
        dist_m = distance / 100.0
        yaw_rad_ccw = math.radians(self._get_current_relative_yaw_ccw_deg())
        self.x -= distance * math.cos(yaw_rad_ccw)
        self.y -= distance * math.sin(yaw_rad_ccw)
        self._send_command(f"fc_pos -{dist_m} 0 0 0")

    def land(self) -> None:
        """
        Land the drone.
        """
        self._send_command("fc_land")
        self.x = 0.0
        self.y = 0.0
        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt_m = None
        self.origin_yaw_deg = None

    def take_off(self) -> None:
        """
        Take off the drone.
        起飞后无人机会距离地面50cm.
        """
        resp = self._send_command("fc_takeoff")
        if resp.get("status") == "ok":
            self.x = 0.0
            self.y = 0.0
            time.sleep(5) 
            pose = self._get_realtime_pose()
            print("起飞成功。")
            if pose.get("lat") != 0.0 or pose.get("lon") != 0.0:
                self.origin_lat = pose["lat"]
                self.origin_lon = pose["lon"]
                self.origin_alt_m = pose["alt_m"]
                self.origin_yaw_deg = pose["yaw_deg"] 
                print(f"原点已设置: Lat={self.origin_lat}, Lon={self.origin_lon}, Alt={self.origin_alt_m}m, Yaw={self.origin_yaw_deg}deg (CW North)")
            else:
                print("警告: 起飞后未能获取原点 GPS/Yaw。")
                self.origin_yaw_deg = 0.0 
        else:
            print("起飞失败。")

    def move_up(self, distance: int) -> None:
        """
        This is a tool that moves the drone up by a specified distance in centimeters.

        Args:
            distance (int): The distance to move up in centimeters.最少是20cm
        """
        dist_m = distance / 100.0
        self._send_command(f"fc_pos 0 0 {dist_m} 0")

    def move_down(self, distance: int) -> None:
        """
        Move the drone down by a specified distance in centimeters.

        Args:
            distance (int): The distance to move down in centimeters.最少是20cm
        """
        dist_m = distance / 100.0
        self._send_command(f"fc_pos 0 0 -{dist_m} 0")

    def get_height(self) -> int:
        """
        获取当前无人机的高度（单位：厘米）。

        Returns:
            int: 当前高度（厘米）
        """
        pose = self._get_realtime_pose()
        alt_m = pose.get("alt_m", 0.0)
        alt_cm = int(alt_m * 100.0)
        print(f"当前高度: {alt_cm} cm")
        return alt_cm

    def get_current_pose(self) -> dict:
        """
        获取无人机当前在世界坐标系中的位姿（位置和姿态）。

        该函数返回一个相对于无人机启动点的坐标。坐标系是一个标准的右手笛卡尔坐标系：
        - 原点 (0,0,0): 无人机启动时的位置。
        - +X轴: 无人机启动时正对的方向 (前进方向)。
        - +Y轴: 无人机启动时左侧的方向。
        - +Z轴: 垂直向上的方向。
        - 偏航角 (Yaw): 从+X轴开始的逆时针旋转角度。0度表示无人机朝向世界坐标系的+X轴。

        Returns:
            dict: 一个包含位姿信息的字典。
                  {'x': float, 'y': float, 'z': float, 'yaw': float}
                  - 'x', 'y', 'z': 在世界坐标系中的位置，单位为厘米（cm）。
                  - 'yaw': 在世界坐标系中的偏航角，单位为度 (0-360)，逆时针为正。
        """
        pose = self._get_realtime_pose()
        
        current_alt_cm = pose["alt_m"] * 100.0
        relative_yaw_ccw_deg = self._get_current_relative_yaw_ccw_deg()
        
        pose_tello_coords = {
            "x": round(self.x, 2), # 累积的 X
            "y": round(self.y, 2), # 累积的 Y
            "z": round(current_alt_cm, 2), # 实时的 Z
            "yaw": round(relative_yaw_ccw_deg, 2), # 实时的相对 Yaw
        }
        print(f"当前位姿 (X/Y累积, Z实时, Yaw相对): {pose_tello_coords}")
        return pose_tello_coords


    def move_left(self, distance: int) -> None:
        """
        Move the drone left by a specified distance in centimeters.

        Args:
            distance (int): The distance to move left in centimeters.
        """
        dist_m = distance / 100.0
        yaw_rad_ccw = math.radians(self._get_current_relative_yaw_ccw_deg())
        self.x -= distance * math.sin(yaw_rad_ccw)
        self.y += distance * math.cos(yaw_rad_ccw)
        self._send_command(f"fc_pos 0 -{dist_m} 0 0")

    def move_right(self, distance: int) -> None:
        """
        Move the drone right by a specified distance in centimeters.

        Args:
            distance (int): The distance to move right in centimeters.
        """
        dist_m = distance / 100.0
        yaw_rad_ccw = math.radians(self._get_current_relative_yaw_ccw_deg())
        self.x += distance * math.sin(yaw_rad_ccw)
        self.y -= distance * math.cos(yaw_rad_ccw)
        self._send_command(f"fc_pos 0 {dist_m} 0 0")

    def turn_clockwise(self, degrees: int) -> None:
        """
        Turn the drone clockwise by a specified number of degrees.
        (旋转是相对的)
        Args:
            degrees (int): The number of degrees to turn clockwise.
        """
        self._send_command(f"fc_pos 0 0 0 {degrees}")

    def turn_counter_clockwise(self, degrees: int) -> None:
        """
        Turn the drone counter-clockwise by a specified number of degrees.
        (旋转是相对的)
        Args:
            degrees (int): The number of degrees to turn counter-clockwise.
        """
        self._send_command(f"fc_pos 0 0 0 -{degrees}")

    def talk(self, text: str) -> None:
        """
        使用本地语音引擎将输入的文字朗读出来。每开始或完成一个阶段的无人机任务，必须调用此函数让无人机说话，不能省略。

        Args:
            text (str): 需要朗读的中文或英文文本。
        """
        self.speaker.speak(text)

    def take_picture(self, desc: str) -> None:
        """
        拍照并保存当前帧到 pictures 目录下，文件名为 时间戳_自定义文字.png。自定义文字必须是英文
        注意, 拍照前一定要确认想要拍摄的内容在画面中。
        **注意**，拍照时候应该在距离拍照目标**2.5米**处.调用take_picture一定要谨慎，除非明确要拍照，否则不要调用此函数

        Args:
            desc (str): 图片自定义描述，会拼接到文件名中。例如 "table"，则文件名为 20250929_153012_table.png。
        """
        import datetime
        import os
        img_bgr = self.get_frame_vlm()
        if img_bgr is None:
            print("拍照失败，未能获取图像。")
            return
        pic_dir = "pictures"
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_desc = desc.replace(" ", "_").replace(":", "-")
        file_path = os.path.join(pic_dir, f"{ts}_{safe_desc}.png")
        cv2.imwrite(file_path, img_bgr)
        print(f"图片已保存: {file_path}")

    def watch(self, prompt: str) -> str:
        """
        让大模型描述当前画面。
        Args:
            prompt (str): 你想让模型描述的问题。
        Returns:
            str: 大模型的回答。
            watch_result = watch("xxx"),
            if "ZZZ" in watch_result:  # 这种用法是绝对**不**允许的
        """
        bgr_image = self.get_frame_vlm()
        if bgr_image is None:
            return "错误：未能获取 VLM 图像。"
        base64_rgb_str = _cv2_to_base64(bgr_image, ".png") 
        response = self.llm_client.chat.completions.create(
            model=API_VL_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "你正在操控一个无人机。下面是无人机当前视角的彩色图片。请根据这张图的信息用**最简洁的不说废话的**话来回答我的问题。\n\n问题：" + prompt },
                        { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{base64_rgb_str}" } },
                    ],
                }
            ],
            temperature=0.01,
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content

    def objects_vlm(self, obj_name_list: list) -> list:
        """
        让大模型（视觉语言模型/VLM）检测当前无人机视角下的指定物体，并根据图像内容估算每个物体距离无人机的距离和角度。
        本方法适用于仅有RGB相机，距离估算完全依赖大模型的视觉推理能力。坐标系遵循标准相机坐标系：X轴向右，Y轴向下，Z轴向前。

        Args:
            obj_name_list (list): 需要检测和估算距离的物体名称列表。例如：["显示器", "蓝色球"]。

        Returns:
            list: 每个物体的检测结果，包含：
                - name: 物体名称 (str)
                - x: 物体中心点在摄像头坐标系下的x轴距离（单位：米，float）
                - y: 物体中心点在摄像头坐标系下的y轴距离（单位：米，float）
                - z: 物体中心点在摄像头坐标系下的z轴距离（单位：米，float，通常为深度）
        """
        bgr_image = self.get_frame_vlm()
        if bgr_image is None:
            print("objects_vlm 错误：未能获取 VLM 图像。")
            return []
        base64_rgb_str = _cv2_to_base64(bgr_image, ".png")
        obj_list_str = ", ".join(obj_name_list)
        prompt = f"""
        Detect all {obj_list_str} in the image.
        - If no objects are found, return an empty JSON array: []
        - If objects are found, output ONLY a valid JSON array in this format:
        [{{"bbox_3d": [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw], "label": "category"}}]
        
        Important instructions:
        1. Return ONLY the JSON array with no additional text.
        2. Use empty array [] when no objects are detected.
        3. For each object, provide 3D bounding box coordinates in meters relative to the camera.
        4. Never output explanations or other text.
        """
        response = self.llm_client.chat.completions.create(
            model=API_VL_MODEL,
            messages=[
                { "role": "user", "content": [ {"type": "text", "text": prompt}, { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{base64_rgb_str}" } }, ], }
            ],
            temperature=0.0,
        )
        raw_response = response.choices[0].message.content
        print(f"Raw VLM response: {raw_response}")
        try:
            try:
                result = json.loads(raw_response)
            except json.JSONDecodeError:
                start_index = raw_response.find("[")
                end_index = raw_response.rfind("]")
                if start_index != -1 and end_index != -1:
                    json_string = raw_response[start_index : end_index + 1]
                    result = json.loads(json_string)
                else:
                    result = []
            
            annotated_img = bgr_image.copy(); save_annotated = False
            if result and isinstance(result, list) and len(result) > 0:
                save_annotated = True; h, w = annotated_img.shape[:2]; f = 500.0; cx, cy = w / 2, h / 2
                category_colors = {}; colors = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 0),(255, 0, 255),(0, 255, 255),]
                for obj in result:
                    bbox = obj.get("bbox_3d", []); label = obj.get("label", "unknown")
                    if len(bbox) < 9: continue
                    x_c, y_c, z_c = bbox[0], bbox[1], bbox[2]; dx, dy, dz = bbox[3], bbox[4], bbox[5]; roll, pitch, yaw = bbox[6], bbox[7], bbox[8]
                    if label not in category_colors: category_colors[label] = colors[len(category_colors) % len(colors)]
                    color = category_colors[label]
                    corners_local = np.array([[dx/2,dy/2,dz/2],[dx/2,dy/2,-dz/2],[dx/2,-dy/2,dz/2],[dx/2,-dy/2,-dz/2],[-dx/2,dy/2,dz/2],[-dx/2,dy/2,-dz/2],[-dx/2,-dy/2,dz/2],[-dx/2,-dy/2,-dz/2],])
                    R_x = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]]); R_y = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]]); R_z = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]]); R = R_z @ R_y @ R_x
                    corners_3d = (R @ corners_local.T).T + np.array([x_c, y_c, z_c]); corners_2d = []
                    for corner in corners_3d:
                        if corner[2] <= 0: corners_2d.append(None); continue
                        u = int(f * corner[0] / corner[2] + cx); v = int(f * corner[1] / corner[2] + cy); corners_2d.append((u, v))
                    lines = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
                    for i, j in lines:
                        if corners_2d[i] and corners_2d[j]: cv2.line(annotated_img, corners_2d[i], corners_2d[j], color, 2)
                    valid_corners = [c for c in corners_2d if c is not None]
                    if valid_corners: centroid = np.mean(valid_corners, axis=0).astype(int); cv2.putText(annotated_img, label, (centroid[0], centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if save_annotated: cv2.imwrite("current_frame_xyz.png", annotated_img); print(f"Saved 3D bounding box visualization: current_frame_xyz.png")
            
            processed = []
            for obj in result:
                bbox = obj.get("bbox_3d", []); label = obj.get("label", "unknown")
                if len(bbox) >= 3:
                    processed.append({ "name": label, "x": round(float(bbox[0]), 3), "y": round(float(bbox[1]), 3), "z": round(float(bbox[2]), 3), })
            return processed
        except Exception as e:
            print(f"解析VLM响应时出错: {e}")
            return []

    # [!! FIXED !!]
    def scan(self, obj_name_list: list) -> list:
        """
        让无人机原地旋转一圈以扫描周围环境，检测指定物体。
        无人机从当前位置开始，逆时针旋转，每次旋转45°，调用 objects_vlm 检测目标。
        一旦检测到目标（objects_vlm 返回非空），立即返回检测结果。
        如果旋转一圈都未检测到目标，则返回空列表。
        函数返回的时候的视角就是无人机最终的视角，就是找到目标的视角。

        Args:
            obj_name_list (list): 需要检测的物体名称列表，例如 ["显示器", "蓝色球"]。

        Returns:
            list: objects_vlm 的检测结果（如未检测到则返回 []）。
        """
        # (M4D) 原地360度扫描
        for i in range(8):
            result = self.objects_vlm(obj_name_list)
            if result and len(result) > 0:
                print(f"[scan] 在第 {i+1} 次尝试 (旋转 {i*45} 度) 时找到目标。")
                return result
            print(f"[scan] 第 {i+1}/8 次尝试 (原地) 未找到目标，逆时针旋转 45 度...")
            self.turn_counter_clockwise(45)

        # (M4D) 向左 1.5m, 扫描360度
        print("[scan] 原地扫描失败。向左移动 150cm...")
        self.move_left(150) 
        for i in range(8):
            result = self.objects_vlm(obj_name_list)
            if result and len(result) > 0:
                print(f"[scan] 在左侧 {i+1} 次尝试时找到目标。")
                return result
            print(f"[scan] 第 {i+1}/8 次尝试 (左侧) 未找到目标，逆时针旋转 45 度...")
            self.turn_counter_clockwise(45)

        # (M4D) 向右 3m, 扫描360度
        print("[scan] 左侧扫描失败。向右移动 300cm (穿过原点)...")
        self.move_right(300) # (从左 150 -> 右 150)
        for i in range(8):
            result = self.objects_vlm(obj_name_list)
            if result and len(result) > 0:
                print(f"[scan] 在右侧 {i+1} 次尝试时找到目标。")
                return result
            print(f"[scan] 第 {i+1}/8 次尝试 (右侧) 未找到目标，逆时针旋转 45 度...")
            self.turn_counter_clockwise(45)

        # (M4D) 回到原点
        print("[scan] 右侧扫描失败。返回起始水平位置...")
        self.move_left(150)

        print("[scan] 所有扫描尝试均失败。")
        return []

    def move_to_object(
        self, object_name: str, target_distance_cm: int, target_height_cm: int = 0
    ) -> bool:
        """
        适合除了**货架**之外的场景
        扫描、对准并移动到指定物体前方的特定距离和高度。
        该函数会执行以下四步操作：
        1. 使用 scan 找到物体。
        2. 根据物体的x坐标，自动旋转无人机，将物体对准到视野中心。
        3. 根据物体的y坐标，自动升降无人机，将物体对准到视野中心（可指定目标高度）。
        4. 根据物体的z坐标，前进到目标距离。

        Args:
            object_name (str): 要寻找的物体名称。
            target_distance_cm (int): 最终希望与物体保持的距离（单位：厘米）。
            target_height_cm (int, optional): 希望与物体保持的垂直距离（单位：厘米），默认0。除非特殊需求，一般保持0即可。这个不是距离地面的高度，而是物体在y轴方向的偏移。

        Returns:
            bool: 操作是否成功完成。成功返回True, 如果物体未找到或移动失败，返回 False。
        """
        print(f"开始执行 move_to_object: 目标 '{object_name}', 距离 {target_distance_cm}cm")
        scan_result = self.scan([object_name])
        if not scan_result:
            print(f"未找到目标物体 '{object_name}'，无法执行 move_to_object。")
            return False
        obj_info = scan_result[0]
        x_m = obj_info.get("x", 0); y_m = obj_info.get("y", 0); z_m = obj_info.get("z", 0)
        if z_m <= 0:
            print(f"VLM 返回无效 Z 距离 ({z_m})，无法计算移动。")
            return False
        actions_taken = []
        horizontal_threshold_m = 0.05
        if abs(x_m) > horizontal_threshold_m:
            angle_rad = math.atan(x_m / z_m); angle_deg = int(math.degrees(angle_rad))
            if angle_deg > 0:
                print(f"目标在右侧，需向右转 {angle_deg} 度。")
                self.turn_clockwise(angle_deg); actions_taken.append(f"向右转 {angle_deg} 度")
            elif angle_deg < 0:
                print(f"目标在左侧，需向左转 {abs(angle_deg)} 度。")
                self.turn_counter_clockwise(abs(angle_deg)); actions_taken.append(f"向左转 {abs(angle_deg)} 度")
        vertical_threshold_m = 0.05
        vertical_error_m = y_m - (target_height_cm / 100.0); move_z_cm = int(vertical_error_m * 100)
        if abs(move_z_cm) > int(vertical_threshold_m * 100):
            if move_z_cm > 0:
                print(f"目标在下方，需下降 {move_z_cm} cm。"); self.move_down(move_z_cm); actions_taken.append(f"下降 {move_z_cm} cm")
            else:
                print(f"目标在上方，需上升 {abs(move_z_cm)} cm。"); self.move_up(abs(move_z_cm)); actions_taken.append(f"上升 {abs(move_z_cm)} cm")
        current_distance_cm = int(z_m * 100); move_distance_cm = current_distance_cm - target_distance_cm
        if move_distance_cm > 20: 
            print(f"正在前进 {move_distance_cm} cm。"); self.move_forward(min(move_distance_cm, 500)); actions_taken.append(f"前进 {move_distance_cm} cm")
        elif move_distance_cm < -20:
            print(f"正在后退 {abs(move_distance_cm)} cm。"); self.move_backward(min(abs(move_distance_cm), 500)); actions_taken.append(f"后退 {abs(move_distance_cm)} cm")
        if not actions_taken:
            print("物体已经在目标位置，无需移动。"); return False
        else:
            print("完成 move_to_object 操作，动作包括: " + ", ".join(actions_taken)); return True  

    def move_by_delta_pose_sequence(self, delta_pose_array, speed=40):
        """
        按照输入的N个delta点依次移动无人机。
        输入格式：array([[delta_x, delta_y, delta_z, delta_yaw], ...])，单位为厘米和度。
        
        坐标系: FRD (VLA 模型标准)
        delta_x: 前后（正为前，负为后）
        delta_y: 左右（正为右，负为左）
        delta_z: 上下（正为下，负为上)
        delta_yaw: 旋转（正为顺时针，负为逆时针）
        
        speed: 飞行速度（在此版本中未使用，因为 move_x 命令是基于距离的）。
        """
        if not isinstance(delta_pose_array, np.ndarray):
            delta_pose_array = np.array(delta_pose_array)
        assert delta_pose_array.shape[1] == 4, "输入必须为N x 4的数组"
        n_steps = len(delta_pose_array)
        if n_steps == 0:
            return
        cmd_parts = [f"fc_seq {float(speed)} {n_steps}"]
        print("=="*10)
        for i, (dx_cm, dy_cm_vla, dz_cm_vla, dyaw_deg) in enumerate(delta_pose_array):
            cmd_parts.append(f"{dx_cm} {dy_cm_vla} {dz_cm_vla} {dyaw_deg}")
            try:
                yaw_rad_ccw = math.radians(self._get_current_relative_yaw_ccw_deg()) 
                c, s = math.cos(yaw_rad_ccw), math.sin(yaw_rad_ccw)
                body_x = dx_cm; body_y = -dy_cm_vla
                self.x += body_x * c - body_y * s
                self.y += body_x * s + body_y * c
            except Exception as e:
                print(f"[move_by_delta_pose_sequence] 航位推算更新失败: {e}")
            # if i == 0: 
            print(f"【第{i + 1}步】 (-> C++ fc_seq): dx={dx_cm:.1f}cm, dy={dy_cm_vla:.1f}cm(R), dz={dz_cm_vla:.1f}cm(D), dyaw={dyaw_deg:.1f}度(CW)")
        print("=="*10)
        full_cmd_str = " ".join(cmd_parts)
        
        resp = self._send_command(full_cmd_str) 
        
        if resp.get("status") == "ok":
            print(f"   -> C++ 'fc_seq' 序列命令已成功发送 (异步执行中)。")
        else:
            print(f"   -> 警告: C++ 'fc_seq' 序列命令发送失败: {resp.get('message')}")

    # ---  VLA  ---
    def _get_vla_proprio(self) -> np.ndarray:
        # (此函数保持不变)
        pose = self.get_current_pose()
        proprio_x_cm = pose['x']
        proprio_y_cm = -pose['y']
        proprio_z_cm = pose['z']
        yaw_deg_ccw = pose['yaw']
        proprio_yaw_deg = (yaw_deg_ccw + 180) % 360 - 180
        return np.array([proprio_x_cm, proprio_y_cm, proprio_z_cm, proprio_yaw_deg])

    def _convert_openvla_poses_to_deltas(self, poses_frd_ccw_rad: list) -> np.ndarray:
        # (此函数保持不变)
        deltas_frd_cw_deg = []
        last_pose_frd_ccw_deg = np.array([0.0, 0.0, 0.0, 0.0])
        for pose in poses_frd_ccw_rad:
            current_pose_frd_ccw_deg = np.array([ pose[0], pose[1], pose[2], np.degrees(pose[3]) ])
            delta_frd_ccw_deg = current_pose_frd_ccw_deg - last_pose_frd_ccw_deg
            delta_frd_cw_deg = delta_frd_ccw_deg
            delta_frd_cw_deg[3] = -delta_frd_ccw_deg[3]
            deltas_frd_cw_deg.append(delta_frd_cw_deg)
            last_pose_frd_ccw_deg = current_pose_frd_ccw_deg
        return np.array(deltas_frd_cw_deg)

    def vla(
        self, 
        instruction: str, 
        max_steps: int = 100, 
        speed: int = 40
    ):
        """
        VLA 控制的主逻辑。
        (此函数保持不变)
        """
        if self.vla_client is None:
            print("[VLA] 错误: VLA 客户端未初始化。无法执行指令。")
            self.talk("VLA 客户端未连接，任务取消")
            return
        try:
            print(f"[VLA] 开始执行 VLA 指令: {instruction}")
            first_image_rgb = self.get_frame()
            if first_image_rgb is None:
                print("[VLA] 错误: 无法获取 VLA 的第一帧图像。")
                return
            first_image = Image.fromarray(first_image_rgb)
            step_count = 0
            while True: 
                current_image_rgb = self.get_frame()
                if current_image_rgb is None:
                    print("[VLA] 错误: 无法获取 VLA 的当前帧图像。")
                    time.sleep(0.5)
                    continue
                current_image = Image.fromarray(current_image_rgb)
                proprio = self._get_vla_proprio()
                obs = { 'first_image': first_image, 'image': current_image, 'proprio': proprio, 'instr': instruction }
                print(f"[VLA] 请求 VLA 模型... (步骤 {step_count + 1}/{max_steps})")
                print(f"[VLA] Proprio (cm, deg): {proprio}")
                t_start = time.time()
                response = self.vla_client.get_action(obs) 
                t_end = time.time()
                print(f"[VLA] VLA 模型响应时间: {t_end - t_start:.2f}s")
                deltas_frd_cw_deg = None
                if self.vla_model_type == 'gr00t':
                    deltas_frd_cw_deg = response.get('action_ori')
                    print("gr00t =====================")
                    if deltas_frd_cw_deg:
                        for deltas_frd_cw_deg_ in deltas_frd_cw_deg: print("   ", deltas_frd_cw_deg_)
                    print("gr00t =====================")
                    if not deltas_frd_cw_deg:
                        print("[VLA] 警告: GR00T 未返回 'action_ori'，任务终止。"); break
                    print(f"[VLA] 收到 {len(deltas_frd_cw_deg)} 步 GR00T 增量，开始执行...")
                
                elif self.vla_model_type == 'openvla':
                    poses_frd_ccw_rad = response.get('action_ori')
                    print("openvla =====================")
                    if poses_frd_ccw_rad:
                        for poses_frd_ccw_rad_ in poses_frd_ccw_rad: print("   ", poses_frd_ccw_rad_)
                    print("openvla =====================")
                    if not poses_frd_ccw_rad:
                        print("[VLA] 警告: OpenVLA 未返回 'action_ori'，任务终止。"); break
                    print(f"[VLA] 收到 {len(poses_frd_ccw_rad)} 步 OpenVLA 姿态，转换为增量...")
                    deltas_frd_cw_deg = self._convert_openvla_poses_to_deltas(poses_frd_ccw_rad) 

                if deltas_frd_cw_deg is not None and len(deltas_frd_cw_deg) > 0:
                    self.move_by_delta_pose_sequence( np.array(deltas_frd_cw_deg), speed=speed )
                else:
                    print("[VLA] 未收到有效动作，任务终止。"); break
                step_count += 1
                if response.get('done', False):
                    print("[VLA] VLA 模型报告任务完成。"); self.talk("任务完成"); break
                if max_steps is not None and step_count >= max_steps:
                    print(f"[VLA] 已达到 {max_steps} 步最大限制，任务终止。"); self.talk("达到最大步数，任务停止"); break
        except (KeyboardInterrupt, InterruptedError):
            print("\n[VLA] 检测到中断。停止当前VLA任务..."); self.talk("任务已中断")
            self._send_command("fc_vel 0 0 0 0") 
        except Exception as e:
            print(f"[VLA] VLA 控制循环出错: {e}")
        finally:
            print("[VLA] VLA 线程已完成。")

    def fly_dynamic_kmz_mission(self, relative_points_ne: list, altitude: float = 50.0) -> bool:
        """
        根据相对于当前位置的 [北, 东] 偏移列表，动态生成并执行一个 KMZ 航线任务。
        此函数不包括起飞或降落。假定无人机已在空中并处于 N 挡。

        Args:
            relative_points_ne (list): 一个包含 [北(m), 东(m)] 偏移的列表。
                                       例如: [[50, 0], [0, 50]] 
                                       表示"先向北 50m，再从该点向东 50m"。
            altitude (float, optional): 整个航线的绝对飞行高度 (米)。
                                        默认为 50m。

        Returns:
            bool: 任务是否成功发送。
        """
        print(f"[fly_dynamic_kmz_mission] 开始执行动态 KMZ 任务...")
        
        # 1. 获取当前 GPS 位置作为航点 0
        pose_data = self._get_realtime_pose()
        if pose_data.get("lat") == 0.0 or pose_data.get("lon") == 0.0:
            print("[fly_dynamic_kmz_mission] 错误: 无法获取有效的 GPS 起始点。")
            return False
            
        start_lat = pose_data["lat"]
        start_lon = pose_data["lon"]
        start_alt = pose_data.get("alt_m", altitude) # 使用当前高度或指定高度
        
        print(f"  -> 起始点: Lat={start_lat:.6f}, Lon={start_lon:.6f}, Alt={start_alt:.1f}m")

        # 2. 计算所有绝对航点坐标
        path_coords = [(start_lon, start_lat, start_alt)] # (lon, lat, alt)
        current_lat, current_lon = start_lat, start_lon

        for i, (north_m, east_m) in enumerate(relative_points_ne):
            new_lat, new_lon = dji_kmz_mission_generator.calculate_new_gps(
                current_lat, current_lon, north_m, east_m
            )
            path_coords.append((new_lon, new_lat, start_alt)) 
            current_lat, current_lon = new_lat, new_lon
            print(f"  -> 航点 {i+1} (N:{north_m}m, E:{east_m}m): Lat={new_lat:.6f}, Lon={new_lon:.6f}")

        # 3. 准备 KMZ 文件内容
        drone_info = {'enum': 77, 'sub': 0}
        payload_info = {'enum': 66, 'sub': 0, 'pos': 0}
        
        template_xml = dji_kmz_mission_generator.generate_template_kml()
        wayline_xml = dji_kmz_mission_generator.generate_waylines_wpml(
            path_coords,
            drone_info,
            payload_info
        )
        
        # 4. 创建 KMZ 文件
        if not dji_kmz_mission_generator.create_kmz_file(
            template_xml, 
            wayline_xml, 
            dji_kmz_mission_generator.KMZ_SAVE_PATH
        ):
            print("[fly_dynamic_kmz_mission] 错误: KMZ 文件生成失败。")
            return False
            
        print(f"  -> 成功: 动态 KMZ 文件已保存到: {dji_kmz_mission_generator.KMZ_SAVE_PATH}")

        # 5. 发送 C++ Server 执行命令
        command = f"fc_pos_wp {dji_kmz_mission_generator.KMZ_SAVE_PATH}"
        resp = self._send_command(command)
        
        if resp.get("status") == "ok":
            print("[fly_dynamic_kmz_mission] 成功: 'fc_pos_wp' 命令已发送。")
            return True
        else:
            print(f"[fly_dynamic_kmz_mission] 错误: C++ Server 报告执行失败: {resp.get('message')}")
            return False

    def fly_dynamic_kmz_mission_gps(self, absolute_points_lla: list, use_current_pos_as_start: bool = True) -> bool:
        """
        根据一个 *绝对* GPS 坐标列表 (纬度, 经度, 高度)，动态生成并执行一个 KMZ 航线任务。
        此函数不包括起飞或降落。假定无人机已在空中并处于 N 挡。

        Args:
            absolute_points_lla (list): 一个包含 [纬度(deg), 经度(deg), 高度(m)] 的列表。
                                       例如: [[22.5, 113.9, 50.0], [22.6, 114.0, 50.0]]
            use_current_pos_as_start (bool, optional): 是否自动将无人机当前位置作为航线的第一个点 (航点 0)。
                                                       强烈建议保持 True。默认为 True。

        Returns:
            bool: 任务是否成功发送。
        """
        print(f"[fly_dynamic_kmz_mission_gps] 开始执行 (绝对 GPS) KMZ 任务...")

        # 1. 准备航点列表 (lon, lat, alt)
        path_coords = []
        
        if use_current_pos_as_start:
            # 1a. 获取当前 GPS 位置作为航点 0
            pose_data = self._get_realtime_pose()
            if pose_data.get("lat") == 0.0 or pose_data.get("lon") == 0.0:
                print("[fly_dynamic_kmz_mission_gps] 错误: 无法获取有效的 GPS 起始点。")
                return False
            
            start_lat = pose_data["lat"]
            start_lon = pose_data["lon"]
            start_alt = pose_data.get("alt_m", 50.0) # 使用当前高度
            
            path_coords.append((start_lon, start_lat, start_alt)) # 格式 (lon, lat, alt)
            print(f"  -> 起始点 (航点 0): Lat={start_lat:.6f}, Lon={start_lon:.6f}, Alt={start_alt:.1f}m")
        
        # 2. 转换并添加用户提供的绝对坐标
        if not absolute_points_lla:
             print("[fly_dynamic_kmz_mission_gps] 警告: 未提供航点。")
             if not use_current_pos_as_start:
                 print("[fly_dynamic_kmz_mission_gps] 错误: 航点列表为空且未使用当前位置。")
                 return False # 没有任何航点
        
        # 转换用户输入的 [lat, lon, alt] 为 (lon, lat, alt)
        for i, (lat, lon, alt) in enumerate(absolute_points_lla):
            path_coords.append((lon, lat, alt)) # 格式 (lon, lat, alt)
            print(f"  -> 航点 {i+1}: Lat={lat:.6f}, Lon={lon:.6f}, Alt={alt:.1f}m")

        # 3. 准备 KMZ 文件内容
        drone_info = {'enum': 77, 'sub': 0}
        payload_info = {'enum': 66, 'sub': 0, 'pos': 0}
        
        template_xml = dji_kmz_mission_generator.generate_template_kml()
        wayline_xml = dji_kmz_mission_generator.generate_waylines_wpml(
            path_coords,
            drone_info,
            payload_info
        )
        
        # 4. 创建 KMZ 文件
        if not dji_kmz_mission_generator.create_kmz_file(
            template_xml, 
            wayline_xml, 
            dji_kmz_mission_generator.KMZ_SAVE_PATH
        ):
            print("[fly_dynamic_kmz_mission_gps] 错误: KMZ 文件生成失败。")
            return False
            
        print(f"  -> 成功: 动态 KMZ 文件已保存到: {dji_kmz_mission_generator.KMZ_SAVE_PATH}")

        # 5. 发送 C++ Server 执行命令
        command = f"fc_pos_wp {dji_kmz_mission_generator.KMZ_SAVE_PATH}"
        resp = self._send_command(command)
        
        if resp.get("status") == "ok":
            print("[fly_dynamic_kmz_mission_gps] 成功: 'fc_pos_wp' 命令已发送。")
            return True
        else:
            print(f"[fly_dynamic_kmz_mission_gps] 错误: C++ Server 报告执行失败: {resp.get('message')}")
            return False
        
    def face_compare(self) -> bool:
        """
        人脸比较。立刻 get_frame 获得一张图片和 /home/dji/LMFly/UAV-Isaac-GR00T/uav_script/person_A.jpg 这张人图片对比相似度，
        大于50% (即 similarity > 0.5) 就判定是一个人，返回 True。

        Returns:
            bool: 如果是同一个人 (相似度 > 50%) 返回 True, 否则返回 False。
        """
        if LVFaceONNXInferencer is None:
            print("[face_compare] 错误: 'lvface_inferencer' 库未成功导入。")
            self.talk("人脸识别库未安装，无法比较")
            return False

        reference_image_path = "/home/dji/LMFly/UAV-Isaac-GR00T/uav_script/person_A.jpg"
        temp_current_frame_path = "/home/dji/LMFly/UAV-Isaac-GR00T/uav_script/current_drone_face.jpg"
        model_file = "/open_app/models/LVFace/LVFace-B_Glint360K.onnx" 

        # 1. 初始化推理器
        try:
            inferencer = LVFaceONNXInferencer(
                model_path=model_file,  # Path to your ONNX model
                use_gpu=True            # 假设环境支持 GPU
            )
        except Exception as e:
            print(f"[face_compare] 错误: 初始化 LVFaceONNXInferencer 失败: {e}")
            self.talk("人脸识别模块初始化失败")
            return False

        # 2. 获取无人机当前帧并保存
        print("[face_compare] 正在从无人机获取当前帧 (使用 get_frame)...")
        # get_frame() 返回 480x360 RGB numpy 数组
        current_frame_rgb = self.get_frame_vlm() 
        
        if current_frame_rgb is None:
            print("[face_compare] 错误: 从 get_frame() 未能获取图像。")
            self.talk("获取当前图像失败")
            return False

        # 将 RGB 帧转换为 BGR (LVFace 可能期望 BGR，但更重要的是保存)
        # current_frame_bgr = cv2.cvtColor(current_frame_rgb, cv2.COLOR_RGB2BGR)
        try:
            cv2.imwrite(temp_current_frame_path, current_frame_rgb)
        except Exception as e:
            print(f"[face_compare] 错误: 保存当前帧到临时文件失败: {e}")
            self.talk("保存临时图像失败")
            return False
            
        # 3. 提取特征
        try:
            # 提取参考图片特征
            if not os.path.exists(reference_image_path):
                print(f"[face_compare] 错误: 参考图片未找到: {reference_image_path}")
                self.talk("未找到参考照片")
                return False
                
            print("[face_compare] 正在提取参考图片特征...")
            feat_ref = inferencer.infer_from_image(reference_image_path)
            if feat_ref is None:
                 print("[face_compare] 错误: 无法从参考图片中提取人脸特征。")
                 self.talk("无法从参考照片中提取人脸")
                 return False

            # 提取当前帧特征
            print("[face_compare] 正在提取当前帧特征...")
            feat_current = inferencer.infer_from_image(temp_current_frame_path)
            if feat_current is None:
                 print("[face_compare] 警告: 无法从当前帧中提取人脸特征。")
                 self.talk("当前画面中未发现人脸")
                 return False

            # 4. 计算相似度
            similarity = inferencer.calculate_similarity(feat_ref, feat_current)
            
            print(f"[face_compare] 相似度分数: {similarity:.6f}")

            # 5. 判断
            similarity_threshold = 0.3
            if similarity > similarity_threshold:
                self.talk(f"匹配成功，相似度 {similarity:.2f} 大于百分之五十")
                return True
            else:
                self.talk(f"未找到匹配的人，相似度 {similarity:.2f} 不足百分之五十")
                return False

        except Exception as e:
            print(f"[face_compare] 人脸识别过程中发生错误: {e}")
            self.talk("人脸识别过程中发生错误")
            return False
        # finally:
             # 清理临时文件
            # if os.path.exists(temp_current_frame_path):
            #     os.remove(temp_current_frame_path)
       
    def shutdown(self):
        """
        关闭与 C++ PSDKServer 的持久连接。
        """
        print("DjiM4DDrone wrapper shutting down...")
        with self.command_lock:
            if self.socket:
                try:
                    self.socket.close()
                except Exception as e:
                    print(f"[shutdown] 关闭套接字时出错: {e}")
                self.socket = None
        print("DjiM4DDrone wrapper shutdown complete.")
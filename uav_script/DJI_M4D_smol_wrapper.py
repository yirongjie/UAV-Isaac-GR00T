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
import httpx
from aip import AipFace
from typing import Dict, Any

import glob
import os
import torch
import cv2
import dotenv
import numpy as np
from openai import OpenAI
from ultralytics import YOLO
# --- 新增阿里云 FaceBody imports ---
from alibabacloud_facebody20191230.client import Client as AliClient
from alibabacloud_facebody20191230.models import CompareFaceAdvanceRequest
from alibabacloud_tea_openapi.models import Config as AliConfig
from alibabacloud_tea_util.models import RuntimeOptions
# --------------------------------

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

try:
    from depth_anything_v2.dpt import DepthAnythingV2 # <-- 新增
except ImportError:
    print("[Drone Wrapper] 警告: 未找到 'depth_anything_v2'。move_to_person_2 (深度模型) 将不可用。")
    DepthAnythingV2 = None
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}











API_VL_MODEL = "qwen3-vl-30b-a3b-instruct"# "qwen/qwen3-vl-30b-a3b-instruct"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" #"https://openrouter.ai/api/v1"
API_KEY = os.environ["OPENROUTER_API_KEY"]


# API_VL_MODEL = "gpt-3.5-turbo"
# BASE_URL = "http://0.0.0.0:9999/v1"
# API_KEY = "sk-fake-key"


BAIDU_API_APP_ID = os.environ["BAIDU_API_APP_ID"]
BAIDU_API_KEY = os.environ["BAIDU_API_KEY"]
BAIDU_SECRET_KEY = os.environ["BAIDU_SECRET_KEY"]

ALIBABA_CLOUD_ACCESS_KEY_ID=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"]
ALIBABA_CLOUD_ACCESS_KEY_SECRET=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"]

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
        self.llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=300.0)
        
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

        # self.yolo_model = None
        
        self.yolo_model_path = "/home/dji/LMFly/UAV-Isaac-GR00T/uav_script/yolov11n-face.pt" # 确保此文件在运行目录下，或写绝对路径
        self.yolo_model = YOLO(self.yolo_model_path).to("cuda") 

        
        self.inferencer = LVFaceONNXInferencer(model_path="/open_app/models/LVFace/LVFace-B_Glint360K.onnx", use_gpu=True)

        self.depth_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.depth_model = None # 懒加载 (Lazy load)
        if self.depth_model is None:
            if not self._load_depth_model():
                print("[_get_xyz_from_depth_model] 错误: 深度模型无法加载。")



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
            self.socket.settimeout(500.0) 
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
                
                self.socket.settimeout(500.0) 
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
                    self.socket.settimeout(500.0)
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
                    self.socket.settimeout(500.0)
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

    def get_frame_vlm(self, res_param: int = 480) -> np.ndarray:
        """Get the current frame from the drone. (For VLM models)
        此函数返回一个 BGR 图像。图像的实际分辨率取决于 C++ Server 对 res_param 的响应，
        通常为 480p 或 1080p（如果 res_param=1080）。

        Args:
            res_param (int, optional): 决定向 C++ Server 请求的图像分辨率参数。
                                       有效值包括 480 (默认) 或 1080。Defaults to 480.

        Returns:
            np.ndarray: The processed frame as a NumPy array (BGR format)。
                        图像尺寸取决于 Server 返回的实际分辨率。
        """
        if res_param not in [480, 720, 1080, 2160, -1]:
            print(f"[get_frame_vlm] 警告: 请求了不支持的分辨率参数 '{res_param}'。将使用默认 480。")
            res_param = 480
            
        command = f"tp {res_param}"
        print(f"[get_frame_vlm] 正在请求分辨率参数为 {res_param} 的源图像...")

        resp = self._send_command(command)
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
            
        # 关键修改：移除强制 resize
        print(f"[get_frame_vlm] 成功获取 BGR 图像，实际尺寸: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}")
        cv2.imwrite("current_frame_vlm.png", frame_bgr) 
        return frame_bgr
    
    def get_frame_vlm_fake(self, res_param: int = 480) -> np.ndarray:
        """
        [!! FAKE !!] 模拟 get_frame_vlm。
        此函数不调用摄像头，而是按顺序循环返回 /home/dji/LMFly/UAV-Isaac-GR00T/uav_script/example_pic/ 目录中的图片。
        
        Get the current frame from the drone. (For VLM models)
        此函数返回一个 BGR 图像。

        Args:
            res_param (int, optional): (此模拟函数中未使用) 决定向 C++ Server 请求的图像分辨率参数。
                                       有效值包括 480 (默认) 或 1080。Defaults to 480.

        Returns:
            np.ndarray: The processed frame as a NumPy array (BGR format)。
                        图像尺寸取决于读取的图片文件。
        """
        
        # 1. 检查是否已初始化 (懒加载)
        if not hasattr(self, 'fake_image_files'):
            print("[get_frame_vlm_fake] 首次调用，正在初始化模拟图片列表...")
            # --- Fake Image Loader for get_frame_vlm_fake ---
            fake_image_dir = "/home/dji/LMFly/UAV-Isaac-GR00T/uav_script/example_pic"
            # 按照您提供的已知文件列表
            file_names = ["ex1.png", "ex2.png", "ex0.png", "ex3.png", "ex4.png"]
            self.fake_image_files = [os.path.join(fake_image_dir, f) for f in file_names]
            self.fake_image_index = 0
            
            if not self.fake_image_files:
                print(f"[get_frame_vlm_fake] 警告: 未能构建 'example_pic' 模拟图片列表。")
            else:
                print(f"[get_frame_vlm_fake] 'get_frame_vlm_fake' 已初始化, 将循环 {len(self.fake_image_files)} 张图片。")
            # --- End Fake Image Loader ---

        # 2. 检查是否有可用的模拟图片
        if not self.fake_image_files:
            print("[get_frame_vlm_fake] 错误: 模拟图片列表为空。请检查路径配置。")
            return None

        # 3. 获取当前要返回的图片路径
        image_path = self.fake_image_files[self.fake_image_index]
        
        # 4. 更新索引，用于下一次调用 (循环)
        self.fake_image_index = (self.fake_image_index + 1) % len(self.fake_image_files)

        # 5. 读取图片
        print(f"[get_frame_vlm_fake] 正在读取模拟图片: {image_path}")
        try:
            frame_bgr = cv2.imread(image_path)
            
            if frame_bgr is None:
                print(f"[get_frame_vlm_fake] 错误: cv2.imread 无法读取 {image_path}")
                # 尝试读取列表中的下一个，以防只有一个文件损坏
                if len(self.fake_image_files) > 1:
                     print("[get_frame_vlm_fake] 尝试读取下一个...")
                     # 递归调用一次 (注意：如果所有文件都损坏，可能导致栈溢出，但对于4个文件的小列表是安全的)
                     return self.get_frame_vlm_fake(res_param)
                return None
        
        except Exception as e:
            print(f"[get_frame_vlm_fake] 读取模拟图片时出错: {e}")
            return None

        # 6. 返回 BGR 图像 (不调整大小，模拟原函数的行为)
        print(f"[get_frame_vlm_fake] 成功获取 BGR 模拟图像，实际尺寸: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}")
        # 模拟原函数保存调试图片
        cv2.imwrite("current_frame_vlm.png", frame_bgr) 
        return frame_bgr
    # --- [!! 结束 !!] ---

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

    def move_forward(self, distance: int) -> None:
        """
        Move the drone forward by a specified distance in centimeters.

        Args:
            distance (int): The distance to move forward in centimeters. 必须是正数
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
            distance (int): The distance to move backward in centimeters. 必须是正数
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
        return response.choices[0].message.content # type: ignore
    
    def objects_vlm(self, obj_name_list: list) -> list:
        """
        让大模型（视觉语言模型/VLM）检测当前无人机视角下的指定物体，并根据图像内容估算每个物体距离无人机的距离和角度。
        逻辑已更新：先获取 1080P 图像，将其按比例缩放（长边为 480P）供 VLM 使用，并从 1080P 图像上执行最终的 2D 裁剪。

        Args:
            obj_name_list (list): 需要检测和估算距离的物体名称列表。例如：["显示器", "蓝色球"]。

        Returns:
            list: 每个物体的检测结果，包含：
                - name: 物体名称 (str)
                - x: 物体中心点在摄像头坐标系下的x轴距离（单位：米，float）
                - y: 物体中心点在摄像头坐标系下的y轴距离（单位：米，float）
                - z: 物体中心点在摄像头坐标系下的z轴距离（单位：米，float，通常为深度）
        """
        # 1. 获取 1080P (或 Server 返回的最高分辨率) BGR 图像
        bgr_image_full_res = self.get_frame_vlm(res_param=-1)
        if bgr_image_full_res is None:
            print("objects_vlm 错误：未能获取 VLM 图像。")
            return []
        
        # 获取全分辨率图像尺寸
        H_full, W_full = bgr_image_full_res.shape[:2]
        print(f"[objects_vlm] 原始图像尺寸: {W_full}x{H_full}")
        
        # 2. 缩放图像至 VLM 尺寸，保持长宽比 (长边为 480)
        MAX_VLM_SIDE = 480 
        
        if W_full >= H_full: # 横屏或正方形 (以宽度为基准)
            W_vlm = MAX_VLM_SIDE
            H_vlm = int(H_full * (MAX_VLM_SIDE / W_full))
        else: # 竖屏 (以高度为基准)
            H_vlm = MAX_VLM_SIDE
            W_vlm = int(W_full * (MAX_VLM_SIDE / H_full))
            
        bgr_image_vlm = cv2.resize(bgr_image_full_res, (W_vlm, H_vlm))
        print(f"[objects_vlm] VLM 使用图像尺寸 (按比例缩放): {W_vlm}x{H_vlm}")

        # 比例因子：用于将 VLM 坐标映射回 1080P 坐标
        scale_x = W_full / W_vlm
        scale_y = H_full / H_vlm
        
        # 3. 准备 VLM 输入
        base64_rgb_str = _cv2_to_base64(bgr_image_vlm, ".png") # 使用 VLM 缩放后的图像编码
        
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
            # 4. 解析 VLM 响应
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
            
            # 使用 VLM 图像进行标注（current_frame_xyz.png）
            annotated_img = bgr_image_vlm.copy(); save_annotated = False 
            
            # VLM 图像的尺寸 (h, w 用于 3D 投影计算)
            h, w = H_vlm, W_vlm 
            f = 500.0; cx, cy = w / 2, h / 2
            category_colors = {}; colors = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 0),(255, 0, 255),(0, 255, 255),]
            
            processed = []
            if result and isinstance(result, list) and len(result) > 0:
                save_annotated = True
                
                for obj_id, obj in enumerate(result):
                    bbox = obj.get("bbox_3d", []); label = obj.get("label", "unknown")
                    if len(bbox) < 9: continue
                    x_c, y_c, z_c = bbox[0], bbox[1], bbox[2]; dx, dy, dz = bbox[3], bbox[4], bbox[5]; roll, pitch, yaw = bbox[6], bbox[7], bbox[8]
                    
                    # 1. 3D 边界框计算和绘制 (基于 VLM 图像的投影)
                    if label not in category_colors: category_colors[label] = colors[len(category_colors) % len(colors)]
                    color = category_colors[label]
                    corners_local = np.array([[dx/2,dy/2,dz/2],[dx/2,dy/2,-dz/2],[dx/2,-dy/2,dz/2],[dx/2,-dy/2,-dz/2],[-dx/2,dy/2,dz/2],[-dx/2,dy/2,-dz/2],[-dx/2,-dy/2,dz/2],[-dx/2,-dy/2,-dz/2],])
                    R_x = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]]); R_y = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]]); R_z = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]]); R = R_z @ R_y @ R_x
                    corners_3d = (R @ corners_local.T).T + np.array([x_c, y_c, z_c]); corners_2d = []
                    
                    # 2. 投影到 VLM 图像
                    min_u, max_u, min_v, max_v = w, 0, h, 0
                    valid_projection = False
                    for corner in corners_3d:
                        if corner[2] <= 0: corners_2d.append(None); continue
                        u = int(f * corner[0] / corner[2] + cx); v = int(f * corner[1] / corner[2] + cy); corners_2d.append((u, v))
                        min_u = min(min_u, u); max_u = max(max_u, u)
                        min_v = min(min_v, v); max_v = max(max_v, v)
                        valid_projection = True

                    # 3. 绘制 3D 边界框 (在 VLM 图像上)
                    lines = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
                    for i, j in lines:
                        if corners_2d[i] and corners_2d[j]: cv2.line(annotated_img, corners_2d[i], corners_2d[j], color, 2)
                    
                    # 4. 绘制标签和中心点
                    valid_corners = [c for c in corners_2d if c is not None]
                    if valid_corners: 
                        centroid = np.mean(valid_corners, axis=0).astype(int)
                        cv2.putText(annotated_img, label, (centroid[0], centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 5. 裁剪和保存 2D 图像 (从 1080P 图像上裁剪)
                    if valid_projection:
                        # 扩展区域（VLM 图像上的 PADDING）
                        PADDING = 3
                        
                        # 1) 计算 VLM 图像上的裁剪区域
                        crop_x_min_vlm = max(0, min_u - PADDING)
                        crop_y_min_vlm = max(0, min_v - PADDING)
                        crop_x_max_vlm = min(w, max_u + PADDING)
                        crop_y_max_vlm = min(h, max_v + PADDING)

                        # 2) 将 VLM 坐标按比例映射回 1080P 坐标
                        # 使用 int() 进行向下取整，确保像素索引有效
                        crop_x_min_full = int(crop_x_min_vlm * scale_x)
                        crop_y_min_full = int(crop_y_min_vlm * scale_y)
                        crop_x_max_full = int(crop_x_max_vlm * scale_x)
                        crop_y_max_full = int(crop_y_max_vlm * scale_y)
                        
                        # 3) 确保 1080P 裁剪区域在图像边界内
                        crop_x_min_full = max(0, crop_x_min_full)
                        crop_y_min_full = max(0, crop_y_min_full)
                        crop_x_max_full = min(W_full, crop_x_max_full)
                        crop_y_max_full = min(H_full, crop_y_max_full)
                        
                        # 4) 裁剪 1080P 图像
                        cropped_img = bgr_image_full_res[crop_y_min_full:crop_y_max_full, crop_x_min_full:crop_x_max_full]
                        
                        # 5) 保存文件
                        safe_label = label.replace(" ", "_")
                        crop_file_path = f"current_frame_vlm_{safe_label}_{obj_id}.png"
                        cv2.imwrite(crop_file_path, cropped_img)
                        print(f"Saved 2D crop from 1080P: {crop_file_path}")
                        
                    # 6. 处理结果列表
                    if len(bbox) >= 3:
                        processed.append({ 
                            "name": label, 
                            "x": round(float(bbox[0]), 3), 
                            "y": round(float(bbox[1]), 3), 
                            "z": round(float(bbox[2]), 3), 
                        })

            # 7. 保存带 3D 框的完整图像 (VLM 缩放分辨率)
            if save_annotated: 
                cv2.imwrite("current_frame_xyz.png", annotated_img)
                print(f"Saved 3D bounding box visualization (VLM scale): current_frame_xyz.png")
            
            return processed
        except Exception as e:
            print(f"解析VLM响应时出错: {e}")
            return []
        
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
            print(f"正在前进 {move_distance_cm} cm。")
            self.move_forward(min(move_distance_cm, 5000))
            actions_taken.append(f"前进 {move_distance_cm} cm")
        elif move_distance_cm < -20:
            print(f"正在后退 {abs(move_distance_cm)} cm。"); self.move_backward(min(abs(move_distance_cm), 5000)); actions_taken.append(f"后退 {abs(move_distance_cm)} cm")
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
    
    def face_compare(self, local: bool = False, target_image_path: str = None, use_api: bool = False, detection_method: str = "vlm") -> bool:
        """
        统一的人脸比较函数。
        1. 获取当前图像。
        2. 检测并裁剪人脸 (支持 VLM 或 YOLO)。
        3. 进行特征比对 (支持 本地ONNX 或 百度API)。

        Args:
            local (bool): 调试模式，使用本地固定图片。
            target_image_path (str): 指定图片路径。
            use_api (bool): True=百度API, False=本地ONNX。
            detection_method (str): "vlm" (原有逻辑) 或 "yolo" (使用 ultralytics)。

        Returns:
            bool: 是否匹配成功。
        """
        # --- 0. 基础配置 ---
        reference_image_path = "/home/dji/LMFly/UAV-Isaac-GR00T/uav_script/person_A.jpg"
        temp_original_path = ""

        # --- 1. 获取/确定原始图像 ---
        if target_image_path:
            temp_original_path = target_image_path
            print(f"[face_compare] 使用外部图片: {temp_original_path}")
        elif local:
            temp_original_path = "/home/dji/LMFly/UAV-Isaac-GR00T/uav_script/current_frame_vlm.png"
            print(f"[face_compare] 使用本地调试图片: {temp_original_path}")
        else:
            temp_original_path = "/home/dji/LMFly/UAV-Isaac-GR00T/uav_script/current_drone_face_original.jpg"
            # print("[face_compare] 正在获取当前帧...")
            current_frame_bgr = self.get_frame_vlm()
            if current_frame_bgr is None:
                self.talk("获取图像失败")
                return False
            cv2.imwrite(temp_original_path, current_frame_bgr)

        temp_cropped_face_path = temp_original_path.replace(".png", "_face.png").replace(".jpg", "_face.jpg")
        original_bgr = cv2.imread(temp_original_path)
        if original_bgr is None:
            print(f"[face_compare] 无法读取图片: {temp_original_path}")
            return False
        
        H, W = original_bgr.shape[:2]
        
        # ==========================================
        # 分支 1: 使用 YOLO 进行人脸检测与裁剪
        # ==========================================
        if detection_method == "yolo":
            try:
                from ultralytics import YOLO
                
                # 懒加载模型
                if self.yolo_model is None:
                    print(f"[face_compare] 首次加载 YOLO 模型: {self.yolo_model_path} ...")
                    self.yolo_model = YOLO(self.yolo_model_path).to("cuda") 
                
                # 推理
                # print("[face_compare] 正在运行 YOLO 检测...")
                results = self.yolo_model.predict(source=original_bgr, conf=0.25, max_det=100, verbose=False)
                
                best_face_crop = None
                max_area = 0
                scale = 1.2 # 放大系数
                
                # 寻找最大的人脸 (Most prominent)
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        area = (x2 - x1) * (y2 - y1)
                        
                        if area > max_area:
                            max_area = area
                            # 计算中心和宽高 (应用 scale)
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            w = (x2 - x1) * scale
                            h = (y2 - y1) * scale
                            
                            new_x1 = max(int(cx - w/2), 0)
                            new_y1 = max(int(cy - h/2), 0)
                            new_x2 = min(int(cx + w/2), W)
                            new_y2 = min(int(cy + h/2), H)
                            
                            best_face_crop = original_bgr[new_y1:new_y2, new_x1:new_x2]

                if best_face_crop is not None and best_face_crop.size > 0:
                    cv2.imwrite(temp_cropped_face_path, best_face_crop)
                    cH, cW = best_face_crop.shape[:2]
                    print(f"[face_compare] YOLO 裁剪完成: {cW}x{cH}")
                else:
                    print("[face_compare] YOLO 未检测到人脸。")
                    self.talk("未检测到人脸")
                    return False

            except ImportError:
                print("[face_compare] 错误: 未安装 ultralytics 库。")
                return False
            except Exception as e:
                print(f"[face_compare] YOLO 检测出错: {e}")
                return False

        # ==========================================
        # 分支 2: 使用 VLM 进行人脸检测与裁剪 (原有逻辑)
        # ==========================================
        else: 
            # print("[face_compare] 正在调用 VLM 检测人脸...")
            try:
                base64_rgb_str = _cv2_to_base64(original_bgr, ".png")
                prompt = f"""
                Detect the **most prominent face** in the image.
                - If faces found, output JSON: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} (normalized 0-1000).
                - If no face, output: {{}}
                """
                response = self.llm_client.chat.completions.create(
                    model=API_VL_MODEL,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_rgb_str}"}}]}],
                    temperature=0.01,
                )
                raw_response = response.choices[0].message.content
                
                try:
                    vlm_result = json.loads(raw_response.strip())
                except:
                    start = raw_response.find("{"); end = raw_response.rfind("}")
                    vlm_result = json.loads(raw_response[start:end+1]) if start != -1 else {}

                bbox = vlm_result.get("bbox_2d")
                if not bbox or len(bbox) != 4:
                    print("[face_compare] VLM 未检测到人脸。")
                    self.talk("未检测到清晰人脸")
                    return False

                x1_n, y1_n, x2_n, y2_n = bbox
                # 坐标转换 + Padding (原有 VLM 逻辑的 10% padding)
                x1 = int(x1_n * W / 1000.0); y1 = int(y1_n * H / 1000.0)
                x2 = int(x2_n * W / 1000.0); y2 = int(y2_n * H / 1000.0)
                
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                
                cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
                cx2 = min(W, x2 + pad_x); cy2 = min(H, y2 + pad_y)
                
                cropped_img = original_bgr[cy1:cy2, cx1:cx2]
                if cropped_img.size == 0: return False
                cv2.imwrite(temp_cropped_face_path, cropped_img)
                print(f"[face_compare] VLM 裁剪完成。")

            except Exception as e:
                print(f"[face_compare] VLM 处理出错: {e}")
                return False

        # ==========================================
        # 3. 开始比对 (API 或 Local) - 公共部分
        # ==========================================
        if use_api:
            # --- 百度 API ---
            # print("[face_compare] 正在使用百度 API 比对...")
            try:
                from aip import AipFace
                
                client = AipFace(_API_APP_ID, _API_KEY, _SECRET_KEY) # 建议提取为常量
                
                with open(reference_image_path, 'rb') as f: ref_b64 = base64.b64encode(f.read()).decode()
                with open(temp_cropped_face_path, 'rb') as f: cur_b64 = base64.b64encode(f.read()).decode()
                
                images = [
                    {'image': ref_b64, 'image_type': 'BASE64', 'face_type': 'LIVE', 'quality_control': 'LOW'},
                    {'image': cur_b64, 'image_type': 'BASE64', 'face_type': 'LIVE', 'quality_control': 'LOW'}
                ]
                res = client.match(images)
                
                if res.get('error_code') == 0:
                    score = res['result']['score']
                    print(f"[face_compare] API 相似度: {score:.2f}")
                    if score > 50.0:
                        self.talk(f"匹配成功，相似度 {score:.1f}")
                        return True
                else:
                    if res.get('error_msg') == "face is fuzzy":
                        pass ## 应该怎么办呢？
                    print(f"[face_compare] API 错误: {res.get('error_msg')}")
                    
            except Exception as e:
                print(f"[face_compare] API 出错: {e}")

        else:
            # --- 本地 ONNX ---
            try:
                feat_ref = self.inferencer.infer_from_image(reference_image_path)
                feat_cur = self.inferencer.infer_from_image(temp_cropped_face_path)
                
                if feat_ref is not None and feat_cur is not None:
                    sim = self.inferencer.calculate_similarity(feat_ref, feat_cur)
                    print(f"[face_compare] 本地相似度: {sim:.4f}")
                    if sim > 0.5:
                        self.talk(f"匹配成功，相似度 {sim*100:.1f}")
                        return True
            except Exception as e:
                print(f"[face_compare] 本地比对出错: {e}")

        return False
    

    def check_current_view(self) -> dict:
        """(内部函数) 检查当前单一视图"""
        print("[scan_for_person] 正在检查当前视图 (VLM find person + face_compare verify)...")
        
        # 查找并删除所有名为 'current_frame_vlm_person_*.png' 的文件
        # for old_crop_file in glob.glob("current_frame_vlm_person_*.png"):
        #     try:
        #         os.remove(old_crop_file)
        #         # print(f"已删除旧裁剪文件: {old_crop_file}") # 调试信息
        #     except OSError as e:
        #         print(f"警告: 删除文件 {old_crop_file} 失败: {e}")


        # 1. VLM 检测所有人 (objects_vlm 会自动获取 1080P 图像，并缩放 480P 供 VLM 使用)
        #    objects_vlm 会生成并保存裁剪图文件 current_frame_vlm_person_{ID}.png
        all_persons = self.objects_vlm(["person"]) 
        
        if not all_persons:
            print("[scan_for_person] VLM 未检测到 'person'。")
            return None
            
        print(f"[scan_for_person] VLM 找到 {len(all_persons)} 个 'person'。正在迭代调用 face_compare 验证...")

        # 2. 遍历 VLM 找到的所有 'person'
        for obj_id, person_data in enumerate(all_persons):
            # objects_vlm 裁剪图的文件命名规则
            crop_file_path = f"current_frame_vlm_person_{obj_id}.png"
            
            # 3. 调用修改后的 face_compare 进行验证，传入裁剪图路径
            try:
                # is_person_A = self.face_compare(local=True, target_image_path=crop_file_path)
                is_person_A = self.face_compare(local=True, target_image_path=crop_file_path, use_api=True, detection_method="yolo")
            except Exception as e:
                print(f"[scan_for_person] face_compare() 执行时出错: {e}")
                continue # 继续检查下一个 person
            
            # 4. 检查验证结果
            if is_person_A:
                print(f"[scan_for_person] face_compare 验证成功! VLM ID {obj_id} 匹配 person_A。")
                
                x_m = person_data.get("x", 0)
                y_m = person_data.get("y", 0)
                z_m = person_data.get("z", 0)

                if z_m <= 0:
                        print(f"[scan_for_person] face_compare 成功, 但 VLM 返回无效 Z 距离 ({z_m})。")
                        return None
                
                return {"x": x_m, "y": y_m, "z": z_m}
            else:
                print(f"[scan_for_person] VLM ID {obj_id} 验证失败。")
        
        # 5. 所有 VLM 找到的人都未通过人脸识别
        print("[scan_for_person] 当前视图中所有找到的 'person' 都不是 person_A。")
        return None
        
    def scan_for_person(self) -> dict:
        """
        实现 "VLM检测person + 独立face_compare验证" 逻辑来寻找 person_A。
        在每个角度：
        1. 调用 objects_vlm(["person"]) 获取所有人的 3D 框，并生成裁剪图。
        2. 遍历 VLM 找到的每个 "person" 的裁剪图，调用 face_compare() 进行验证。
        3. 如果 face_compare() 返回 True (匹配成功)，则返回该 person 的 3D 坐标。
        
        如果 360 度扫描（包括平移）后未找到，返回 None。

        Returns:
            dict: 包含 person_A 坐标的字典 (e.g., {'x': 0.1, 'y': 0.2, 'z': 2.5})，
                  如果未找到则返回 None。
        """
        
        
        
        # --- scan_for_person 主循环 (原地、左侧、右侧扫描，代码保持不变) ---

        # (M4D) 原地360度扫描 (每次旋转 90 度，总共 4 次)
        for i in range(6):
            result_3d = self.check_current_view()
            if result_3d:
                print(f"[scan_for_person] 在第 {i+1} 次尝试 (旋转 {i*60} 度) 时找到 person_A。")
                return result_3d
            print(f"[scan_for_person] 第 {i+1}/6 次尝试 (原地) 未找到 person_A，逆时针旋转 90 度...")
            self.turn_counter_clockwise(60) # 修正旋转 90 度

        # (M4D) 向左 1.5m, 扫描360度
        print("[scan_for_person] 原地扫描失败。向左移动 150cm...")
        self.move_left(150) 
        for i in range(6):
            result_3d = self.check_current_view()
            if result_3d:
                print(f"[scan_for_person] 在左侧 {i+1} 次尝试时找到 person_A。")
                return result_3d
            print(f"[scan_for_person] 第 {i+1}/6 次尝试 (左侧) 未找到 person_A，逆时针旋转 90 度...")
            self.turn_counter_clockwise(60) # 修正旋转 90 度

        # (M4D) 向右 3m, 扫描360度
        print("[scan_for_person] 左侧扫描失败。向右移动 300cm (穿过原点)...")
        self.move_right(300) 
        for i in range(6):
            result_3d = self.check_current_view()
            if result_3d:
                print(f"[scan_for_person] 在右侧 {i+1} 次尝试时找到 person_A。")
                return result_3d
            print(f"[scan_for_person] 第 {i+1}/6 次尝试 (右侧) 未找到 person_A，逆时针旋转 90 度...")
            self.turn_counter_clockwise(60) # 修正旋转 90 度

        # (M4D) 回到原点
        print("[scan_for_person] 右侧扫描失败。返回起始水平位置...")
        self.move_left(150)

        print("[scan_for_person] 所有扫描尝试均失败。")
        return None
        
    def move_to_person(
        self, target_distance_cm: int, target_height_cm: int = 0
    ) -> bool:
        """
        [!! 新增 !!]
        扫描、对准并移动到特定人员 (person_A) 前方的特定距离和高度。
        该函数会执行以下四步操作：
        1. 使用 scan_for_person 找到 person_A (结合了 VLM 2D/3D 检测 和 人脸识别)。
        2. 根据 person_A 的x坐标，自动旋转无人机，将其对准到视野中心。
        3. 根据 person_A 的y坐标，自动升降无人机，将其对准到视野中心。
        4. 根据 person_A 的z坐标，前进到目标距离。

        Args:
            target_distance_cm (int): 最终希望与 person_A 保持的距离（单位：厘米）。
            target_height_cm (int, optional): 希望与 person_A 保持的垂直距离（单位：厘米），默认0。

        Returns:
            bool: 操作是否成功完成。成功返回True, 如果 person_A 未找到或移动失败，返回 False。
        """
        print(f"开始执行 move_to_person: 目标 'person_A', 距离 {target_distance_cm}cm")
        
        # [!!] 调用新的 scan_for_person
        scan_result_3d = self.scan_for_person() 
        
        if scan_result_3d is None:
            print(f"未找到目标 'person_A'，无法执行 move_to_person。")
            self.talk("未找到目标人物")
            return False
            
        # 目标已确认是 person_A
        x_m = scan_result_3d.get("x", 0)
        y_m = scan_result_3d.get("y", 0)
        z_m = scan_result_3d.get("z", 0)
        
        if z_m <= 0:
            print(f"VLM 返回无效 Z 距离 ({z_m})，无法计算移动。")
            self.talk("目标距离无效")
            return False
            
        actions_taken = []
        horizontal_threshold_m = 0.05
        
        # 1. 对准 X (左右)
        if abs(x_m) > horizontal_threshold_m:
            angle_rad = math.atan(x_m / z_m); angle_deg = int(math.degrees(angle_rad))
            if angle_deg > 0:
                print(f"目标在右侧，需向右转 {angle_deg} 度。")
                self.turn_clockwise(angle_deg); actions_taken.append(f"向右转 {angle_deg} 度")
            elif angle_deg < 0:
                print(f"目标在左侧，需向左转 {abs(angle_deg)} 度。")
                self.turn_counter_clockwise(abs(angle_deg)); actions_taken.append(f"向左转 {abs(angle_deg)} 度")
                
        # 2. 对准 Y (上下)
        vertical_threshold_m = 0.05
        vertical_error_m = y_m - (target_height_cm / 100.0); move_z_cm = int(vertical_error_m * 100)
        
        if abs(move_z_cm) > int(vertical_threshold_m * 100):
            if move_z_cm > 0:
                print(f"目标在下方，需下降 {move_z_cm} cm。"); self.move_down(move_z_cm); actions_taken.append(f"下降 {move_z_cm} cm")
            else:
                print(f"目标在上方，需上升 {abs(move_z_cm)} cm。"); self.move_up(abs(move_z_cm)); actions_taken.append(f"上升 {abs(move_z_cm)} cm")
                
        # 3. 对准 Z (前后)
        current_distance_cm = int(z_m * 100); move_distance_cm = current_distance_cm - target_distance_cm
        
        if move_distance_cm > 20: # 最小移动阈值
            print(f"正在前进 {move_distance_cm} cm。"); self.move_forward(min(move_distance_cm, 500)); actions_taken.append(f"前进 {move_distance_cm} cm")
        elif move_distance_cm < -20: # 最小移动阈值
            print(f"正在后退 {abs(move_distance_cm)} cm。"); self.move_backward(min(abs(move_distance_cm), 500)); actions_taken.append(f"后退 {abs(move_distance_cm)} cm")
            
        if not actions_taken:
            print("person_A 已经在目标位置，无需移动。"); return False
        else:
            print("完成 move_to_person 操作，动作包括: " + ", ".join(actions_taken))
            self.talk("已移动到目标人物面前")
            return True
        

    ###################################
    ###################################
    ###################################
    ###################################

    def _load_depth_model(self):
        """
        (内部辅助) 懒加载 DepthAnythingV2 模型。
        """
        if self.depth_model is not None:
            return True
            
        if DepthAnythingV2 is None:
            print("[_load_depth_model] 错误: DepthAnythingV2 库未导入。")
            return False

        print("[_load_depth_model] 正在首次加载 DepthAnythingV2 (vits, vkitti)...")
        try:
            # --- 参数 (来自 main_depth_test2.py) ---
            encoder = 'vits' 
            dataset = 'vkitti' 
            max_depth = 20 
            load_from = f'/open_app/models/depth_anything_v2/depth_anything_v2_metric_{dataset}_{encoder}.pth'
            
            if not os.path.exists(load_from):
                print(f"错误：找不到深度模型权重文件 {load_from}")
                return False
                
            self.depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
            self.depth_model.load_state_dict(torch.load(load_from, map_location='cpu'))
            self.depth_model = self.depth_model.to(self.depth_device).eval()
            print("[_load_depth_model] 深度模型加载成功。")
            return True
        except Exception as e:
            print(f"[_load_depth_model] 深度模型加载失败: {e}")
            self.depth_model = None
            return False

    def _get_xyz_from_depth_model(self, image_bgr, target_yolo_box):
        """
        (内部辅助) 使用 DepthAnythingV2 和针孔相机模型计算 XYZ。
        [!! 更新 !!] 
        - 图像在送入模型前，高度 > 720p 会被按比例缩小。
        - YOLO box 坐标会相应缩放。
        - 所有后续计算 (采样, 投影, 可视化) 均在缩放后的 720p 空间中进行。
        """
        # 1. 确保模型已加载
        if self.depth_model is None:
            if not self._load_depth_model():
                print("[_get_xyz_from_depth_model] 错误: 深度模型无法加载。")
                return None
        
        # --- [新增] 2. 图像预处理: 如果高度 > 720p, 则按比例缩小 ---
        MAX_HEIGHT = 720.0
        H_orig, W_orig = image_bgr.shape[:2]
        
        scale_factor = 1.0
        processed_bgr = image_bgr
        processed_yolo_box = target_yolo_box
        
        if H_orig > MAX_HEIGHT:
            # print(f"[_get_xyz_from_depth_model] 图像高度 {H_orig}p > {int(MAX_HEIGHT)}p, 正在缩放...")
            scale_factor = MAX_HEIGHT / H_orig
            W_new = int(W_orig * scale_factor)
            H_new = int(MAX_HEIGHT) # 720
            
            # 使用 INTER_AREA (区域插值) 进行缩小，效果最好
            processed_bgr = cv2.resize(image_bgr, (W_new, H_new), interpolation=cv2.INTER_AREA)
            
            # 按比例缩放 YOLO 框坐标
            processed_yolo_box = [int(coord * scale_factor) for coord in target_yolo_box]
            # print(f"[_get_xyz_from_depth_model] 缩放后尺寸: {W_new}x{H_new} (Scale: {scale_factor:.4f})")
        # else:
        #     print(f"[_get_xyz_from_depth_model] 图像高度 {H_orig}p, 无需缩放。")
        # -----------------------------------------------------------
        
        # 3. 获取深度图 (在 'processed_bgr' 上运行)
        print("[_get_xyz_from_depth_model] 正在推断深度图...")
        start_time = time.time()
        try:
            # [!! 修改 !!] 使用 processed_bgr
            depth_map_proc = self.depth_model.infer_image(processed_bgr, input_size=518)
        except Exception as e:
            print(f"[_get_xyz_from_depth_model] 深度模型推断失败: {e}")
            return None
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[Depth Model] 执行时间: {elapsed_time:.6f} 秒")

        
        # 4. 获取 YOLO 框中心点 (在 'processed_bgr' 空间中)
        # [!! 修改 !!] 使用 processed 变量
        H_proc, W_proc = processed_bgr.shape[:2]
        x1, y1, x2, y2 = processed_yolo_box
        yolo_u = int((x1 + x2) / 2)
        yolo_v = int((y1 + y2) / 2)
        
        # 5. 从深度图中获取 Z (米)
        Z_m = 0.0
        try:
            # [!! 修改 !!] 使用 depth_map_proc
            Z_m = float(depth_map_proc[yolo_v, yolo_u])
        except IndexError:
            print(f"[_get_xyz_from_depth_model] 错误: 坐标 ({yolo_v}, {yolo_u}) 超出深度图边界 ({depth_map_proc.shape})")
            Z_m = -1.0 # 设为无效值

        # --- 6. 生成可视化深度图 (逻辑来自 main_depth_test2.py) ---
        VIS_MIN = 0.0
        # [!! 修正 !!] 确保 VIS_MAX 与 _load_depth_model 中的 max_depth (20.0) 一致
        VIS_MAX = 20.0 
        
        # [!! 修改 !!] 使用 depth_map_proc
        depth_clipped = np.clip(depth_map_proc, VIS_MIN, VIS_MAX) 
        
        if VIS_MAX - VIS_MIN > 1e-6:
            depth_normalized = (depth_clipped - VIS_MIN) / (VIS_MAX - VIS_MIN)
        else:
            depth_normalized = np.zeros_like(depth_clipped)
            
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

        # --- 7. 标注并保存图像 (在 'processed_bgr' 空间中) ---
        try:
            # [!! 无需修改 !!] x1,y1,x2,y2, yolo_u,v, Z_m 已经都是 processed 空间的值
            cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(depth_vis, (yolo_u, yolo_v), 5, (0, 0, 255), -1)
            
            text = f"Z: {Z_m:.2f}m"
            text_pos = (x1, y1 - 10 if y1 > 10 else y1 + 20)
            cv2.putText(depth_vis, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            save_path = "temp_current_frame_depth_annotated.png"
            cv2.imwrite(save_path, depth_vis)
            print(f"[_get_xyz_from_depth_model] 带标注的深度图已保存: {save_path}")

        except Exception as e:
            print(f"[_get_xyz_from_depth_model] 标注深度图时出错: {e}")

        # 8. 检查 Z 有效性
        if Z_m <= 0.1: # 最小有效距离
            print(f"[_get_xyz_from_depth_model] 深度值无效 (Z={Z_m:.2f}m)，无法计算XYZ。")
            return None
            
        # 9. 计算 X 和 Y (反向投影) - [!! 修改 !!]
        #    现在所有计算都基于 'processed_bgr' (H_proc, W_proc) 的维度
        
        MAX_VLM_SIDE = 480 # VLM 基准 (f=500)
        
        # [!! 修改 !!] 使用 H_proc, W_proc
        if W_proc >= H_proc:
            W_vlm = MAX_VLM_SIDE
            H_vlm = int(H_proc * (MAX_VLM_SIDE / W_proc))
        else:
            H_vlm = MAX_VLM_SIDE
            W_vlm = int(W_proc * (MAX_VLM_SIDE / H_proc))
        
        # [!! 修改 !!] 计算 'processed' 空间下的焦距和中心点
        f_proc_x = 500.0 * (W_proc / W_vlm)
        f_proc_y = 500.0 * (H_proc / H_vlm)
        cx_proc = W_proc / 2.0
        cy_proc = H_proc / 2.0
        
        # [!! 无需修改 !!] yolo_u, yolo_v 已经是 processed 空间的值
        X_m = (yolo_u - cx_proc) * Z_m / f_proc_x
        Y_m = (yolo_v - cy_proc) * Z_m / f_proc_y
        
        xyz_coords = {"x": X_m, "y": Y_m, "z": Z_m}
        print(f"[_get_xyz_from_depth_model] 匹配成功! 目标 XYZ : {xyz_coords}")
        
        return xyz_coords
      
    def _project_3d_box_to_2d(self, bbox_3d, f, cx, cy, img_w, img_h):
        """
        (内部辅助) 将 LLM 返回的 3D bbox 投影到 2D 图像平面，计算 2D 中心点。
        """
        x_c, y_c, z_c = bbox_3d[0], bbox_3d[1], bbox_3d[2]
        dx, dy, dz = bbox_3d[3], bbox_3d[4], bbox_3d[5]
        roll, pitch, yaw = bbox_3d[6], bbox_3d[7], bbox_3d[8]

        # 构建 3D 角点 (Local)
        corners_local = np.array([
            [dx/2, dy/2, dz/2], [dx/2, dy/2, -dz/2],
            [dx/2, -dy/2, dz/2], [dx/2, -dy/2, -dz/2],
            [-dx/2, dy/2, dz/2], [-dx/2, dy/2, -dz/2],
            [-dx/2, -dy/2, dz/2], [-dx/2, -dy/2, -dz/2],
        ])

        # 旋转矩阵
        R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R = R_z @ R_y @ R_x

        # 变换到相机坐标系
        corners_3d = (R @ corners_local.T).T + np.array([x_c, y_c, z_c])

        corners_2d = []
        for corner in corners_3d:
            if corner[2] <= 0: continue
            u = int(f * corner[0] / corner[2] + cx)
            v = int(f * corner[1] / corner[2] + cy)
            corners_2d.append((u, v))

        if not corners_2d:
            return None, None

        # 计算投影后的 2D 边界框中心
        corners_2d = np.array(corners_2d)
        min_u, min_v = np.min(corners_2d, axis=0)
        max_u, max_v = np.max(corners_2d, axis=0)
        
        # 限制在图像范围内
        min_u, min_v = max(0, min_u), max(0, min_v)
        max_u, max_v = min(img_w, max_u), min(img_h, max_v)

        center_x = (min_u + max_u) / 2
        center_y = (min_v + max_v) / 2
        
        return (center_x, center_y), (min_u, min_v, max_u, max_v)

    def _get_xyz_from_vlm_matching_yolo(self, image_bgr, target_yolo_box, use_depth_model: bool = False): # <-- 新增 use_depth_model
        """
        (内部辅助) 
        根据 use_depth_model 标志，选择 VLM (LLM) 或 DepthAnythingV2 来获取 3D 坐标。
        """
        # import openai 

        # ==========================================================
        # 分支 1: 使用 DepthAnythingV2 模型 (新逻辑)
        # ==========================================================
        if use_depth_model:
            print("[move_to_person_2] 正在使用 DepthAnythingV2 模型获取 3D 坐标...")
            return self._get_xyz_from_depth_model(image_bgr, target_yolo_box)

        # ==========================================================
        # 分支 2: 使用 VLM (LLM) (原有逻辑)
        # ==========================================================
        print("[move_to_person_2] 正在使用 VLM (LLM) 获取 3D 坐标...")

        # --- 1. 图像预处理 ---
        H_full, W_full = image_bgr.shape[:2]
        MAX_VLM_SIDE = 480
        if W_full >= H_full:
            W_vlm = MAX_VLM_SIDE
            H_vlm = int(H_full * (MAX_VLM_SIDE / W_full))
        else:
            H_vlm = MAX_VLM_SIDE
            W_vlm = int(W_full * (MAX_VLM_SIDE / H_full))
        
        img_vlm = cv2.resize(image_bgr, (W_vlm, H_vlm))
        
        # 计算 YOLO 坐标在 VLM 图像上的位置
        scale_x = W_vlm / W_full
        scale_y = H_vlm / H_full
        yolo_cx = ((target_yolo_box[0] + target_yolo_box[2]) / 2) * scale_x
        yolo_cy = ((target_yolo_box[1] + target_yolo_box[3]) / 2) * scale_y

        # --- 2. 构建请求 ---
        base64_rgb_str = _cv2_to_base64(img_vlm, ".png")
        prompt = """
        Detect all 'person' in the image.
        - If no objects are found, return an empty JSON array: []
        - If objects are found, output ONLY a valid JSON array in this format:
        [{{"bbox_3d": [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw], "label": "person"}}]
        
        Important instructions:
        1. Return ONLY the JSON array with no additional text.
        2. Use empty array [] when no objects are detected.
        3. For each object, provide 3D bounding box coordinates in meters relative to the camera.
        4. Never output explanations or other text.
        """
        
        print("[move_to_person_2] 正在调用 VLM 获取 3D 坐标...")
        try:
            start_time = time.time()
            response = self.llm_client.chat.completions.create(
                model=API_VL_MODEL,
                messages=[
                    { "role": "system", "content": "You are a spatial analysis assistant for a robot." },
                    { "role": "user", "content": [ {"type": "text", "text": prompt}, { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{base64_rgb_str}" } }, ], }
                ],
                temperature=0.0,
            )
            raw_response = response.choices[0].message.content
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[move_to_person_2] VLM执行时间: {elapsed_time:.6f} 秒")  # 保留6位小数
            print("[move_to_person_2] VLM返回: ", raw_response)
            try:
                result_list = json.loads(raw_response)
            except json.JSONDecodeError:
                start = raw_response.find("["); end = raw_response.rfind("]")
                result_list = json.loads(raw_response[start:end+1]) if start != -1 else []

        except openai.BadRequestError as e:
            err_body = e.body or {}
            if isinstance(err_body, dict) and err_body.get('code') == 'data_inspection_failed':
                print(f"⚠️ [安全拦截] 阿里云认为图片包含敏感内容 (Error 400). 跳过此帧。")
            else:
                print(f"❌ [API 请求错误] 400 Bad Request: {e}")
            return None
        except Exception as e:
            print(f"❌ [VLM 未知错误] 调用失败: {e}")
            return None

        if not result_list:
            print("[move_to_person_2] VLM 未返回任何 3D 物体。")
            return None

        # --- 3. 计算匹配逻辑 (第一轮循环：找最佳) ---
        best_match_xyz = None
        best_match_idx = -1
        min_dist = float('inf')
        
        f = 500.0
        cx, cy = W_vlm / 2, H_vlm / 2

        print(f"[move_to_person_2] VLM 返回 {len(result_list)} 个候选框，正在计算最佳匹配...")
        
        # 预计算所有框的 2D 中心，方便后续画图和匹配
        parsed_boxes = [] 

        for idx, item in enumerate(result_list):
            bbox_3d = item.get("bbox_3d")
            if not bbox_3d or len(bbox_3d) < 9: 
                parsed_boxes.append(None)
                continue
            
            center_2d, _ = self._project_3d_box_to_2d(bbox_3d, f, cx, cy, W_vlm, H_vlm)
            
            parsed_boxes.append({"bbox_3d": bbox_3d, "center_2d": center_2d})

            if center_2d:
                dist = math.sqrt((center_2d[0] - yolo_cx)**2 + (center_2d[1] - yolo_cy)**2)
                # 记录距离以便调试
                item['_debug_dist'] = dist 
                
                # 判定阈值 (图像宽度的 1/4)
                if dist < min_dist and dist < (W_vlm / 2):
                    min_dist = dist
                    best_match_idx = idx
                    best_match_xyz = { "x": bbox_3d[0], "y": bbox_3d[1], "z": bbox_3d[2] }

        # --- 4. 调试绘图 (第二轮循环：画所有框) ---
        try:
            debug_img = img_vlm.copy()
            
            # 绘制 YOLO 参考中心 (红色实心圆)
            cv2.circle(debug_img, (int(yolo_cx), int(yolo_cy)), 6, (0, 0, 255), -1)
            cv2.putText(debug_img, "YOLO Ref", (int(yolo_cx)+10, int(yolo_cy)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            for idx, data in enumerate(parsed_boxes):
                if data is None: continue
                
                bbox_3d = data["bbox_3d"]
                
                # 颜色选择: 最佳匹配 = 绿色 (0, 255, 0), 其他 = 蓝色 (255, 0, 0)
                if idx == best_match_idx:
                    color = (0, 255, 0) # Green
                    thickness = 2
                    label_prefix = "MATCH"
                else:
                    color = (255, 0, 0) # Blue
                    thickness = 1
                    label_prefix = "vlm"

                # --- 绘制 3D 线框 ---
                x_c, y_c, z_c = bbox_3d[0], bbox_3d[1], bbox_3d[2]
                dx, dy, dz = bbox_3d[3], bbox_3d[4], bbox_3d[5]
                roll, pitch, yaw = bbox_3d[6], bbox_3d[7], bbox_3d[8]

                corners_local = np.array([
                    [dx/2, dy/2, dz/2], [dx/2, dy/2, -dz/2],
                    [dx/2, -dy/2, dz/2], [dx/2, -dy/2, -dz/2],
                    [-dx/2, dy/2, dz/2], [-dx/2, dy/2, -dz/2],
                    [-dx/2, -dy/2, dz/2], [-dx/2, -dy/2, -dz/2],
                ])
                R_x = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
                R_y = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
                R_z = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
                R = R_z @ R_y @ R_x
                corners_3d = (R @ corners_local.T).T + np.array([x_c, y_c, z_c])
                
                corners_2d_pts = []
                for corner in corners_3d:
                    if corner[2] <= 0: corners_2d_pts.append(None)
                    else:
                        u = int(f * corner[0] / corner[2] + cx)
                        v = int(f * corner[1] / corner[2] + cy)
                        corners_2d_pts.append((u, v))
                
                lines = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
                valid_pts = []
                for i, j in lines:
                    if corners_2d_pts[i] is not None and corners_2d_pts[j] is not None:
                        cv2.line(debug_img, corners_2d_pts[i], corners_2d_pts[j], color, thickness)
                        valid_pts.append(corners_2d_pts[i])
                        valid_pts.append(corners_2d_pts[j])
                
                # --- 绘制文字信息 ---
                if valid_pts:
                    pts_arr = np.array(valid_pts)
                    cx_text = int(np.mean(pts_arr[:, 0]))
                    cy_text = int(np.min(pts_arr[:, 1])) - 5
                    
                    dist_val = result_list[idx].get('_debug_dist', 9999)
                    info_text = f"{label_prefix} Z={z_c:.1f}m Err={dist_val:.0f}"
                    
                    cv2.putText(debug_img, info_text, (max(0, cx_text-40), max(15, cy_text)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 保存图片 (只要 result_list 不为空，代码走到这里都会保存)
            save_path = "temp_current_frame_xyz_2.png"
            cv2.imwrite(save_path, debug_img)
            print(f"[move_to_person_2] 调试图已保存 (含 {len(result_list)} 个框): {save_path}")

        except Exception as e:
            print(f"[move_to_person_2] 调试绘图失败: {e}")

        # --- 5. 返回结果 ---
        if best_match_xyz:
            print(f"[move_to_person_2] 匹配成功! 目标 XYZ: {best_match_xyz}")
            return best_match_xyz
        else:
            print("[move_to_person_2] 匹配失败: VLM 检测到了物体，但距离 YOLO 人脸中心太远。")
            return None

    def check_view_optimized(self, use_api: bool = True, api_provider: str = "baidu", debug_image_path: str = None, use_depth_model: bool = False):
        """
        Args:
            use_api (bool): 是否使用 API。
            api_provider (str): "baidu" 或 "ali"。
            debug_image_path (str): 调试图片路径。
            use_depth_model (bool): 是否使用本地深度模型替代VLM获取XYZ。
        """
        # 1. 获取图像
        image_bgr = None
        if debug_image_path:
            if os.path.exists(debug_image_path):
                print(f"[check_view_optimized] DEBUG模式: 读取本地图片: {debug_image_path}")
                image_bgr = cv2.imread(debug_image_path)
            else:
                return None
        else:
            image_bgr = self.get_frame_vlm(res_param=-1)  #  YRJ
            # image_bgr = self.get_frame_vlm_fake(res_param=-1) 
        
        if image_bgr is None: return None
        
        # 2. YOLO 检测
        if self.yolo_model is None:
            self.yolo_model = YOLO(self.yolo_model_path).to("cuda")
        
        start_time = time.time()
        results = self.yolo_model.predict(source=image_bgr, classes=[0], conf=0.4, verbose=False)
        if not results or len(results[0].boxes) == 0:
            print("[move_to_person_2] YOLO 未检测到人脸。")
            return None
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[move_to_person_2] YOLO 检测到 {len(results[0].boxes)} 个人脸。执行时间: {elapsed_time:.6f} 秒")

        target_yolo_box = None 
        ref_path = "/home/dji/LMFly/UAV-Isaac-GR00T/uav_script/person_A.jpg"

        # 3. 验证循环
        for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            face_crop = image_bgr[y1:y2, x1:x2]
            if face_crop.size == 0: continue
            
            temp_crop_path = f"temp_face_check_{i}.png"
            cv2.imwrite(temp_crop_path, face_crop)
            is_match = False
            
            if use_api:
                if api_provider == "ali":
                    # --- 阿里 API ---
                    ali_ak = ALIBABA_CLOUD_ACCESS_KEY_ID
                    ali_sk = ALIBABA_CLOUD_ACCESS_KEY_SECRET
                    config = AliConfig(access_key_id=ali_ak, access_key_secret=ali_sk, endpoint='facebody.cn-shanghai.aliyuncs.com', region_id='cn-shanghai')
                    client = AliClient(config)
                    req = CompareFaceAdvanceRequest()
                    with open(ref_path, 'rb') as f1, open(temp_crop_path, 'rb') as f2:
                        req.image_urlaobject = f1
                        req.image_urlbobject = f2
                        resp = client.compare_face_advance(req, RuntimeOptions())
                        if hasattr(resp.body, 'data') and resp.body.data:
                            score = resp.body.data.confidence
                            if score > 60.0:
                                print(f"  -> [Ali] 人脸 {i} 匹配成功 (Score: {score:.1f})")
                                is_match = True
                            else:
                                print(f"  -> [Ali] 人脸 {i} 匹配失败 (Score: {score:.1f})")
                   

                else:
                    # --- 百度 API ---
                    try:
                        with open(ref_path, 'rb') as f: ref_b64 = base64.b64encode(f.read()).decode()
                        with open(temp_crop_path, 'rb') as f: cur_b64 = base64.b64encode(f.read()).decode()
                        
                        client = AipFace(_API_APP_ID, _API_KEY, _SECRET_KEY)
                        images = [
                            {'image': ref_b64, 'image_type': 'BASE64', 'face_type': 'LIVE', 'quality_control': 'LOW'},
                            {'image': cur_b64, 'image_type': 'BASE64', 'face_type': 'LIVE', 'quality_control': 'LOW'}
                        ]
                        res = client.match(images)
                        
                        if res.get('error_code') == 0:
                            score = res['result']['score']
                            if score > 60.0: 
                                print(f"  -> [API] 人脸 {i} 匹配成功 (Score: {score:.1f})")
                                is_match = True
                            else:
                                print(f"  -> [API] 人脸 {i} 匹配失败 (Score: {score:.1f})")
                        else:
                            print(f"  -> [API] 错误: {res.get('error_msg')}")
                    except Exception as e:
                        print(f"  -> [API] 验证异常: {e}")

            else:
                # --- 本地 Local ---
                try:
                    if self.inferencer is None:
                        print("  -> [Local] 错误: 本地 Inferencer 未初始化。")
                        continue

                    start_time = time.time()
                    feat_ref = self.inferencer.infer_from_image(ref_path)
                    feat_cur = self.inferencer.infer_from_image(temp_crop_path)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    if feat_ref is not None and feat_cur is not None:
                        sim = self.inferencer.calculate_similarity(feat_ref, feat_cur)
                        if sim > 0.3:
                            print(f"  -> [Local] 人脸 {i} 匹配成功 (Sim: {sim:.3f})    执行时间: {elapsed_time:.6f} 秒")
                            is_match = True
                        else:
                            print(f"  -> [Local] 人脸 {i} 匹配失败 (Sim: {sim:.3f})     执行时间: {elapsed_time:.6f} 秒")
                except Exception as e:
                    print(f"  -> [Local] 验证异常: {e}")

            if is_match:
                target_yolo_box = [x1, y1, x2, y2]
                break 

        if target_yolo_box is not None:
            # [!!] 将 use_depth_model 标志传递下去
            return self._get_xyz_from_vlm_matching_yolo(image_bgr, target_yolo_box, use_depth_model=use_depth_model)
        
        return None
    
    def move_to_person_2(self, target_distance_cm: int, target_height_cm: int = 0, use_api: bool = True, use_depth_model: bool = True) -> bool:
        """
        [move_to_person_2]
        [优化版] 寻找并移动到 Person_A。
        此版本强制使用 'DepthAnythingV2' (深度模型) 来获取 XYZ 坐标，而不是 VLM。

        Args:
            target_distance_cm: 目标距离
            target_height_cm: 目标高度
            use_api (bool): 是否使用在线 API 进行人脸识别 (默认为 True)
        """
        print(f"开始执行 move_to_person_2 (DepthAnythingV2 版): 目标 'person_A', 距离 {target_distance_cm}cm, API模式={'开' if use_api else '关'}")
        
        found_target_xyz = None
        
        # 原地旋转 6 次 (每次 60 度)
        for i in range(6):
            print(f"\n[move_to_person_2] 扫描角度 {i+1}/6 ...")
            
            # [!!] 关键区别: 强制 use_depth_model=True
            result_3d = self.check_view_optimized(use_api=use_api, use_depth_model=use_depth_model)
            
            if result_3d:
                found_target_xyz = result_3d
                break
            
            print("[move_to_person_2] 当前视角未找到目标，逆时针旋转 60 度...")
            self.turn_counter_clockwise(60)
            time.sleep(1) 

        if not found_target_xyz:
            print("[move_to_person_2] 扫描结束，未找到 person_A。")
            self.talk("未找到目标人物")
            return False

        # --- 移动逻辑 (与 move_to_person_2 相同) ---
        x_m = found_target_xyz["x"]
        y_m = found_target_xyz["y"]
        z_m = found_target_xyz["z"]
        
        if z_m <= 0:
            print(f"深度模型返回无效 Z 距离 ({z_m})。")
            return False
            
        actions_taken = []
        
        # 1. 旋转 (X轴)
        if abs(x_m) > 0.05:
            angle_rad = math.atan(x_m / z_m)
            angle_deg = int(math.degrees(angle_rad))
            if angle_deg > 0:
                self.turn_clockwise(angle_deg); actions_taken.append(f"右转{angle_deg}")
            elif angle_deg < 0:
                self.turn_counter_clockwise(abs(angle_deg)); actions_taken.append(f"左转{abs(angle_deg)}")

        # 2. 高度 (Y轴)
        vertical_error_m = y_m - (target_height_cm / 100.0)
        move_z_cm = int(vertical_error_m * 100)
        if abs(move_z_cm) > 5:
            if move_z_cm > 0:
                self.move_down(move_z_cm); actions_taken.append(f"下降{move_z_cm}")
            else:
                self.move_up(abs(move_z_cm)); actions_taken.append(f"上升{abs(move_z_cm)}")

        # 3. 距离 (Z轴)
        move_distance_cm = int(z_m * 100) - target_distance_cm
        if abs(move_distance_cm) > 20:
            if move_distance_cm > 0:
                self.move_forward(min(move_distance_cm, 5000)); actions_taken.append(f"前进{move_distance_cm}")
            else:
                self.move_backward(min(abs(move_distance_cm), 5000)); actions_taken.append(f"后退{abs(move_distance_cm)}")

        if actions_taken:
            print("完成移动: " + ", ".join(actions_taken))
            self.talk("已移动到目标人物面前")
            return True
        else:
            print("已经在目标位置。")
            return True

        


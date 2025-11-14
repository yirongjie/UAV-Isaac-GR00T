import os
import threading
import time

import cv2
import dotenv
import yaml
from PIL import Image

import re
import json
import inspect
import math
from openai import OpenAI

# --- 本地模型导入 ---
import torch
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("警告: 'transformers' 库未安装。本地 LLM 模式将不可用。")
    print("请运行: pip install transformers torch")
    AutoModelForCausalLM, AutoTokenizer = None, None
# --- [新增结束] ---


# USE_VLM_AGENT = False
# --- 动态导入无人机 Wrapper ---
TelloDrone, MockDrone, AirSimDrone, FanciSwarmDrone, DjiM4DDrone, MockDjiM4DDrone = None, None, None, None, None, None

dotenv.load_dotenv()

# --- 全局配置 ---
API_KEY = os.environ.get("OPENROUTER_API_KEY") # API 模式下仍然需要
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen/qwen3-vl-235b-a22b-instruct")
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))
AGENT_MODE = os.environ.get("AGENT_MODE", "tool")
USE_CUSTOM_PROMPT = True
PROMPT_PATH = os.environ.get("PROMPT_PATH", "toolcalling_agent_cn.yaml")

DRONE_TYPE = os.environ.get("DRONE_TYPE", "djim4d").lower()  

# --- [修改] LLM 路径配置 ---
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "") # 默认值为空字符串
# --- [修改结束] ---

# ----------------------------------------------------

# --- [新增] 辅助函数：模拟 GPS 坐标计算 ---
# 注意：在实际环境中，此函数应该引用 DJI M4D wrapper 中的具体实现。
# 这里提供一个简单的地球近似模型计算（仅用于保持代码完整性）。
def calculate_new_gps(lat, lon, dx_m, dy_m):
    """
    根据给定的米制位移 (dx: 东向, dy: 北向)，计算新的 GPS 坐标。
    (WGS-84 球体近似)
    """
    R_earth = 6378137.0 # 地球平均半径 (米)
    
    # 纬度变化：1米对应的纬度变化
    dlat = dy_m / R_earth * (180.0 / math.pi)
    
    # 经度变化：1米对应的经度变化 (取决于当前纬度)
    dlon = dx_m / (R_earth * math.cos(math.pi * lat / 180.0)) * (180.0 / math.pi)
    
    new_lat = lat + dlat
    new_lon = lon + dlon
    return new_lat, new_lon
# --- [新增结束] ---


class SimpleLiteLLMModel:
    # ... (SimpleLiteLLMModel 类的定义保持不变)
    def __init__(self, model_id, api_key, base_url, temperature):
        self.model_id = model_id
        self.temperature = temperature
        if not api_key:
             print("警告: [SimpleLiteLLMModel] 未提供 API_KEY。")
             raise ValueError("API 模式需要 OPENROUTER_API_KEY。")
             
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300.0
        )

    def chat(self, messages: list) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
            )
            response_content = completion.choices[0].message.content
            return response_content
        except Exception as e:
            print(f"[SimpleLiteLLMModel] 调用 LLM 出错: {e}")
            return f"错误: 调用模型失败 - {e}"

# --- 本地 Transformers 模型封装器 ---
class TransformersLocalModel:
    # ... (TransformersLocalModel 类的定义保持不变)
    def __init__(self, model_path: str):
        if AutoModelForCausalLM is None:
            raise ImportError("Transformers 库未成功导入。无法使用本地模式。")
            
        print(f"[TransformersLocalModel] 正在从 {model_path} 加载模型和 Tokenizer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TransformersLocalModel] 使用设备: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                            use_fast=False,
                                                            trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map=self.device
                )
            self.model.eval() # 设置为评估模式
            print("[TransformersLocalModel] 模型加载完成。")
        except Exception as e:
            print(f"[TransformersLocalModel] 加载本地模型失败: {e}")
            print(f"请确保路径 '{model_path}' 是一个有效的 Hugging Face 模型目录。")
            raise

    def chat(self, messages: list) -> str:
        try:
            prompt_str = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
            input_token_len = inputs.input_ids.shape[1]
            
            with torch.no_grad(): # 推理时不需要梯度
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1536, 
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=TEMPERATURE if TEMPERATURE > 0 else 1.0, 
                    top_p=0.9,
                    do_sample=True if TEMPERATURE > 0 else False,
                )
            
            new_tokens = outputs[0][input_token_len:]
            response_content = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return response_content.strip()

        except Exception as e:
            print(f"[TransformersLocalModel] 调用本地 LLM 出错: {e}")
            import traceback
            traceback.print_exc() 
            return f"错误: 调用本地模型失败 - {e}"


class SimpleToolCallingAgent:
    # ... (SimpleToolCallingAgent 类的定义保持不变)
    def __init__(self, tools: list, drone_instance, model, system_prompt_template: str, max_steps: int = 100):
        self.drone_instance = drone_instance
        self.model = model 
        self.max_steps = max_steps

        self.tools_list = tools 
        
        self.system_prompt = self._build_system_prompt(system_prompt_template, tools)
        self.messages_history = []
        
        self.action_regex = re.compile(r"Action:[\s\S]*?(\{[\s\S]*?\})(?=Action:|final_answer:|$)", re.DOTALL | re.IGNORECASE)
        self.final_answer_regex = re.compile(r"final_answer:\s*(.*)", re.DOTALL | re.IGNORECASE)

    def _build_system_prompt(self, template: str, tools: list) -> str:
        tool_definitions = []
        for tool in tools:
            inputs_str_parts = []
            for name, details in tool['inputs'].items():
                inputs_str_parts.append(f"'{name}' ({details.get('type', 'any')}) - {details.get('description', 'No description')}")
            inputs_str = "{ " + ", ".join(inputs_str_parts) + " }" if inputs_str_parts else "{}"
            
            tool_def = f"  - {tool['name']}: {tool['description']}\n     输入参数: {inputs_str}\n     返回类型: {tool['output_type']}"
            tool_definitions.append(tool_def)
            
        tools_block = "\n".join(tool_definitions)
        
        prompt = re.sub(
            r"{%-.*?for tool in tools.values\(\).*?%}(.*?){%-.*?endfor.*?%}", 
            tools_block, 
            template, 
            flags=re.DOTALL | re.IGNORECASE
        )
        return prompt

    def run(self, prompt: str, images=None):
        if images:
            print("警告: SimpleToolCallingAgent (非VLM模式) 不支持图像输入，图像将被忽略。")

        self.messages_history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        print("--- 系统提示 (已生成) ---")
        print(self.system_prompt)
        print("-----------------")
        print(f"\n--- 开始执行任务: {prompt} ---")

        final_answer_match = None 

        for step in range(self.max_steps):
            print(f"\n--- 步骤 {step + 1}/{self.max_steps} ---")

            # 1. 思考 (调用 LLM)
            llm_response_text = self.model.chat(self.messages_history)
            
            self.messages_history.append({"role": "assistant", "content": llm_response_text})
            print(f"\n[思考/行动]\n{llm_response_text}")

            # 2. 解析所有行动
            final_answer_match = self.final_answer_regex.search(llm_response_text)
            actions_to_execute = []
            
            action_starts = [m.start() for m in re.finditer(r"Action:", llm_response_text, re.IGNORECASE | re.DOTALL)]
            
            for i, start_index in enumerate(action_starts):
                if i + 1 < len(action_starts):
                    segment_end_index = action_starts[i+1]
                else:
                    segment_end_index = len(llm_response_text)
                
                json_str_segment = llm_response_text[start_index : segment_end_index]
                json_to_load = ""
                try:
                    json_start = json_str_segment.find('{')
                    if json_start == -1:
                        raise ValueError("No starting '{' found.")

                    raw_content = json_str_segment[json_start:].lstrip('\ufeff')
                    raw_content_super_clean = re.sub(r'[\s\u00A0\x00-\x1F\x7F]', ' ', raw_content) 
                    json_str_super_clean = ' '.join(raw_content_super_clean.split())
                    
                    brace_count = 0
                    json_end = -1
                    in_string = False
                    
                    for k in range(len(json_str_super_clean)):
                        char = json_str_super_clean[k]
                        
                        if char == '"' and (k == 0 or json_str_super_clean[k-1] != '\\'):
                            in_string = not in_string
                        
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = k
                                    break
                    
                    if json_end == -1:
                        raise ValueError("Closing '}' was not found/balanced.")
                    
                    json_to_load = json_str_super_clean[0 : json_end + 1]
                    action_json = json.loads(json_to_load)
                    actions_to_execute.append(action_json)

                except Exception as e:
                    print(f"\n[观察] 错误：解析 Action JSON 失败: {e}. 尝试解析的 JSON 字符串: {json_to_load}. 跳过此 Action。")
                    continue
            
            # 3. 任务结束判断逻辑
            if not actions_to_execute:
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
                    print(f"\n--- 任务完成 (检测到 final_answer，且没有 Action 需要执行) ---")
                    print(f"最终答案: {final_answer}")
                    break
                else:
                    print("\n[观察] 错误：模型未提供有效的 Action 或 final_answer。正在请求重试...")
                    self.messages_history.append({"role": "user", "content": "你没有提供任何有效的 Action 或 final_answer。请思考并提供一个工具调用或任务总结。"})
                    continue

            # 4. 顺序执行所有工具
            all_observations = []
            execution_successful = True
            for action_json in actions_to_execute:
                tool_name = action_json.get("name")
                tool_args = action_json.get("arguments", {})
                
                if not tool_name:
                    all_observations.append("Action failed: Missing 'name' field.")
                    execution_successful = False
                    break 

                try:
                    tool_to_call = self.drone_instance.get_tool_by_name(tool_name)
                    
                    sig = inspect.signature(tool_to_call)
                    typed_args = {}
                    for param_name, param in sig.parameters.items():
                        if param_name in tool_args:
                            arg_val = tool_args[param_name]
                            if param.annotation == int:
                                typed_args[param_name] = int(float(arg_val)) 
                            elif param.annotation == float:
                                typed_args[param_name] = float(arg_val)
                            else:
                                typed_args[param_name] = arg_val 
                    
                    print(f"调用工具: {tool_name}({typed_args})")
                    observation = tool_to_call(**typed_args)
                    
                    obs_text = f"Action '{tool_name}' executed. Result: {str(observation)}"
                    all_observations.append(obs_text)

                except Exception as e:
                    error_msg = f"错误：执行工具 '{tool_name}' 失败: {e}"
                    print(f"\n[观察] {error_msg}. 停止本步骤后续操作。")
                    all_observations.append(f"Action '{tool_name}' failed. Error: {e}")
                    execution_successful = False
                    import traceback; traceback.print_exc()
                    break 

            # 5. 合并观察结果并结束本步骤
            combined_observation_str = "Observation: \n" + "\n".join(all_observations)
            print(f"\n[观察]\n{combined_observation_str}")
            
            self.messages_history.append({"role": "user", "content": combined_observation_str})

            # 6. 检查 final_answer
            if final_answer_match and execution_successful:
                 final_answer = final_answer_match.group(1).strip()
                 print(f"\n--- 任务完成 (检测到 final_answer，且所有 Actions 成功) ---")
                 print(f"最终答案: {final_answer}")
                 break

        if step == self.max_steps - 1:
            print(f"\n--- 已达到最大步骤 ({self.max_steps})，任务终止 ---")
            
        if final_answer_match:
             if 'final_answer:' not in self.messages_history[-1]['content']:
                final_answer = final_answer_match.group(1).strip()
                print(f"最终答案: {final_answer}")

                
# --- Agent 配置 ---
AGENT_CLASS_MAP = {
    "tool": SimpleToolCallingAgent,
}
assert AGENT_MODE in AGENT_CLASS_MAP, (
    f"AGENT_MODE 必须是 {list(AGENT_CLASS_MAP.keys())} 中的一个"
)
AGENT_CLS = AGENT_CLASS_MAP[AGENT_MODE]

# --- 线程协调事件 ---
stop_event = threading.Event()

# --- 无人机初始化 (保持不变) ---
drone = None
if DRONE_TYPE == "airsim":
    print("正在初始化 AirSim 无人机...")
    from airsim_smol_wrapper import AirSimDrone 
    drone = AirSimDrone()
elif DRONE_TYPE == "djim4d": # 
    print("正在初始化 DjiM4DDrone 实体无人机...")
    from DJI_M4D_smol_wrapper import DjiM4DDrone 
    drone = DjiM4DDrone()
elif DRONE_TYPE == "mock":
    print("正在初始化 模拟 M4D 无人机 (MockDjiM4DDrone)...")
    from DJI_M4D_smol_wrapper import MockDjiM4DDrone 
    drone = MockDjiM4DDrone()
elif DRONE_TYPE == "tello":
    print("正在初始化 真实 Tello 无人机...")
    from tello_smol_wrapper import Drone as TelloDrone 
    drone = TelloDrone()
else:
    raise ValueError(
        f"不支持的 DRONE_TYPE: '{DRONE_TYPE}'. 请使用 'airsim', 'djim4d', 'tello', 或 'mock'."
    )


def keep_tello_alive():
    # ... (keep_tello_alive 函数保持不变)
    """定期发送命令防止 Tello 自动降落。 (仅 Tello 需要)"""
    while not stop_event.is_set():
        if DRONE_TYPE != "tello":
            break 
        try:
            drone.turn_clockwise(1)
        except Exception as e:
            print(f"Keepalive 错误: {e}")
        time.sleep(8)


def drone_live_feed():
    # ... (drone_live_feed 函数保持不变)
    """
    (此函数保持不变)
    """
    import datetime
    import os
    import cv2  
    import time 

    print("正在启动视频流...")
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(log_dir, f"{ts}_DroneLiveFeed.mp4")
    video_writer = None

    print("正在获取第一帧以确定视频尺寸...")
    first_frame = None
    for _ in range(10): 
        frame_check = drone.get_frame() 
        if frame_check is not None and frame_check.size > 0 and len(frame_check.shape) == 3:
            first_frame = frame_check
            break
        print("等待第一帧...")
        time.sleep(0.5)
    
    if first_frame is None:
        print("错误：无法获取视频流的第一帧，视频流线程退出。")
        return

    EXPECTED_H, EXPECTED_W = first_frame.shape[:2] 
    print(f"视频流已确定。尺寸: {EXPECTED_W}x{EXPECTED_H}")
    
    cv2.namedWindow("Drone Live Feed", cv2.WINDOW_NORMAL)

    try:
        while not stop_event.is_set():
            frame_rgb = drone.get_frame() 
            if frame_rgb is None or frame_rgb.size == 0:
                print("未能获取到有效的视频帧，跳过...")
                time.sleep(0.1)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            h, w = frame_bgr.shape[:2]
            if w != EXPECTED_W or h != EXPECTED_H:
                print(f"警告: 帧尺寸变化! 从 {EXPECTED_W}x{EXPECTED_H} 变为 {w}x{h}. 尝试缩放...")
                frame_bgr = cv2.resize(frame_bgr, (EXPECTED_W, EXPECTED_H))

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, 30, (EXPECTED_W, EXPECTED_H))
            
            video_writer.write(frame_bgr)
            
            cv2.imshow("Drone Live Feed", frame_bgr)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                print("用户通过 'q' 键停止了视频流")
                stop_event.set()
                break

    except Exception as e:
        print(f"视频流发生错误: {e}")
    finally:
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            print(f"视频已保存到: {video_path}")
        print("视频流线程已终止。")


HIS_IMAGE_TIMES = 1


# --- [关键修改] 采用用户提供的 run_waypoint_mission 逻辑 ---
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
        
        print("  -> 起飞指令已接受。等待 2 秒让飞机升空并稳定...")
        time.sleep(2) 

        # 2. 获取当前 GPS 并规划 3 个“非正交”航点
        print("\n[航点] [步骤 2/4] 正在获取当前 GPS 位置以规划航线...")
        
        # 检查 origin_lat/lon/alt 是否存在（通常在 DJI M4D Wrapper 中定义）
        base_lat = getattr(drone, 'origin_lat', None)
        base_lon = getattr(drone, 'origin_lon', None)
        base_alt = getattr(drone, 'origin_alt_m', None)
        
        if not base_lat or not base_lon:
             # 如果没有起点GPS，尝试实时获取
             pose = drone._get_realtime_pose()
             if not pose or pose.get('lat') == 0.0:
                 raise Exception("无法获取有效的 GPS 起始点。")
             base_lat = pose['lat']
             base_lon = pose['lon']
             base_alt = pose.get('alt_m', 50.0)
        
        # 确保 base_alt 有一个默认值
        if base_alt is None:
            base_alt = 50.0

        print(f"  -> 航线基准点: Lat={base_lat:.6f}, Lon={base_lon:.6f}, Alt={base_alt:.1f}m")

        # --- 规划 3 个绝对航点 (采用用户提供的示例逻辑) ---
        # 东面，背面
        lat_1, lon_1 = calculate_new_gps(base_lat, base_lon, 0.0, -10.0)
        lat_2, lon_2 = calculate_new_gps(lat_1, lon_1, -0.0, 10.0)
        # lat_3, lon_3 = calculate_new_gps(lat_2, lon_2, 0.0, 10.0) # 假设这些辅助函数和坐标存在
        # lat_3, lon_3 = calculate_new_gps(lat_2, lon_2, 10.0, -10.0)

        mission_points_lla = [
            [lat_1, lon_1, base_alt], # [纬度, 经度, 高度]
            [lat_2, lon_2, base_alt],
            # [lat_3, lon_3, base_alt]
        ]
        
        print(f"  -> 航点 1 (E): Lat={lat_1:.6f}, Lon={lon_1:.6f}")
        print(f"  -> 航点 2 (SE): Lat={lat_2:.6f}, Lon={lon_2:.6f}")
        # print(f"  -> 航点 3 (W) : Lat={lat_3:.6f}, Lon={lon_3:.6f}")

        # 4. 执行动态 KMZ 航线
        print("\n[航点] [步骤 3/4] 正在发送 'fly_dynamic_kmz_mission_gps' 指令...")
        
        success = drone.fly_dynamic_kmz_mission_gps(
            absolute_points_lla=mission_points_lla,
            use_current_pos_as_start=True # 自动将当前位置作为航点 0
        )
        
        if not success:
            raise Exception("发送 KMZ 任务失败，请检查 C++ Server 日志。")
        
        print("  -> 动态航线任务已成功发送。")
        
        # 5. 等待任务执行
        print("\n[航点] [步骤 4/4] --- 任务执行中 ---")
        print("--- 飞机将飞行 [起点 -> 1 -> ...] ---")
        print("--- 飞机现在应该在最后一个航点 [原地悬停] ---")
        
        time.sleep(5) 
        
        print(f"--- [阶段 1: 航点任务完成] ---")
        print(f"--- 飞机正在悬停，准备进入VLA模式 ---")
        return True

    except Exception as e:
        print(f"\n[航点错误] 航点任务阶段发生错误: {e}")
        print("无法继续 VLA 任务。将尝试降落...")
        if drone:
             try:
                 drone.land()
             except Exception as e_land:
                 print(f"航点阶段失败后尝试降落也失败: {e_land}")
        return False
# --- [关键修改结束] ---


def agent_main():
    """
    配置并运行 Agent 的主函数。
    修改为循环等待键盘输入作为任务指令。
    """
    try:
        # 1. 初始化模型和 Agent 所需的通用组件
        
        # 1.1. 加载 YAML 提示模板内容
        # ... (加载提示模板逻辑保持不变)
        prompt_template_content = ""
        print(PROMPT_PATH)
        if USE_CUSTOM_PROMPT and PROMPT_PATH and os.path.exists(PROMPT_PATH):
            with open(PROMPT_PATH, "r", encoding="utf-8") as f:
                prompt_data = yaml.safe_load(f)
                prompt_template_content = prompt_data.get('system_prompt', '')
        else:
            print(f"警告: 未找到或未指定有效的 prompt 文件, 将使用默认模板。")
            prompt_template_content = """
            你是一个无人机助手。使用工具完成任务。
            可用工具列表:
            {%- for tool in tools.values() %}
            - {{ tool.name }}: {{ tool.description }}
               输入参数: {{tool.inputs}}
               返回类型: {{tool.output_type}}
            {%- endfor %}
            
            按以下格式回应:
            思考: ...
            Action:
            {
              "name": "工具名称",
              "arguments": { "参数": "值" }
            }
            """

        tools = drone.get_tools()

        # 1.2. 初始化模型
        # ... (模型初始化逻辑保持不变)
        model = None
        if LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH):
            print(f"--- 检测到本地模型路径，使用本地 LLM 模式 ---")
            try:
                model = TransformersLocalModel(model_path=LOCAL_MODEL_PATH)
            except Exception as e:
                print(f"!! 严重错误: 加载本地模型失败: {e}")
                print("!! 将回退到 API 模式 (如果 API_KEY 可用)...")
                model = None 
        
        elif LOCAL_MODEL_PATH:
            print(f"警告: LOCAL_MODEL_PATH ('{LOCAL_MODEL_PATH}') 已设置，但不是一个有效的目录。")
            print("将回退到 API 模式 (如果 API_KEY 可用)...")

        if model is None:
            print(f"--- 使用 API (OpenRouter) LLM 模式 ---")
            if not API_KEY:
                print("错误: 未设置 LOCAL_MODEL_PATH，也未在 .env 中找到 OPENROUTER_API_KEY。")
                return 

            base_url = os.environ.get("BASE_URL", "https://openrouter.ai/api/v1")
            if "]" in base_url or "[" in base_url:
                 base_url = "https://openrouter.ai/api/v1"
                 
            is_local_server = "0.0.0.0" in base_url or "localhost" in base_url

            effective_api_key = API_KEY 
            model_id=MODEL_NAME
            if is_local_server:
                print(f"--- 检测到本地 llama-server 模式 (Base URL: {base_url}) ---")
                effective_api_key = "sk-fake-key"
                model_id="gpt-3.5-turbo"

            else:
                print(f"--- 使用 API (OpenRouter) LLM 模式 ---")
                if not effective_api_key: 
                    print("错误: 未设置 OPENROUTER_API_KEY。")
                    return 
        
            model = SimpleLiteLLMModel(
                model_id=model_id,
                api_key=effective_api_key,
                base_url=base_url,
                temperature=TEMPERATURE,
            )
        
        # 1.3. 初始化 Agent 实例
        agent = AGENT_CLS(
            tools=tools,
            drone_instance=drone,  
            model=model,
            max_steps=100,
            system_prompt_template=prompt_template_content, 
        )
        print("\n--- Agent 初始化完成，等待任务指令 ---")

        # 2. 循环等待用户输入并执行任务
        while not stop_event.is_set():
            # 2.1. 等待用户输入任务指令
            prompt = input("\n请为无人机输入一个任务指令 (输入 'quit' 或 'exit' 退出): \n> ")

            # 2.2. 检查退出指令
            if prompt.lower() in ['quit', 'exit']:
                print("接收到退出指令。")
                stop_event.set()
                break

            if not prompt.strip():
                 print("任务指令为空，请重新输入。")
                 continue
                 
            # 2.3. 执行 Agent 任务
            print(f"\n开始执行 Agent 任务: **{prompt}**")
            agent.run(prompt) 

            print("\n--- 任务执行完毕，等待下一个指令 ---")

    except Exception as e:
        print(f"Agent 任务执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not stop_event.is_set():
             stop_event.set()
        print("Agent 主循环已终止。")


if __name__ == "__main__":
    try:
        # --- [关键调用] 航点任务预执行 ---
        print("--- 任务开始：首先执行预设航点任务 ---")
        waypoint_success = run_waypoint_mission(drone) # 不传入 waypoints 参数
        
        if not waypoint_success:
            print("预设航点任务执行失败。程序终止。")
        else:
            print("预设航点任务执行成功。继续执行 Agent 任务。")
            
            # --- Tello 模式处理 (保留原逻辑) ---
            if DRONE_TYPE == "tello":
                print("Tello 模式已激活，正在启动 Keepalive 线程...")
                keepalive_thread = threading.Thread(target=keep_tello_alive, daemon=True)
                keepalive_thread.start()

                agent_main_thread = threading.Thread(target=agent_main, daemon=True)
                agent_main_thread.start()

                # 在主线程中运行视频流，以便处理 GUI 事件
                drone_live_feed()

                # 等待 agent 线程结束
                agent_main_thread.join(timeout=5)
            else:
                # 直接在主线程中运行 Agent (交互式循环)
                agent_main()
                
    except KeyboardInterrupt:
        print("\n检测到用户中断 (Ctrl+C)，正在关闭程序...")
        stop_event.set()
    finally:
        # 增加安全的无人机关闭流程
        if drone and hasattr(drone, "shutdown"):
            print("正在执行无人机特定的关闭程序...")
            drone.shutdown()

        print("程序已退出。")
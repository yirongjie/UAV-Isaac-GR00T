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
# 如果设置了此路径，将优先使用本地模型。否则使用 API。
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "") # 默认值为空字符串
# --- [修改结束] ---

# ----------------------------------------------------

class SimpleLiteLLMModel:
    """
    一个精简的模型封装器，使用 OpenAI 客户端（兼容 OpenRouter）
    """
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
        """
        调用 LLM 的 chat completion 接口
        """
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
    """
    一个封装器，用于本地加载 Transformers 模型，使其兼容 Agent。
    """
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
        """
        调用本地 LLM 生成响应
        """
        try:
            # 1. 应用聊天模板 (将 OpenAI 格式转换为模型需要的格式)
            #    add_generation_prompt=True 会在末尾添加助手角色的起始标记
            prompt_str = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 2. Tokenize
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
            
            # 3. Generate
            #    我们需要记录输入 token 的长度，以便只解码新生成的部分
            input_token_len = inputs.input_ids.shape[1]
            
            with torch.no_grad(): # 推理时不需要梯度
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1536, # 允许 Agent 输出较长的思考过程
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=TEMPERATURE if TEMPERATURE > 0 else 1.0, # 0.0 可能导致卡住，设为 1.0
                    top_p=0.9,
                    do_sample=True if TEMPERATURE > 0 else False,
                )
            
            # 4. Decode
            #    只解码新生成的部分 (outputs[0] 是 [batch_size=0])
            new_tokens = outputs[0][input_token_len:]
            response_content = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return response_content.strip()

        except Exception as e:
            print(f"[TransformersLocalModel] 调用本地 LLM 出错: {e}")
            import traceback
            traceback.print_exc() # 打印详细堆栈
            return f"错误: 调用本地模型失败 - {e}"
# --- [新增结束] ---


class SimpleToolCallingAgent:
    """
    一个精简的工具调用代理，用于替换 smolagents.ToolCallingAgent
    """
    def __init__(self, tools: list, drone_instance, model, system_prompt_template: str, max_steps: int = 100):
        # model 现在可以是 SimpleLiteLLMModel 或 TransformersLocalModel
        self.drone_instance = drone_instance
        self.model = model 
        self.max_steps = max_steps

        self.tools_list = tools 
        
        self.system_prompt = self._build_system_prompt(system_prompt_template, tools)
        self.messages_history = []
        
        self.action_regex = re.compile(r"Action:[\s\S]*?(\{[\s\S]*?\})(?=Action:|final_answer:|$)", re.DOTALL | re.IGNORECASE)
        self.final_answer_regex = re.compile(r"final_answer:\s*(.*)", re.DOTALL | re.IGNORECASE)

    def _build_system_prompt(self, template: str, tools: list) -> str:
        """
        将工具定义注入到 YAML 模板中
        """
        tool_definitions = []
        for tool in tools:
            inputs_str_parts = []
            for name, details in tool['inputs'].items():
                inputs_str_parts.append(f"'{name}' ({details.get('type', 'any')}) - {details.get('description', 'No description')}")
            inputs_str = "{ " + ", ".join(inputs_str_parts) + " }" if inputs_str_parts else "{}"
            
            tool_def = f"  - {tool['name']}: {tool['description']}\n     输入参数: {inputs_str}\n     返回类型: {tool['output_type']}"
            tool_definitions.append(tool_def)
            
        tools_block = "\n".join(tool_definitions)
        
        # 使用一个更健壮的正则表达式来匹配整个 for 循环块
        # 它会匹配从 {%- for ... %} 到 {%- endfor %} 的所有内容，并将其替换为 tools_block
        prompt = re.sub(
            r"{%-.*?for tool in tools.values\(\).*?%}(.*?){%-.*?endfor.*?%}", 
            tools_block, 
            template, 
            flags=re.DOTALL | re.IGNORECASE
        )
        
        # 同样替换 YAML 模板中的 `{{task}}` 占位符
        # (虽然在 system_prompt 中没有，但在 planning 部分有，这样做更安全)
        # prompt = prompt.replace("{{task}}", "") 
        
        return prompt

    def run(self, prompt: str, images=None):
        """
        执行代理的 "思考->行动->观察" 循环。
        【终极修复版本】：手动分割 Action 片段 + 极度激进的清洗 + 大括号计数。
        """
        if images:
            print("警告: SimpleToolCallingAgent (非VLM模式) 不支持图像输入，图像将被忽略。")

        # 1. 初始化 (保持不变)
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
            
            # 记录模型的完整思考过程
            self.messages_history.append({"role": "assistant", "content": llm_response_text})
            print(f"\n[思考/行动]\n{llm_response_text}")

            # 2. 解析所有行动 (Actions) - 放弃正则 finditer，使用手动分割
            final_answer_match = self.final_answer_regex.search(llm_response_text)
            actions_to_execute = []
            
            # 查找所有 Action: 的起始位置
            action_starts = [m.start() for m in re.finditer(r"Action:", llm_response_text, re.IGNORECASE | re.DOTALL)]
            
            for i, start_index in enumerate(action_starts):
                
                # 确定当前 Action 片段的结束点 (下一个 Action 的起始，或文本末尾)
                if i + 1 < len(action_starts):
                    segment_end_index = action_starts[i+1]
                else:
                    segment_end_index = len(llm_response_text)
                
                # 当前 Action 的原始文本片段
                json_str_segment = llm_response_text[start_index : segment_end_index]
                
                json_to_load = ""
                try:
                    # 1. 找到起始 '{'
                    json_start = json_str_segment.find('{')
                    if json_start == -1:
                        raise ValueError("No starting '{' found.")

                    # 2. 预清理：移除 BOM 标记，并扁平化 JSON 结构
                    raw_content = json_str_segment[json_start:].lstrip('\ufeff')
                    
                    # 3. 替换所有空白符、非标准字符、控制字符为标准空格
                    raw_content_super_clean = re.sub(r'[\s\u00A0\x00-\x1F\x7F]', ' ', raw_content) 
                    json_str_super_clean = ' '.join(raw_content_super_clean.split())
                    
                    # 4. 大括号计数逻辑 (找到平衡的闭合 '}')
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
                    
                    # 5. 最终切片和加载
                    json_to_load = json_str_super_clean[0 : json_end + 1]
                    action_json = json.loads(json_to_load)
                    actions_to_execute.append(action_json)

                except Exception as e:
                    # 在最终版本中，应该重新启用 try/except
                    print(f"\n[观察] 错误：解析 Action JSON 失败: {e}. 尝试解析的 JSON 字符串: {json_to_load}. 跳过此 Action。")
                    continue
            
            # 3. 任务结束判断逻辑 (不变)
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

            # 4. 顺序执行所有工具 (保持不变)
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
                    
                    # 转换参数类型（LLM 总是返回字符串或数字）
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
                    
                    # 记录每个动作的观察结果
                    obs_text = f"Action '{tool_name}' executed. Result: {str(observation)}"
                    all_observations.append(obs_text)

                except Exception as e:
                    # 如果任何 Action 失败，立即记录错误并设置标志位
                    error_msg = f"错误：执行工具 '{tool_name}' 失败: {e}"
                    print(f"\n[观察] {error_msg}. 停止本步骤后续操作。")
                    all_observations.append(f"Action '{tool_name}' failed. Error: {e}")
                    execution_successful = False
                    import traceback; traceback.print_exc()
                    break 

            # 5. 合并观察结果并结束本步骤 (保持不变)
            combined_observation_str = "Observation: \n" + "\n".join(all_observations)
            print(f"\n[观察]\n{combined_observation_str}")
            
            # 将合并的观察结果作为用户消息传回给 LLM
            self.messages_history.append({"role": "user", "content": combined_observation_str})

            # 6. 检查 final_answer (保持不变)
            if final_answer_match and execution_successful:
                 final_answer = final_answer_match.group(1).strip()
                 print(f"\n--- 任务完成 (检测到 final_answer，且所有 Actions 成功) ---")
                 print(f"最终答案: {final_answer}")
                 break

        if step == self.max_steps - 1:
            print(f"\n--- 已达到最大步骤 ({self.max_steps})，任务终止 ---")
            
        # 确保在退出前捕获最后的 final_answer
        if final_answer_match:
             if 'final_answer:' not in self.messages_history[-1]['content']:
                final_answer = final_answer_match.group(1).strip()
                print(f"最终答案: {final_answer}")

                
# --- Agent 配置 ---
# (我们只实现了 ToolCallingAgent)
AGENT_CLASS_MAP = {
    # "code": CodeAgent, # 未实现
    "tool": SimpleToolCallingAgent,
}
assert AGENT_MODE in AGENT_CLASS_MAP, (
    f"AGENT_MODE 必须是 {list(AGENT_CLASS_MAP.keys())} 中的一个"
)
AGENT_CLS = AGENT_CLASS_MAP[AGENT_MODE]

# --- 线程协调事件 ---
stop_event = threading.Event()

# --- 无人机初始化 ---
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
    """定期发送命令防止 Tello 自动降落。 (仅 Tello 需要)"""
    while not stop_event.is_set():
        if DRONE_TYPE != "tello": #  确保 M4D 不会调用
            break 
        try:
            # 一个几乎不产生移动的指令，仅用于保持连接
            drone.turn_clockwise(1)
        except Exception as e:
            print(f"Keepalive 错误: {e}")
        time.sleep(8)


def drone_live_feed():
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

    # --- 动态确定录制尺寸 ---
    print("正在获取第一帧以确定视频尺寸...")
    first_frame = None
    for _ in range(10): # 尝试 10 次
        first_frame = drone.get_frame() # get_frame() 返回 480x360 RGB
        if first_frame is not None and first_frame.size > 0:
            break
        print("等待第一帧...")
        time.sleep(0.5)
    
    if first_frame is None:
        print("错误：无法获取视频流的第一帧，视频流线程退出。")
        return

    # 从帧确定尺寸 (M4D/Tello 统一返回 480x360)
    EXPECTED_H, EXPECTED_W = first_frame.shape[:2] 
    print(f"视频流已确定。尺寸: {EXPECTED_W}x{EXPECTED_H}")
    
    cv2.namedWindow("Drone Live Feed", cv2.WINDOW_NORMAL)

    try:
        while not stop_event.is_set():
            frame_rgb = drone.get_frame() # get_frame() 返回 RGB 格式
            if frame_rgb is None or frame_rgb.size == 0:
                print("未能获取到有效的视频帧，跳过...")
                time.sleep(0.1)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 检查尺寸是否与第一帧匹配
            h, w = frame_bgr.shape[:2]
            if w != EXPECTED_W or h != EXPECTED_H:
                print(f"警告: 帧尺寸变化! 从 {EXPECTED_W}x{EXPECTED_H} 变为 {w}x{h}. 尝试缩放...")
                frame_bgr = cv2.resize(frame_bgr, (EXPECTED_W, EXPECTED_H))

            # 初始化 video_writer
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



def agent_main():
    """配置并运行 Agent 的主函数。"""
    try:
        # 加载 YAML 内容
        prompt_template_content = ""
        print(PROMPT_PATH)
        if USE_CUSTOM_PROMPT and PROMPT_PATH and os.path.exists(PROMPT_PATH):
            with open(PROMPT_PATH, "r", encoding="utf-8") as f:
                prompt_data = yaml.safe_load(f)
                # 我们需要 'system_prompt' 部分
                prompt_template_content = prompt_data.get('system_prompt', '')
        else:
            print(f"警告: 未找到或未指定有效的 prompt 文件, 将使用默认模板。")
            # 在此定义一个非常基础的备用模板
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

        # --- 根据 LOCAL_MODEL_PATH 初始化模型 ---
        model = None
        # 检查 LOCAL_MODEL_PATH 是否已设置且是一个存在的目录
        if LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH):
            print(f"--- 检测到本地模型路径，使用本地 LLM 模式 ---")
            print(f"模型路径: {LOCAL_MODEL_PATH}")
            try:
                model = TransformersLocalModel(model_path=LOCAL_MODEL_PATH)
            except Exception as e:
                print(f"!! 严重错误: 加载本地模型失败: {e}")
                print("!! 将回退到 API 模式 (如果 API_KEY 可用)...")
                model = None # 确保 model 为 None，以便进入 API 逻辑
        
        elif LOCAL_MODEL_PATH:
            # 如果路径被设置了，但不是一个有效的目录
            print(f"警告: LOCAL_MODEL_PATH ('{LOCAL_MODEL_PATH}') 已设置，但不是一个有效的目录。")
            print("将回退到 API 模式 (如果 API_KEY 可用)...")

        # 如果模型未成功加载 (无论是未设置路径还是加载失败)
        if model is None:
            print(f"--- 使用 API (OpenRouter) LLM 模式 ---")
            if not API_KEY:
                print("错误: 未设置 LOCAL_MODEL_PATH，也未在 .env 中找到 OPENROUTER_API_KEY。")
                print("请设置其中一个来运行 Agent。")
                return # 无法继续

            base_url = os.environ.get("BASE_URL", "[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)")
            # 修复之前代码中可能存在的 Markdown 链接错误
            if "]" in base_url or "[" in base_url:
                 base_url = "[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)"
                 
            # 2. 检查是否为本地 llama-server
            # (环境变量 BASE_URL="http://0.0.0.0:XXXX/v1" 匹配这个)
            is_local_server = "0.0.0.0" in base_url or "localhost" in base_url

            effective_api_key = API_KEY # API_KEY 是从文件顶部加载的
            model_id=MODEL_NAME
            if is_local_server:
                print(f"--- 检测到本地 llama-server 模式 (Base URL: {base_url}) ---")
                effective_api_key = "sk-fake-key"
                model_id="gpt-3.5-turbo"

            else:
                print(f"--- 使用 API (OpenRouter) LLM 模式 ---")
                if not effective_api_key: # 只有在不是本地服务器时，才检查 API_KEY
                    print("错误: 未设置 LOCAL_MODEL_PATH，也未在 .env 中找到 OPENROUTER_API_KEY。")
                    print("请设置其中一个来运行 Agent。")
                    return # 无法继续
        
            model = SimpleLiteLLMModel(
                model_id=model_id,
                api_key=effective_api_key,
                base_url=base_url,
                temperature=TEMPERATURE,
            )
        # --- 初始化模型结束 ---


        agent = AGENT_CLS(
            tools=tools,
            drone_instance=drone,  #  传入无人机实例
            model=model,
            max_steps=100,
            system_prompt_template=prompt_template_content, #  传入模板
        )
        

        # --- 在这里修改你的任务指令 ---
        # (保持不变, 使用您原来的 prompt)
        print("\n开始执行 Agent 任务...")
        # prompt = "在距离白板1.5米的位置看一下白板上的红色笔写的任务。完成白板上的任务"
        prompt = "请去查看白板上红色笔写的发货单(在距离白板1.5米的位置)，然后到飞场中找到需要发货的东西，并依次进行拍照"
        prompt = "按顺序依次飞到地面上标记有‘2’，‘7’，‘3’的地砖的正上方大约1米，水平距离0米处，最后然后降落"
        prompt = "先去白板上看一下，红色笔写的订货信息，然后去标记有数字‘1’的区域查看货物，看看货物数目是否满足发货要求，缺少什么告诉我，在周围查找散落的缺少的目标货物，对它拍照并在面前1米降落"
        prompt = """
        1. 先去白板上看一下，说出白板上写了什么物品，必须确认看到了白板上的物品才进行下一步，没有看清楚白板的话通过向后移动获得更好视野，绝对不能自己假定白板上写的物品。
        2. 然后找到货架，看一下货架上的货物数目是否满足发货要求。
        3. 如果有缺少货物，告诉我缺少什么，在周围查找散落的缺少的目标货物，并对它拍照并降落。"""
        prompt = """
        1. 先去白板上看一下，说出白板上写了什么物品。
        2. 然后找到货架，说出货架上全部物品，然后说出货架上的货物种类是否满足发货要求。
        3. 如果有缺少货物，告诉我缺少什么，在周围查找散落的缺少的目标货物，并对在他面前1米降落。"""
        prompt = "起飞，向前飞行1000厘米，后降落。"
        
        agent.run(prompt) #  调用我们新代理的 run 方法

    except Exception as e:
        print(f"Agent 任务执行出错: {e}")
    finally:
        print("Agent 任务已结束，正在关闭所有线程...")
        stop_event.set()


if __name__ == "__main__":
    try:
        # Tello 无人机需要一个独立的 keepalive 线程
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
            # 直接在主线程中运行 Agent
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
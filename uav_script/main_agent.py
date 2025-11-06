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

# USE_VLM_AGENT = False
# --- 动态导入无人机 Wrapper ---
TelloDrone, MockDrone, AirSimDrone, FanciSwarmDrone, DjiM4DDrone, MockDjiM4DDrone = None, None, None, None, None, None

dotenv.load_dotenv()

# --- 全局配置 ---
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen/qwen3-vl-235b-a22b-instruct")
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))
AGENT_MODE = os.environ.get("AGENT_MODE", "tool")
USE_CUSTOM_PROMPT = True
PROMPT_PATH = os.environ.get("PROMPT_PATH", "toolcalling_agent_cn.yaml")

DRONE_TYPE = os.environ.get("DRONE_TYPE", "djim4d").lower()  

# ----------------------------------------------------

class SimpleLiteLLMModel:
    """
    一个精简的模型封装器，使用 OpenAI 客户端（兼容 OpenRouter）
    """
    def __init__(self, model_id, api_key, base_url, temperature):
        self.model_id = model_id
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0  # [!! 新增 !!] 增加30秒超时
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

class SimpleToolCallingAgent:
    """
    一个精简的工具调用代理，用于替换 smolagents.ToolCallingAgent
    """
    def __init__(self, tools: list, drone_instance, model: SimpleLiteLLMModel, system_prompt_template: str, max_steps: int = 100):
        self.drone_instance = drone_instance
        self.model = model
        self.max_steps = max_steps

        self.tools_list = tools 
        
        self.system_prompt = self._build_system_prompt(system_prompt_template, tools)
        self.messages_history = []
        
        # 用于从响应中提取 Action JSON 的正则表达式
        # (这匹配 'Action:' 直到 '}' 或 '```')
        self.action_regex = re.compile(r"Action:[\s\S]*?(\{[\s\S]*\})", re.DOTALL | re.IGNORECASE)
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
        执行代理的 "思考->行动->观察" 循环
        """
        if images:
            print("警告: SimpleToolCallingAgent (非VLM模式) 不支持图像输入，图像将被忽略。")

        # 1. 初始化
        self.messages_history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        print("--- 系统提示 (已生成) ---")
        print(self.system_prompt)
        print("-----------------")
        print(f"\n--- 开始执行任务: {prompt} ---")

        for step in range(self.max_steps):
            print(f"\n--- 步骤 {step + 1}/{self.max_steps} ---")

            # 1. 思考 (调用 LLM)
            llm_response_text = self.model.chat(self.messages_history)
            
            # 记录模型的完整思考过程
            self.messages_history.append({"role": "assistant", "content": llm_response_text})
            print(f"\n[思考/行动]\n{llm_response_text}")

            # 2. 检查是否为最终答案
            final_answer_match = self.final_answer_regex.search(llm_response_text)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
                print(f"\n--- 任务完成 (检测到 final_answer) ---")
                print(f"最终答案: {final_answer}")
                break

            # 3. 解析行动 (Action)
            action_match = self.action_regex.search(llm_response_text)
            if not action_match:
                print("\n[观察] 错误：模型未提供有效的 'Action:' JSON。正在请求重试...")
                self.messages_history.append({"role": "user", "content": "你没有提供有效的 'Action:' JSON 块。请思考并提供一个工具调用。"})
                continue
            
            # 提取 JSON 字符串 (优先匹配 ```json ... ```)
            json_str = action_match.group(1)
            
            try:
                json_str_cleaned = json_str.replace("\u00A0", " ")
                action_json = json.loads(json_str_cleaned.strip())
                
                tool_name = action_json.get("name")
                tool_args = action_json.get("arguments", {})
                
                if not tool_name:
                    raise ValueError("Action JSON 中缺少 'name' 字段。")
                
                print(f"调用工具: {tool_name}({tool_args})")
                
            except json.JSONDecodeError as e:
                print(f"\n[观察] 错误：解析 Action JSON 失败: {e}")
                print(f"原始 JSON 字符串: {json_str}")
                self.messages_history.append({"role": "user", "content": f"解析 Action JSON 失败: {e}。请检查你的 JSON 格式。"})
                continue
            except ValueError as e:
                print(f"\n[观察] 错误：Action JSON 格式无效: {e}")
                self.messages_history.append({"role": "user", "content": f"Action JSON 格式无效: {e}。"})
                continue

            # 4. 执行工具
            try:
                tool_to_call = self.drone_instance.get_tool_by_name(tool_name)
                
                # 转换参数类型（LLM 总是返回字符串或数字）
                # 我们的工具需要特定类型（例如 int）
                sig = inspect.signature(tool_to_call)
                typed_args = {}
                for param_name, param in sig.parameters.items():
                    if param_name in tool_args:
                        arg_val = tool_args[param_name]
                        if param.annotation == int:
                            typed_args[param_name] = int(float(arg_val)) # 允许 LLM 发送 100.0
                        elif param.annotation == float:
                            typed_args[param_name] = float(arg_val)
                        else:
                            typed_args[param_name] = arg_val # 默认为 str 或 list
                
                observation = tool_to_call(**typed_args)
                
                # 将观察结果格式化为字符串
                if observation is None:
                    observation_str = "操作已执行，无返回值。"
                else:
                    observation_str = str(observation)
                
                print(f"\n[观察]\n{observation_str}")
                
                # 不使用 OpenAI 的 'tool_call' role
                # 按照 YAML 提示，只添加一个 "Observation: ..." 的用户消息
                # (或者我们可以用 'tool' role，但必须伪造 tool_call_id。简单起见，使用用户消息)
                self.messages_history.append({"role": "user", "content": f"Observation: {observation_str}"})

            except Exception as e:
                print(f"\n[观察] 错误：执行工具 '{tool_name}' 失败: {e}")
                self.messages_history.append({"role": "user", "content": f"错误：执行工具 '{tool_name}' 失败: {e}。请重新规划。"})

        if step == self.max_steps - 1:
            print(f"\n--- 已达到最大步骤 ({self.max_steps})，任务终止 ---")



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

        #  初始化我们自己的模型和代理
        model = SimpleLiteLLMModel(
            model_id=MODEL_NAME,
            api_key=API_KEY,
            base_url=os.environ.get("BASE_URL", "[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)"),
            temperature=TEMPERATURE,
        )

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
        prompt = "起飞，向前飞行10米，后降落。"
        
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
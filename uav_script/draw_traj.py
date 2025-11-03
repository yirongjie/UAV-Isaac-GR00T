import time

import json_numpy
import numpy as np
import requests
from pathlib import Path

from utils.rotation import body_to_world_pose

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config

json_numpy.patch()

# Build observation dictionary. Prefer using local `test.jpg` for the ego view when available.
def load_test_ego_view(path: str = "test.jpg") -> np.ndarray:
    from PIL import Image
    p = Path(path)
    if p.exists():
        img = Image.open(p).convert("RGB")
        img = img.resize((256, 256))
        arr = np.array(img, dtype=np.uint8)
    else:
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
    # many code paths expect a batch dimension of 1
    return arr[np.newaxis, ...]

obs = {
    "video.ego_view": load_test_ego_view("test.jpg"),
    "state.drone": np.array([[0.0]]),
    "annotation.human.action.task_description": ["Go straight"],
}

ACTION_KEY = "action.delta_pose"

if __name__ == "__main__":
    t = time.time()
    response = requests.post(
        "http://10.29.230.87:5555/act",
        json={"observation": obs},
    )
    print(f"used time {time.time() - t}")
    action = response.json()

    points = []
    curr_point = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x,y,z,yaw,pitch,roll
    last_point = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x,y,z,yaw,pitch,roll
    for i, pose in enumerate(action[ACTION_KEY]):
        print(f"Step {i}: {pose}")
        curr_point = np.array([pose[0], pose[1], pose[2], 0.0, 0.0, pose[3]])  # x,y,z,roll,pitch,yaw
        world_pose = body_to_world_pose(last_point, curr_point)
        points.append([world_pose[0].item(), world_pose[1].item(), world_pose[2].item()])
        last_point = world_pose
        
    for i, p in enumerate(points):
        print(f"Point {i}: {p}")
    from utils.draw import plot_3d_trajectory
    plot_3d_trajectory(points, save_path="out.png", title="3D Trajectory")
    
    # DATA_CONFIG_CLS = "examples.UAV_Flow.custom_data_config:UAVFlowDataConfig"
    # DATASET_PATH = "../UAV-Flow-Gr00t/UAVFlowLeRobot"

    # data_config_cls = load_data_config(DATA_CONFIG_CLS)
    # modality_configs = data_config_cls.modality_config()
    # # transforms = data_config_cls.transform()
    
    # dataset = LeRobotSingleDataset(
    #     dataset_path=DATASET_PATH,
    #     modality_configs=modality_configs,
    #     embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    #     video_backend="torchvision_av",
    # )

    # print(dataset[0])

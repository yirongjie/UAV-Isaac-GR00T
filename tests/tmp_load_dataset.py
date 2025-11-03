from gr00t.utils.misc import any_describe
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.dataset import ModalityConfig
from gr00t.data.schema import EmbodimentTag
import os
import gr00t

if __name__ == "__main__":
    # REPO_PATH is the path of the pip install gr00t repo and one level up
    REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
    DATA_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")

    print("Loading dataset... from", DATA_PATH)
    
    # 2. modality configs
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=[0, 1],
            modality_keys=["video.ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "state.left_arm",
                "state.left_hand",
                "state.left_leg",
                "state.neck",
                "state.right_arm",
                "state.right_hand",
                "state.right_leg",
                "state.waist",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "action.left_hand",
                "action.right_hand",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description", "annotation.human.validity"],
        ),
    }
    
    # 3. gr00t embodiment tag
    embodiment_tag = EmbodimentTag.GR1

    # load the dataset
    dataset = LeRobotSingleDataset(DATA_PATH, modality_configs,  embodiment_tag=embodiment_tag)

    print('\n'*2)
    print("="*100)
    print(f"{' Humanoid Dataset ':=^100}")
    print("="*100)

    # print the 7th data point
    resp = dataset[7]
    any_describe(resp)
    print("--------------------------------------------------------")
    print(resp.keys())
    print("--------------------------------------------------------")
    print(type(resp))
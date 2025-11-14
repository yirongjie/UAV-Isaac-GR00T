import cv2
import time
from DJI_M4D_smol_wrapper import DjiM4DDrone

def main():
    try:
        # 初始化无人机实例
        print("正在初始化无人机...")
        drone = DjiM4DDrone()
        print("无人机初始化完成")

        time.sleep(2)

        # 调用get_frame_vlm获取图像（使用480p分辨率）
        print("正在获取无人机摄像头图像...")
        start_time = time.time()
        # ret = drone.check_current_view()
        ret = drone.face_compare(False, "/home/dji/LMFly/UAV-Isaac-GR00T/uav_script/current_frame_vlm_person_0.png", use_api=True, detection_method="yolo")
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 输出结果和耗时
        print(f"人脸比对结果: {ret}")
        print(f"face_compare 方法执行时间: {elapsed_time:.6f} 秒")  # 保留6位小数
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 可以在这里添加资源释放代码（如果需要）
        pass

if __name__ == "__main__":
    # main()

    drone = DjiM4DDrone()
    
    # 测试路径
    test_img = "/home/dji/LMFly/UAV-Isaac-GR00T/uav_script/temp2.jpg"
    
    print("=== 开始离线测试 check_view_optimized ===")
    result = drone.check_view_optimized(use_api=False, api_provider="baidu", debug_image_path=test_img, use_depth_model=True)
    
    if result:
        print(f"测试成功！返回坐标: {result}")
        # 此时目录下应该生成了 current_frame_xyz_2.png，可以打开查看画图结果
    else:
        print("测试失败：未找到目标或匹配失败。")
import zipfile
import os
import math
import time
from datetime import datetime

# --- 配置 ---
# C++ Server 和此 Python 脚本都必须能访问这个路径
KMZ_SAVE_PATH = "/home/dji/LMFly/UAV-Isaac-GR00T/uav_script/dynamic_mission.kmz" 
EARTH_RADIUS_M = 6371000.0

# --- KML/WPML 生成器 (基于您上传的 waylines.wpml) ---
# (这些函数是从 run_dynamic_mission.py 迁移过来的)

def generate_template_kml():
    """
    生成一个最小化的、有效的 template.kml 内容。
    """
    current_time_ms = int(time.time() * 1000)
    template_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:wpml="http://www.dji.com/wpmz/1.0.3">
  <Document>
    <name>template</name>
    <wpml:createTime>{current_time_ms}</wpml:createTime>
    <wpml:updateTime>{current_time_ms}</wpml:updateTime>
  </Document>
</kml>
"""
    return template_content

def generate_waylines_wpml(path_coords, drone_info, payload_info):
    """
    生成一个 100% 模仿 waylines.wpml 结构的 .wpml 文件。
    
    :param path_coords: (lon, lat, alt) 坐标元组的列表
    :param drone_info: 字典, e.g., {'enum': 77, 'sub': 0}
    :param payload_info: 字典, e.g., {'enum': 66, 'sub': 0, 'pos': 0}
    """
    
    # 1. 为每个航点(Placemark)创建 XML 块
    placemark_list_str = ""
    for i, (lon, lat, alt) in enumerate(path_coords):
        
        # --- 航点动作 ---
        action_group_str = ""
        # 示例：在第一个航点拍照
        # if i == 0:
        #     action_group_str = f"""
        # <wpml:actionGroup>
        #   <wpml:actionGroupId>{i}</wpml:actionGroupId>
        #   <wpml:actionGroupStartIndex>{i}</wpml:actionGroupStartIndex>
        #   <wpml:actionGroupEndIndex>{i}</wpml:actionGroupEndIndex>
        #   <wpml:actionGroupMode>sequence</wpml:actionGroupMode>
        #   <wpml:actionTrigger>
        #     <wpml:actionTriggerType>reachPoint</wpml:actionTriggerType>
        #   </wpml:actionTrigger>
        #   <wpml:action>
        #     <wpml:actionId>0</wpml:actionId>
        #     <wpml:actionActuatorFunc>takePhoto</wpml:actionActuatorFunc>
        #     <wpml:actionActuatorFuncParam>
        #       <wpml:payloadPositionIndex>{payload_info['pos']}</wpml:payloadPositionIndex>
        #       <wpml:useGlobalPayloadLensIndex>0</wpml:useGlobalPayloadLensIndex>
        #     </wpml:actionActuatorFuncParam>
        #   </wpml:action>
        # </wpml:actionGroup>
        # """
        if i == len(path_coords) - 1: #很重要hover
            # 这是最后一个航点。
            # 我们必须添加一个 "hover" 动作来覆盖 PSDK 的 "goHome" 默认行为。
            action_group_str = f"""
            <wpml:actionGroup>
              <wpml:actionGroupId>{i}</wpml:actionGroupId>
              <wpml:actionGroupStartIndex>{i}</wpml:actionGroupStartIndex>
              <wpml:actionGroupEndIndex>{i}</wpml:actionGroupEndIndex>
              <wpml:actionGroupMode>sequence</wpml:actionGroupMode>
              <wpml:actionTrigger>
                <wpml:actionTriggerType>reachPoint</wpml:actionTriggerType>
              </wpml:actionTrigger>
              <wpml:action>
                <wpml:actionId>0</wpml:actionId>
                <wpml:actionActuatorFunc>hover</wpml:actionActuatorFunc>
                <wpml:actionActuatorFuncParam>
                    <wpml:hoverTime>5000</wpml:hoverTime>
                </wpml:actionActuatorFuncParam>
              </wpml:action>
            </wpml:actionGroup>
            """

        # --- 模仿 waylines.wpml 的 Placemark 结构 ---
        placemark_list_str += f"""
      <Placemark>
        <Point>
          <coordinates>{lon:.8f},{lat:.8f}</coordinates>
        </Point>
        <wpml:index>{i}</wpml:index>
        <wpml:executeHeight>{alt:.1f}</wpml:executeHeight>
        <wpml:waypointSpeed>5</wpml:waypointSpeed>
        <wpml:waypointHeadingParam>
          <wpml:waypointHeadingMode>followWayline</wpml:waypointHeadingMode>
          <wpml:waypointHeadingAngle>0</wpml:waypointHeadingAngle>
          <wpml:waypointPoiPoint>0.000000,0.000000,0.000000</wpml:waypointPoiPoint>
          <wpml:waypointHeadingAngleEnable>0</wpml:waypointHeadingAngleEnable>
          <wpml:waypointHeadingPoiIndex>0</wpml:waypointHeadingPoiIndex>
        </wpml:waypointHeadingParam>
        <wpml:waypointTurnParam>
          <wpml:waypointTurnMode>toPointAndStopWithDiscontinuityCurvature</wpml:waypointTurnMode>
          <wpml:waypointTurnDampingDist>0</wpml:waypointTurnDampingDist>
        </wpml:waypointTurnParam>
        <wpml:useStraightLine>1</wpml:useStraightLine>
        {action_group_str}
      </Placemark>
        """

    # 2. 组装完整的 KML/WPML
    wpml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:wpml="http://www.dji.com/wpmz/1.0.3">
  <Document>
    <wpml:missionConfig>
        <wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode>
        <wpml:finishAction>hover</wpml:finishAction>
        <wpml:exitOnRCLost>executeLostAction</wpml:exitOnRCLost>
        <wpml:executeRCLostAction>hover</wpml:executeRCLostAction>
      <wpml:takeOffSecurityHeight>20</wpml:takeOffSecurityHeight>
      <wpml:globalTransitionalSpeed>15</wpml:globalTransitionalSpeed>
      <wpml:droneInfo>
        <wpml:droneEnumValue>{drone_info['enum']}</wpml:droneEnumValue>
        <wpml:droneSubEnumValue>{drone_info['sub']}</wpml:droneSubEnumValue>
      </wpml:droneInfo>
      <wpml:payloadInfo>
        <wpml:payloadEnumValue>{payload_info['enum']}</wpml:payloadEnumValue>
        <wpml:payloadSubEnumValue>{payload_info['sub']}</wpml:payloadSubEnumValue>
        <wpml:payloadPositionIndex>{payload_info['pos']}</wpml:payloadPositionIndex>
      </wpml:payloadInfo>
    </wpml:missionConfig>
    
    <Folder>
      <wpml:templateId>0</wpml:templateId>
      <wpml:executeHeightMode>relativeToStartPoint</wpml:executeHeightMode>
      <wpml:waylineId>0</wpml:waylineId>
      <wpml:distance>100.0</wpml:distance> 
      <wpml:duration>30.0</wpml:duration>
      <wpml:autoFlightSpeed>5</wpml:autoFlightSpeed>
      
      {placemark_list_str}
      
    </Folder>
  </Document>
</kml>
"""
    return wpml_content

def create_kmz_file(template_content, waylines_content, output_path):
    """
    (已验证有效)
    将 template.kml 和 waylines.wpml 字符串打包成一个有效的 KMZ 文件。
    """
    temp_zip_file = output_path + ".tmp.zip"
    try:
        with zipfile.ZipFile(temp_zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('wpmz/template.kml', template_content)
            zf.writestr('wpmz/waylines.wpml', waylines_content)
        os.rename(temp_zip_file, output_path)
        return True
    except Exception as e:
        print(f"  -> 错误: 生成 KMZ 文件失败: {e}")
        if os.path.exists(temp_zip_file):
            os.remove(temp_zip_file)
        return False

def calculate_new_gps(lat_deg, lon_deg, north_m, east_m):
    """
    根据北向/东向米数，计算新的 GPS 坐标
    """
    dLat_rad = north_m / EARTH_RADIUS_M
    dLon_rad = east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat_deg)))

    new_lat_deg = lat_deg + math.degrees(dLat_rad)
    new_lon_deg = lon_deg + math.degrees(dLon_rad)
    
    return new_lat_deg, new_lon_deg
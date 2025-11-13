import bpy
import math

def add_camera(cam_pos, cam_rot, name):
    bpy.ops.object.camera_add(enter_editmode=False, location=cam_pos, rotation=cam_rot)
    print(bpy.data.objects.keys())
    cam = bpy.data.objects['Camera']
    cam.scale = [5, 25, 5]
    cam.data.lens = 17.5
    if name == 'Main':
        cam.data.lens = 35
    cam.name = name + '_cam'
    bpy.context.scene.camera = cam # add cam so it's rendered
    
def get_camera(name):
    cam = bpy.data.objects[name]
    bpy.context.scene.camera = cam

def add_orbit_camera(center_point, radius, name="Orbit_cam"):
    # 创建一个Empty作为旋转中心,将其高度调整到与相机相同
    target_height = center_point[2] + 1.2  # 与相机同高
    target_position = (center_point[0], center_point[1], target_height)
    bpy.ops.object.empty_add(location=target_position)
    empty = bpy.context.active_object
    empty.name = f"{name}_target"
    
    # 创建相机
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = name
    
    # 设置相机位置
    camera.location = (center_point[0] + radius + 2.5, center_point[1], target_height)
    
    # 添加Track To约束,使相机始终朝向目标点
    track = camera.constraints.new(type='TRACK_TO')
    track.target = empty
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    
    # 设置父子关系
    camera.parent = empty
    
    return camera, empty

def animate_orbit_camera(camera, empty, start_frame, end_frame, revolutions=1):
    # 设置旋转动画
    empty.rotation_euler = (0, 0, 0)
    empty.keyframe_insert(data_path="rotation_euler", frame=start_frame)
    
    empty.rotation_euler = (0, 0, math.radians(360 * revolutions))
    empty.keyframe_insert(data_path="rotation_euler", frame=end_frame)
    
    # 设置插值方式为线性
    for fcurve in empty.animation_data.action.fcurves:
        for kf in fcurve.keyframe_points:
            kf.interpolation = 'LINEAR'

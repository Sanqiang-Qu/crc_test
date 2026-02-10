import numpy as np
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.io.ponifile import PoniFile
import Tool.SAXS_CRC_for_twodpattern as sc
import pickle

def compute_beam_center(dist, poni1, poni2, rot1, rot2, rot3):
    """
    计算Beam center在探测器上的坐标（与poni1/poni2同单位）及到样品点的距离。
    
    参数:
        poni1 (float): PONI点在探测器y轴方向的坐标（米）
        poni2 (float): PONI点在探测器x轴方向的坐标（米）
        distance (float): 样品到探测器平面的垂直距离（米）
        rot1 (float): 探测器绕自身x轴旋转角（弧度）
        rot2 (float): 探测器绕自身y轴旋转角（弧度）
        rot3 (float): 探测器绕自身z轴旋转角（弧度）
    
    返回:
        xb (float): Beam center在探测器x轴方向的坐标（米）
        yb (float): Beam center在探测器y轴方向的坐标（米）
        beam_to_sample_distance (float): Beam center到样品点的直线距离（米）
    """
    # 1. 初始三个点：坐标原点、探测器原点、BeamCenter点
    Detector0 = np.array([-dist, -poni2, -poni1]) 
    BC0 = np.array([-dist, 0, 0]).reshape(1, 3)

    # 2. 探测器和BC0绕着Z轴顺时针旋转
    Z = np.array([0, 0, 1])
    result1 = sc.rotateVector(BC0, Z, -rot1*180/np.pi)
    BC1 = result1["real_vector"]
    
    # 3. Detector1和BC1绕着Y轴顺时针旋转
    Y = np.array([0, 1, 0])
    result2 = sc.rotateVector(BC1, Y, -rot2*180/np.pi)
    BC2 = result2["real_vector"]
    
    # 4. 计算BC2和X-ray的夹角
    X_ray = np.array([-1, 0, 0])    
    angle = sc.vector_delta_angle(BC2.flatten(), X_ray)
    
    # 5. 计算当前探测器坐标系下的透射光斑坐标
    Tmit2 = np.linalg.norm(BC2) / np.cos(angle) * X_ray
    Tmit2 = Tmit2.reshape(1, 3)
    
    # 6. 计算探测器坐标系下的透射光斑坐标
    result3 = sc.rotateVector(Tmit2, Y, rot2*180/np.pi)
    Tmit1 = result3["real_vector"]
    
    result4 = sc.rotateVector(Tmit1, Z, rot1*180/np.pi)
    Tmit0 = result4["real_vector"]
    
    Tmit = Tmit0 - Detector0       
    
    # 7. 计算透射光斑距离
    Tmit_dist = np.linalg.norm(Tmit0)    
    return Tmit.flatten(), Tmit_dist

def convert_poni_to_fit2d(poni_file: str) -> dict:
    """
    将 pyFAI 标定生成的 PONI 文件转换为 FIT2D 格式的表观参数，包含X射线能量。

    参数:
        poni_file (str): PONI 文件路径。

    返回:
        dict: 包含 FIT2D 格式表观参数及能量的字典，键为：
            - distance: 样品到探测器的直接距离（单位：mm）
            - center_x: 探测器中心的 X 坐标（单位：像素）
            - center_y: 探测器中心的 Y 坐标（单位：像素）
            - tilt: 探测器倾斜角（单位：度）
            - tilt_plan_rotation: 倾斜面的旋转角（单位：度）
            - wavelength: X射线波长（单位：Å）
    """
    # 1. 从 PONI 文件加载 AzimuthalIntegrator
    # poni_file = "C:/Users/85415/Desktop/AgBh_poni.poni"
    ai = AzimuthalIntegrator()
    ai.load(poni_file)
    wavelength, pixel1, pixel2 = ai.wavelength, ai.pixel1, ai.pixel2
    dist, poni1, poni2 = ai.dist, ai.poni1, ai.poni2
    rot1, rot2, rot3 = ai.rot1, ai.rot2, ai.rot3
    
    Tmit, Tmit_dist = compute_beam_center(dist, poni1, poni2, rot1, rot2, rot3)
    center_y, center_z, direct_dist = Tmit[1]/pixel1, Tmit[2]/pixel2, Tmit_dist

    # 整理结果
    return {
        
        # FIT2D
        "distance": direct_dist * 1e3,
        "center_y": center_y,
        "center_z": center_z,
        "pixel": pixel1 * 1e3,
        "wavelength": wavelength * 1e10,
        
        # PONI
        "sample_dist": ai.dist * 1e3,
        "poni1": ai.poni1 * 1e3,
        "poni2": ai.poni2 * 1e3,
        "rot1": ai.rot1,
        "rot2": ai.rot2,
        "rot3": ai.rot3,
        "poni1_pixel": ai.poni1 / pixel1,
        "poni2_pixel": ai.poni2 / pixel2,
    }


# 计算散射矢量q和方位角phi
def calculate_q_phi_from_poni(Y, Z, pattern, poni_para):
    
    # 读取PONI文件
    # Y = np.load("C:/Users/85415/Desktop/Y.npy"); Z = np.load("C:/Users/85415/Desktop/Z.npy");
    # pattern = np.load("C:/Users/85415/Desktop/pattern.npy");
    # # # 从文件加载字典
    # with open("C:/Users/85415/Desktop/para.pkl", "rb") as f:
    #     poni_para = pickle.load(f)  # 反序列化恢复字典
    print(poni_para)
    wavelength = poni_para["wavelength"]
    pixel = poni_para["pixel"]
    sample_dist = poni_para["sample_dist"]
    poni1, poni2 = poni_para["poni1"], poni_para["poni2"]
    rot1, rot2, rot3 = poni_para["rot1"], poni_para["rot2"], poni_para["rot3"]
    
    # 虚假探测器的坐标化
    fake_center_y, fake_center_z = (poni2 / pixel), (poni1 / pixel)
    # print(fake_center_y, fake_center_z)
    
    ls_Y0, ls_Z0 = (Y - fake_center_y) * pixel, (Z - fake_center_z) * pixel
    ls_X0 = -sample_dist * np.ones_like(ls_Y0)
    
    # 虚假探测器的旋转
    # 1. 探测器绕着Z轴顺时针旋转
    Z = [0, 0, 1]
    ls_XYZ0 = np.column_stack((ls_X0.reshape(-1, 1), ls_Y0.reshape(-1, 1), ls_Z0.reshape(-1, 1)))
    
    
    result1 = sc.rotateVector(ls_XYZ0, Z, -rot1 * 180/np.pi)
    ls_XYZ1 = result1["real_vector"]
    
    # 2. 探测器绕着Y轴顺时针旋转
    Y = [0, 1, 0]
    result2 = sc.rotateVector(ls_XYZ1, Y, -rot2 * 180/np.pi)
    ls_XYZ2 = result2["real_vector"]
    
    # 真实探测器
    ls_X = ls_XYZ2[:, 0].reshape(ls_X0.shape)
    ls_Y = ls_XYZ2[:, 1].reshape(ls_Y0.shape)
    ls_Z = ls_XYZ2[:, 2].reshape(ls_Z0.shape)
    
    # 探测器向量化
    norm_detector = np.sqrt(ls_X**2 + ls_Y**2 + ls_Z**2)
    vx, vy, vz = ls_X/norm_detector, ls_Y/norm_detector, ls_Z/norm_detector
    
    v_I0 = [-1, 0, 0]
    sx, sy, sz = vx-v_I0[0], vy-v_I0[1], vz-v_I0[2]    
    return {
        "sx": sx, 
        "sy": sy, 
        "sz": sz, 
        "wavelength": wavelength,
        "ls_Y": ls_Y,
        "ls_Z": ls_Z,
        "ls_X": ls_X,
        }   
    

# 16. 参数标定的计算部分
def calibrate_parameters(Y, Z, pattern, poni_para):
    """
    独立参数标定函数 - 计算散射矢量参数
    
    参数:
        Y: Y坐标网格
        Z: Z坐标网格  
        pattern: 图像数据数组
        poni_para: PONI文件参数
        
    返回:
        dict: 包含标定结果的字典
            - 'wavelength': X射线波长
            - 'sx', 'sy', 'sz': 散射矢量分量
            - 'ls_Y', 'ls_Z': 探测器坐标
            - 'q': 散射矢量模量
            - 'qxy1', 'qz1': GIXRS坐标系下的分量
            - 'qy2', 'qz2': 常规坐标系下的分量  
            - 'phi': 方位角
            - 'pattern1': 处理后的图像数据
            - 'qxy1_min', 'qxy1_max', 'qz1_min', 'qz1_max': 显示范围
            - 'qy2_min', 'qy2_max', 'qz2_min', 'qz2_max': 显示范围
    """
    try:
        # 调用现有标定函数
        params = calculate_q_phi_from_poni(Y, Z, pattern, poni_para)
        
        wavelength = params["wavelength"]
        sx, sy, sz = params["sx"], params["sy"], params["sz"]
        ls_Y, ls_Z = params["ls_Y"], params["ls_Z"]
        
        # qxy-qz模块计算
        k = 2 * np.pi / wavelength
        qx, qy, qz = k * sx, k * sy, k * sz
        
        q = np.sqrt(qx**2 + qy**2 + qz**2)
        qxy1 = np.sqrt(qx**2 + qy**2) 
        qxy1[qy < 0] = -qxy1[qy < 0]
        qz1 = qz
        
        # qy2-qz2模块计算
        norm_YZ = np.sqrt(ls_Y**2 + ls_Z**2) 
        phi = np.arccos(ls_Y / norm_YZ) * 180 / np.pi
        phi[qz < 0] = 360 - phi[qz < 0] 
        phi[np.isnan(phi)] = 0
        qy2, qz2 = q * np.cos(phi * np.pi / 180), q * np.sin(phi * np.pi / 180)
        
        # 构造GIXRS劈裂数据
        p1 = np.where(qxy1[0, :] < 0)[0][-1]
        p2 = np.where(qxy1[0, :] > 0)[0][0]
        
        add_qxy = np.column_stack([
            qxy1[:, p1] + 1e-3, 
            np.zeros_like(qxy1[:, p1]), 
            qxy1[:, p2] - 1e-3
        ])
        
        add_qz = np.column_stack([
            qz1[:, p1], 
            0.5 * (qz1[:, p1] + qz1[:, p2]), 
            qz1[:, p2]
        ])
        
        add_pattern = np.zeros_like(add_qz)
        
        # 合并数据
        qxy1_result = np.column_stack([qxy1[:, 0:p1+1], add_qxy, qxy1[:, p2:]])
        qz1_result = np.column_stack([qz1[:, 0:p1+1], add_qz, qz1[:, p2:]])
        pattern1_result = np.column_stack([pattern[:, 0:p1+1], add_pattern, pattern[:, p2:]])
        
        # 计算显示范围
        qxy1_min_cal = np.max(qxy1_result[:, 0])
        qxy1_max_cal = np.min(qxy1_result[:, -1])
        qz1_min_cal = np.max(qz1_result[0, :])
        qz1_max_cal = np.min(qz1_result[-1, :])
        
        qy2_min_cal = np.max(qy2[:, 0])
        qy2_max_cal = np.min(qy2[:, -1])
        qz2_min_cal = np.max(qz2[0, :])
        qz2_max_cal = np.min(qz2[-1, :])
        
        return {
            'wavelength': wavelength,
            'sx': sx, 'sy': sy, 'sz': sz,
            'ls_Y': ls_Y, 'ls_Z': ls_Z,
            'q': q, 'phi': phi,
            'qxy1': qxy1_result, 'qz1': qz1_result,
            'qy2': qy2, 'qz2': qz2,
            'pattern1': pattern1_result,
            'qxy1_min': qxy1_min_cal, 'qxy1_max': qxy1_max_cal,
            'qz1_min': qz1_min_cal, 'qz1_max': qz1_max_cal,
            'qy2_min': qy2_min_cal, 'qy2_max': qy2_max_cal,
            'qz2_min': qz2_min_cal, 'qz2_max': qz2_max_cal
        }
        
    except Exception as e:
        print(f"参数标定计算错误: {e}")
        return None




# SAXS_CRC_for_twodpattern.py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.stats import binned_statistic
from tkinter import filedialog
import fabio
import tifffile
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks, savgol_filter
from scipy import interpolate
from matplotlib.widgets import Button

# %% 1. create array
def arange(start, step, stop):
    num = int(np.round((stop - start) / step)) + 1  # 自动计算点数
    return np.linspace(start, stop, num).reshape(-1, 1)


# %% 2. get beta distribution
def get_beta(Rmin, Rmax, a, b, number):

    Y = np.zeros((number, 3))

    Y[:, 0] = np.linspace(Rmin, Rmax, number)
    Y[:, 1] = (((Y[:, 0]-Rmin)/(Rmax-Rmin))**(a-1) *
               ((Rmax-Y[:, 0])/(Rmax-Rmin))**(b-1))

    Y1 = simpson(Y[:, 1], Y[:, 0])
    Y[:, 1] = Y[:, 1] / Y1

    Y[:, 2] = 4*(np.pi)*Y[:, 0]**3/3 * Y[:, 1]
    Y2 = simpson(Y[:, 2], Y[:, 0])
    Y[:, 2] = Y[:, 2] / Y2
    Y[0, 0] = 1e-6
    return Y


# %% 3. get Intensity of sphere with volume fraction
def Sphere_I_V(q, Y):
    q[0] = 1e-6
    Y[0, 0] = 1e-6

    U = q.reshape(-1, 1) * Y[0, :].reshape(-1, 1).T
    F = (3 / U**3) * (np.sin(U)-U * np.cos(U))
    I0 = F * F * Y[1, :] * (4 * np.pi * Y[0, :]**3 / 3)
    I1 = I0.mean(axis=1, keepdims=True)
    I1 = np.column_stack((q.reshape(-1, 1), I1, np.log(I1)))
    return I1


# %% 4. change the stepwidth of scattering data
def change_stepwidth(I3, stepwidth):
    I3 = np.array(I3)
    number = int(np.round((I3[:, 0].max() - 0) / stepwidth)) + 1
    q = np.linspace(0, I3[:, 0].max(), number).reshape(-1, 1)
    q[0, 0] = 1e-6
    min_q, max_q = I3[:, 0].min(), I3[:, 0].max()
    k1, k2 = np.searchsorted(q[:, 0], [min_q, max_q], side='right')-1
    linshi = CubicSpline(I3[:, 0], I3[:, 2])(q[k1:k2+1, 0]).reshape(-1, 1)
    return np.column_stack((q[k1:k2+1, :], np.exp(linshi), linshi))


# %% 5. auto_adjust_axes坐标轴自动调整功能
def auto_adjust_axes(ax=None, margin=0.05):
    """自动调整坐标轴范围适配所有数据"""
    ax = ax or plt.gca()
    ax.relim()  # 重新计算数据限制
    ax.autoscale_view()  # 基于新数据自动调整视图
    ax.figure.canvas.draw()  # 立即刷新图表

    # 手动添加边距
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min - (x_max - x_min) * margin,
                x_max + (x_max - x_min) * margin)
    ax.set_ylim(y_min - (y_max - y_min) * margin,
                y_max + (y_max - y_min) * margin)


# %% 6. porod_correct
def porod_correct(I_input):
    plt.close('all')
    C = 4 * np.log(I_input[:, 0]) + np.log(I_input[:, 1])
    E = arange(0, 0.02, 100)
    C = np.tile(C[:, np.newaxis], (1, E.shape[0]))
    # np.newaxis将一维数组转换为列向量（二维数组）

    D = I_input[:, 0] ** 2
    D = np.tile(D[:, np.newaxis], (1, E.shape[0]))

    C_corrected = C - D * E.T
    remove_num = int(np.floor(C_corrected.shape[0] * 4 / 5))
    C_remaining = C_corrected[remove_num:, :]
    # 只保留C_corrected的尾巴部分

    std_devs = np.std(C_remaining, axis=0, ddof=0)  # ddof 用来计算总体标准差
    k = E[np.argmin(std_devs)]

    # 绘图数据准备
    A = np.column_stack((I_input[:, 0] ** 2, 4 * np.log(I_input[:, 0])
                         + np.log(I_input[:, 1])))
    A = np.column_stack((A, A[:, 1] - k * A[:, 0]))

    plt.ion()  # 启用交互模式
    F1 = plt.figure(1, figsize=(10, 10))
    Line1 = F1.add_subplot(111)
    Line1.set_xlabel(r'$q^2\ (1/\mathrm{nm})^2$', fontsize=10)
    Line1.set_ylabel(r'$\ln(q^4I)$', fontsize=10)
    Line1.set_title('Porod Correction', fontsize=12)
    orig_line, = Line1.plot(A[:, 0], A[:, 1],
                            'k-', lw=1.5, label='Original Data')
    B_initial = A[:, 1] - k * A[:, 0]
    corr_line, = Line1.plot(A[:, 0], B_initial, 'r-', lw=1.5,
                            label=f'Corrected (k={k[0]:.2f})')
    Line1.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    print(f'程序给出的Porod校正系数为: {k[0]:.2f}')

    kk = k
    for _ in range(100):
        a = input('\n校正完成输入1:')
        if a == '1':
            new_B = A[:, 1] - kk * A[:, 0]
            break
        else:
            try:
                kk = float(input('\n输入Porod校正系数:'))
                new_B = A[:, 1] - kk * A[:, 0]
                corr_line.set_ydata(new_B)
                corr_line.set_label(f'Corrected (k={kk:.3f})')
                Line1.legend()
                F1.canvas.flush_events()
                F1.canvas.draw()
                plt.pause(0.01)
            except ValueError:
                print("错误：请输入数字")
    plt.ioff()  # 关闭交互模式
    plt.show()
    I_porod = np.column_stack((I_input[:, 0],
                               np.exp(new_B - 4 * np.log(I_input[:, 0])),
                               new_B - 4 * np.log(I_input[:, 0])))

    plt.ion()
    F2 = plt.figure(2, figsize=(10, 10))
    Line2 = F2.add_subplot(111)
    Line2.set_xlabel(r'$q\ (1/\mathrm{nm})$', fontsize=10)
    Line2.set_ylabel(r'$ I (a.u.) $', fontsize=10)
    Line2.set_title('log-log plot', fontsize=12)
    Line2.loglog(I_input[:, 0], I_input[:, 1], 'k-', lw=1.5)
    Line2.loglog(I_porod[:, 0], I_porod[:, 1], 'b-', lw=1.5)
    plt.tight_layout()
    plt.draw()
    F2.canvas.flush_events()
    F2.canvas.draw()
    plt.pause(0.1)
    plt.ioff()
    plt.show()
    return I_porod, kk


# %% 7. cut_Iq
def cut_Iq(I_porod):
    plt.close('all')
    A0 = np.column_stack((np.log(I_porod[:, 0]), I_porod[:, 2]))
    plt.ion()
    F3 = plt.figure(3, figsize=(10, 10))
    Line3 = F3.add_subplot(111)
    Line3.set_xlabel('lnq', fontsize=20)
    Line3.set_ylabel('lnI', fontsize=20)
    Line3.set_title('double log plot', fontsize=12)
    Line3.plot(A0[:, 0], A0[:, 1], 'k-', lw=2)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)
    for _ in range(100):
        a = float(input('\n输入截断的上限:'))
        k = np.argmin(np.abs(A0[:, 0] - a))
        Line3.plot(A0[0:k, 0], A0[0:k, 1], lw=2)
        F3.canvas.flush_events()
        F3.canvas.draw()
        plt.pause(0.01)

        b = input('\n截断完成请输入1:')
        if b == '1':
            break
    plt.ioff()  # 关闭交互模式
    plt.show()
    A1 = np.column_stack((np.exp(A0[0:k, 0]), np.exp(A0[0:k, 1]), A0[0:k, 1]))
    return A1


# %% 8. 计算所有的Beta尺寸分布 (TensorFlow GPU加速版本)
# def SESDC_single(swarm, I_input): 
 
#     # 确保计算在GPU上执行
#     Beta_number = 201
#     Beta_R = np.zeros((swarm.shape[0], Beta_number))
#     Beta_Y = np.zeros((swarm.shape[0], Beta_number))

#     # 遍历每个种群
#     for i in range(swarm.shape[0]):
#         Y1 = get_beta(0, swarm[i, 2], swarm[i, 0], swarm[i, 1], Beta_number)  
#         Beta_R[i, :], Beta_Y[i, :] = Y1[:, 0].T, Y1[:, 1].T

#     # 转换为TensorFlow张量
#     tf_R = tf.constant(Beta_R, dtype=tf.float32)[:, :, tf.newaxis]
#     tf_Y = tf.constant(Beta_Y, dtype=tf.float32)[:, :, tf.newaxis]
#     tf_V_phi = (4 * tf.constant(np.pi, dtype=tf.float32) * tf_R**3 / 3) * tf_Y

#     count_number = 100
#     q = I_input[:, 0].reshape((1, 1, I_input.shape[0]))
#     n = np.arange(0, I_input.shape[0], count_number)
#     n = np.append(n, I_input.shape[0])

#     # 2. TensorFlow GPU计算散射强度
#     I2 = []
#     for i in range(n.shape[0]-1):
#         U1 = np.tile(q[0, 0, n[i]:n[i+1]], (Beta_Y.shape[0], Beta_Y.shape[1], 1))
#         U1 = tf.constant(U1, dtype=tf.float32) * tf_R
#         U1 = ((3 / (U1**3)) * (tf.sin(U1) - U1 * tf.cos(U1)))**2 * tf_V_phi
#         I1 = tf.reduce_sum(U1, axis=1)
#         I2.append(I1)

#     # 合并结果并转换回numpy
#     I3_tf = tf.concat(I2, axis=1)
#     I3 = I3_tf.numpy() / Beta_number

    
#     std_error = np.std(np.log(I3) - I_input[:, 2].T, axis=1, keepdims=True)
    
#     min_index = np.argmin(std_error)
#     I4 = np.column_stack((I_input[:, 0], I3[min_index, :], np.log(I3[min_index, :])))
#     diff_I = np.mean(I_input[:, 2] - I4[:, 2])
    
#     I4[:, 2] = I4[:, 2] + diff_I
#     I4[:, 1] = np.exp(I4[:, 2])
    
#     # 3.输出最终结果
#     best_swarm = swarm[min_index, :]
#     coeff_phi = np.exp(diff_I)
#     best_distri = np.column_stack((Beta_R[min_index, :], coeff_phi * Beta_Y[min_index, :]))
#     min_std_error = std_error[min_index]
    
#     return I4, best_distri, best_swarm, coeff_phi, std_error, min_std_error

# def SESDC_single(swarm, I_input):
#     Beta_number = 201
#     Beta_R = np.zeros((swarm.shape[0], Beta_number))
#     Beta_Y = np.zeros((swarm.shape[0], Beta_number))
#     # 遍历每个种群
#     for i in range(swarm.shape[0]):
#         Y1 = get_beta(0, swarm[i, 2], swarm[i, 0], swarm[i, 1], Beta_number)  
#         Beta_R[i, :], Beta_Y[i, :] = Y1[:, 0].T, Y1[:, 1].T
    
#     gpu_R, gpu_Y = cp.array(Beta_R)[:, :, np.newaxis], cp.array(Beta_Y)[:, :, np.newaxis]
#     gpu_V_phi = (4*cp.pi*gpu_R**3/3) * gpu_Y
    
#     count_number = 100
#     q = I_input[:, 0].reshape((1, 1, I_input.shape[0]))
#     n = np.arange(0, I_input.shape[0], count_number)
#     n = np.append(n, I_input.shape[0])
    
#     # 2.GPU计算散射强度
#     I2 = []
#     for i in range(n.shape[0]-1):
#         U1 = np.tile(q[0, 0, n[i]:n[i+1]], (gpu_Y.shape[0], gpu_Y.shape[1], 1))
#         U1 = cp.array(U1) * gpu_R
#         U1 = ( (3/(U1**3)) * (cp.sin(U1)-U1*cp.cos(U1)))**2 * gpu_V_phi
#         I1 = cp.sum(U1, axis=1)
#         I2.append(I1)
    
#     I3 = cp.asnumpy(cp.concatenate(I2, axis=1))/Beta_number  # 应该计算平均值
#     std_error = np.std(np.log(I3) - I_input[:, 2].T, axis=1, keepdims=True)
    
#     min_index = np.argmin(std_error)
#     I4 = np.column_stack((I_input[:, 0], I3[min_index,:], np.log(I3[min_index,:])))
#     diff_I = np.mean(I_input[:, 2] - I4[:, 2])
    
#     I4[:, 2] = I4[:, 2] + diff_I
#     I4[:, 1] = np.exp(I4[:, 2])
    
#     # 3.输出最终的结果
#     best_swarm = swarm[min_index, :]
#     coeff_phi = np.exp(diff_I)
#     best_distri = np.column_stack((Beta_R[min_index, :], coeff_phi * Beta_Y[min_index, :]))
#     min_std_error = std_error[min_index]
    
#     return I4, best_distri, best_swarm, coeff_phi, std_error, min_std_error


# %% 9. rotation of crystal lattice
def rotate_function_crystal(r_cal,rotate_angle):
    c,s = np.cos(rotate_angle),np.sin(rotate_angle)
    rotate_matrix_x = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    rotate_matrix_y = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    rotate_matrix_z = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    r_cal_rotate_x = np.dot(r_cal.T,rotate_matrix_x.T).T
    r_cal_rotate_y = np.dot(r_cal.T,rotate_matrix_y.T).T
    r_cal_rotate_z = np.dot(r_cal.T,rotate_matrix_z.T).T
    return r_cal_rotate_x, r_cal_rotate_y, r_cal_rotate_z


# %% 10. 计算向量A绕着向量B逆时针旋转
def rotateVector(A, B, alpha):
    """
    旋转矩阵A中的每个三维向量（每行一个向量）
    
    参数：
        A: 形状为(n, 3)的二维矩阵，n为向量数量（如2476525）
        B: 形状为(3,)的一维数组，旋转轴向量
        alpha: 旋转角度（度）
    
    返回：
        A_rotate: 旋转后归一化的向量矩阵（n, 3）
        A_rotate_with_norm: 旋转后保留原始模长的向量矩阵（n, 3）
    """
    alpha = alpha * np.pi / 180  # 角度转弧度
    
    # 1. 计算A中每个向量的模长（按行计算范数，保持维度为(n, 1)方便广播）
    norm_A = np.linalg.norm(A, axis= -1, keepdims=True)  # 形状：(n, 1)
    
    # 2. 归一化A（每个向量除以自身模长）和B（单个向量归一化）
    A_normalized = A / norm_A  # 形状：(n, 3)（每个向量归一化）
    B_normalized = B / np.linalg.norm(B)  # 形状：(3,)（B归一化）
    
    # 3. 计算每个A向量与B的点积（按行计算，结果为(n, 1)）
    dot_BA = np.sum(B_normalized * A_normalized, axis= -1, keepdims=True)  # 等价于点积，形状：(n, 1)
    
    # 4. 计算每个A向量与B的叉积（B×A，形状：(n, 3)）
    cross_BA = np.cross(B_normalized, A_normalized)  # 形状：(n, 3)（逐向量叉积）
    
    # 5. 旋转公式（罗德里格斯旋转公式），逐向量计算
    term1 = A_normalized * np.cos(alpha)  # 形状：(n, 3)
    term2 = cross_BA * np.sin(alpha)      # 形状：(n, 3)
    term3 = B_normalized * (dot_BA * (1 - np.cos(alpha)))  # 形状：(n, 3)（广播B到(n,3)）
    A_rotate = term1 + term2 + term3     # 形状：(n, 3)
    
    # 6. 归一化旋转后的向量（消除数值误差）
    A_rotate_norm = np.linalg.norm(A_rotate, axis= -1, keepdims=True)  # 旋转后每个向量的模长
    A_rotate_normalized = A_rotate / A_rotate_norm  # 归一化
    
    # 7. 恢复原始模长
    A_rotate_with_norm = norm_A * A_rotate_normalized  # 形状：(n, 3)
    
    return {"unit_vector": A_rotate_normalized, 
            "real_vector":A_rotate_with_norm,
            "x": A_rotate_with_norm[:, 0],
            "y": A_rotate_with_norm[:, 1],
            "z": A_rotate_with_norm[:, 2],}


# 10-2. 计算两个向量的夹角
def vector_delta_angle(v1, v2):
    """
    用numpy计算两个向量的夹角（单位：度）
    
    参数:
        v1: 第一个向量（列表、元组或numpy一维数组，元素为数值）
        v2: 第二个向量（同上）
    
    返回:
        float: 两个向量的夹角（度）
    
    异常:
        ValueError: 若向量非一维、维度不同、为空或为零向量时抛出
    """
    # 转换为numpy数组（支持列表/元组输入）
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    
    # 检查是否为一维向量
    if v1.ndim != 1 or v2.ndim != 1:
        raise ValueError("输入必须是一维向量")
    
    # 检查维度是否相同
    if v1.size != v2.size:
        raise ValueError("两个向量必须具有相同的维度")
    
    # 检查向量是否为空（长度为0）
    if v1.size == 0:
        raise ValueError("向量不能为空")
    
    # 计算点积
    dot_product = np.dot(v1, v2)
    
    # 计算模长（L2范数）
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    # 检查是否为零向量（模长为0）
    if norm1 == 0 or norm2 == 0:
        raise ValueError("向量不能为零向量（模长为0）")
    
    # 计算余弦值（钳位处理数值误差）
    cos_theta = dot_product / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 限制在[-1, 1]避免计算错误
    
    # 转换为角度并返回
    theta_rad = np.arccos(cos_theta)
    return theta_rad


# %% 12. 读取edf文件
def open_edf_save_tif():
    file_path = filedialog.askopenfilename(title='Select a EDF file', filetypes=[('EDF files', '*.edf')])
    edf_file = fabio.open(file_path)
    pathName, fileName = os.path.dirname(file_path), os.path.basename(file_path)
    baseName = os.path.splitext(fileName)[0]
    txtname = os.path.join(pathName,'header_info.txt')
    
    open(txtname, 'w').write(
        "键\t类型\t大小\t值\n" + '\n'.join(f"{k}\t{type(v).__name__}\t{len(str(v))}\t{v}" 
                                     for k, v in edf_file.header.items()))

    # 获取图像数据（numpy 数组）
    image_data = edf_file.data
    header = edf_file.header

    print("图像维度:", image_data.shape)
    print("数据类型:", image_data.dtype)

    image_data = np.nan_to_num(image_data, nan=0.0)
    filename = os.path.join(pathName,f"{baseName}.tif")
    tifffile.imwrite(filename, image_data)
    return header
    
    
# %% 13. 扇形积分
def rad_integrate(alpha, alpha_range, q, phi, pattern0, length_L):    
    t1=np.abs(phi-alpha); t1[t1>alpha_range] = 1e3
    I1=np.column_stack([ q[t1<alpha_range], pattern0[t1<alpha_range] ])
    I1 = I1[np.argsort(I1[:,0])]
    q0 = np.linspace(np.min(I1[:,0]), np.max(I1[:,0]), length_L)
    stat, bin_edges, _ = binned_statistic(I1[:, 0], I1[:, 1], statistic='mean', bins=q0)
    Im = np.column_stack([bin_edges[:-1], stat])
    Im[Im<0] = np.nan
    # Im[np.isnan(Im[:,1]), 1] = np.min(pattern0)
    return Im
    
   
# %% 14. 方位角积分
def azimuth_integrate(q_a, q_range, q, phi, pattern0):
    azimuthal_angle = np.linspace(0, 360, 361)
    t1 = np.abs(q-q_a); t1[t1>q_range] = 1e3; 
    I1 = np.column_stack([ phi[t1<q_range], pattern0[t1<q_range] ])
    I1 = I1[np.argsort(I1[:,0])] 
    stat, bin_edges, _ = binned_statistic(I1[:, 0], I1[:, 1], statistic='mean', bins=azimuthal_angle)
    In = np.column_stack([bin_edges[:-1], stat])
    In[In<0] = np.nan
    # In = In[ ~np.isnan(In[:,1]) ]
    return In


# %% 15. 二维极图
def TwoDpolar(q, phi, pattern0, ls_Y, ls_Z, pixel):
    calculation_number = 360
    chi_q_max = np.max(q)
    amount_of_pixel = int(np.floor(np.max( np.sqrt(ls_Y**2 + ls_Z**2)/pixel )))
    chi_q0 = np.linspace(0, chi_q_max, amount_of_pixel+1)
    
    chi_I_2D = [] 

    for i in range(calculation_number):
        # print(i)
        alpha = 1*i
        alpha_range=1; 
        t1=np.abs(phi-alpha); t1[t1>alpha_range]=1e4;
        I1=np.column_stack([ q[t1<alpha_range], pattern0[t1<alpha_range] ]);
        I1 = I1[ np.argsort(I1[:,0]) ] # 按照第一列排序
        stat, bin_edges, _ = binned_statistic(I1[:, 0], I1[:, 1], statistic='mean', bins=chi_q0)
        Im = np.column_stack([bin_edges[:-1], stat])
        Im[np.isnan(Im[:,1]), 1] = 0
        Im[Im[:, 1]<0, 1] = 0
        chi_I_2D.append(Im[:,1])

    chi_I_2D = np.array(chi_I_2D)
    chi_q0 = chi_q0[:-1]
    chi_phi0 = np.linspace(0, 360, calculation_number+1)[:-1]
    chi_q_2d, chi_phi_2d = np.meshgrid(chi_q0, chi_phi0)
    
    return chi_I_2D, chi_q_2d, chi_phi_2d


# %% 16. 二维图标定
def calibrate_Twod(Y, Z, 
                   Center_y, Center_z, 
                   pixel,
                   incident_angle, 
                   distance, 
                   lambda_Xray):
    v_I0 = [-1, 0, 0]; v_rotate_axis = [0, -1, 0]
    v_I_GIXRS = rotateVector(v_I0, v_rotate_axis, incident_angle)
    ls_Y = (Y-Center_y)*pixel; ls_Z = (Z-Center_z)*pixel; ls_X=-distance*np.ones_like(ls_Y);
    norm_detector = np.sqrt(ls_X**2 + ls_Y**2 + ls_Z**2)
    norm_YZ = np.sqrt(ls_Y**2 + ls_Z**2)
    phi = np.acos(ls_Y/norm_YZ)*180/np.pi
    vx = ls_X/norm_detector; vy = ls_Y/norm_detector; vz = ls_Z/norm_detector;
    sx = vx-v_I_GIXRS[0]; sy = vy-v_I_GIXRS[1]; sz = vz-v_I_GIXRS[2];
    qx=(2*np.pi/lambda_Xray)*sx; qy=(2*np.pi/lambda_Xray)*sy; qz=(2*np.pi/lambda_Xray)*sz; 
    q = np.sqrt(qx**2+qy**2+qz**2)
    phi[qz<0] = 360-phi[qz<0]; phi[np.isnan(phi)] = 0
    return q, phi, ls_Y, ls_Z, norm_YZ
    

# %% 17. 方位角自动剔除gap
def azimuth_cut_gap(A):  
    isnan = np.isnan(A[:, 1])
    gap_regions = np.where(np.diff(np.concatenate([[False], isnan, [False]])))[0].reshape(-1, 2)
    B = A.copy()  
    for start, end in gap_regions:  
        B[max(0, start-4):start, 1] = np.nan  # 前扩4点  
        B[end:min(len(B), end+4), 1] = np.nan  # 后扩4点  
    return B  


# %% 18. 方位角积分中的gap填充
def azimuth_gap_fit_fill(B, order):
    q = B[:, 0]  # 横坐标
    I = B[:, 1]  # 强度值
    C = B.copy() # 创建副本
    
    # 找出所有间断区
    nan_mask = np.isnan(I)
    nan_indices = np.where(nan_mask)[0]
    if len(nan_indices) == 0:
        return C
    
    # 获取连续间断区
    gaps = []
    start_idx = nan_indices[0]
    for i in range(1, len(nan_indices)):
        if nan_indices[i] != nan_indices[i-1] + 1:
            gaps.append((start_idx, nan_indices[i-1]))
            start_idx = nan_indices[i]
    gaps.append((start_idx, nan_indices[-1]))
    
    # 对每个间断区进行拟合填充
    for start, end in gaps:
        # 确定拟合区间 [q1-15, q2+15]
        q1 = max(0, start - 15)
        q2 = min(len(q)-1, end + 15)
        
        # 提取临时数据
        linshi = B[q1 : q2 + 1, :]
        x_temp = linshi[:, 0]
        y_temp = linshi[:, 1]
        
        # 提取非nan数据用于拟合
        non_nan_mask = ~np.isnan(y_temp)
        x_fit = x_temp[non_nan_mask]
        y_fit = y_temp[non_nan_mask]
        
        if len(x_fit) < 10:  # 数据点不足时跳过
            continue
        
        # 寻找最佳多项式次数 (1-9次)
        best_score = -np.inf
        # best_degree = 1
        best_model = None
        
        for degree in range(1, order):
            # 分割训练/测试集 (80/20)
            X_train, X_test, y_train, y_test = train_test_split(
                x_fit, y_fit, test_size=0.2, random_state=42
            )
            
            # 多项式拟合
            coeffs = np.polyfit(X_train, y_train, degree)
            model = np.poly1d(coeffs)
            
            # 评估模型 (R²分数)
            y_pred = model(X_test)
            score = r2_score(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                # best_degree = degree
                best_model = model
        
        # 用最佳模型填充间断区
        gap_mask = (q >= q[start]) & (q <= q[end])
        C[gap_mask, 1] = best_model(q[gap_mask])
    
    return C


# %% 19. 方位角积分的衍射峰位
def azimuth_find_peak_position(I_rad):
    # I_rad = I_fillgap.copy()
    q = I_rad[:, 0]; I = I_rad[:, 1]
    
    # 1. 数据预处理
    n_points = len(I)
    window_length = min(21, n_points // 2 * 2 + 1)  # 确保为奇数
    
    if window_length > 3:
        I_smooth = savgol_filter(I, window_length, 3)
    else:
        I_smooth = I  # 数据点过少时不滤波
    
    # 2. 计算噪声水平
    residual = I - I_smooth
    noise_level = np.std(residual) if np.std(residual) > 0 else 1e-6
    
    # 3. 设置自适应阈值
    min_height = noise_level * 3
    min_prominence = noise_level * 2
    min_distance = max(10, n_points // 100)  # 动态间距
    
    # 4. 峰值检测
    peaks, properties = find_peaks(
        I_smooth,
        height=min_height,
        prominence=min_prominence,
        distance=min_distance
    )
    
    # 5. 峰位提取与处理
    if len(peaks) > 0:
        # 按突出度排序 → 取前9个 → 按q值升序排列
        prominences = properties['prominences']
        sorted_indices = np.argsort(prominences)[::-1][:9]
        peak_q_values = q[peaks[sorted_indices]]
        sorted_peak_q = np.sort(peak_q_values)
    else:
        sorted_peak_q = np.nan
        
    return I_smooth, sorted_peak_q

# %% 20. 方位角积分的衍射峰范围
def azimuth_find_peak_boundaries(q, I_smooth, peak_q_value):
    """
    在衍射峰数据中寻找峰两侧的边界点
    
    参数:
    q : numpy.ndarray (N,)
        一维数组，表示q值
    I_smooth : numpy.ndarray (N,)
        一维数组，表示平滑后的强度值
    peak_q_value : float
        目标峰位的q值
    
    返回:
    P1 : float
        左侧边界点的q值
    P2 : float
        右侧边界点的q值
    """
    # 找到最接近峰位的索引位置
    peak_idx = np.argmin(np.abs(q - peak_q_value))
    
    # 初始化左右边界索引
    left_boundary_idx = peak_idx
    right_boundary_idx = peak_idx
    
    # 设置连续上升计数阈值
    consecutive_threshold = 6
    
    # 向左搜索 (递减方向)
    consecutive_rise = 0
    current_min = I_smooth[peak_idx]
    for i in range(peak_idx - 1, -1, -1):  # 从峰左侧开始向左遍历
        if I_smooth[i] < current_min:
            # 仍在下降趋势中
            current_min = I_smooth[i]
            left_boundary_idx = i
            consecutive_rise = 0  # 重置连续上升计数
        else:
            # 检测到上升
            consecutive_rise += 1
            if consecutive_rise >= consecutive_threshold:
                # 连续上升超过阈值，停止搜索
                break
            # 更新当前最小值（允许局部反弹）
            current_min = min(current_min, I_smooth[i])
    
    # 向右搜索 (递增方向)
    consecutive_rise = 0
    current_min = I_smooth[peak_idx]
    for i in range(peak_idx + 1, len(I_smooth)):
        if I_smooth[i] < current_min:
            # 仍在下降趋势中
            current_min = I_smooth[i]
            right_boundary_idx = i
            consecutive_rise = 0  # 重置连续上升计数
        else:
            # 检测到上升
            consecutive_rise += 1
            if consecutive_rise >= consecutive_threshold:
                # 连续上升超过阈值，停止搜索
                break
            # 更新当前最小值（允许局部反弹）
            current_min = min(current_min, I_smooth[i])
    
    # 返回边界点的q值
    P1 = q[left_boundary_idx]
    P2 = q[right_boundary_idx]
    
    return P1, P2


# %% 21. 方位角积分的 峰中心、半峰宽
def azimuth_peak_center_FWHM(I0):
    """
    计算衍射峰的峰中心(质心法)和FWHM
    
    参数:
    I0 : numpy.ndarray (N, 2)
        二维数组，第一列为横坐标，第二列为衍射强度
    
    返回:
    I1 : numpy.ndarray (N, 2)
        减去背底后的衍射强度数据
    peak_center : float
        峰中心位置(质心法)
    FWHM : float
        半高宽
    """
    # 1. 提取第一行和最后一行，构建2x2矩阵A
    A = np.array([I0[0, :], I0[-1, :]])
    
    # 2. 线性插值得到背底B
    interp_func = interpolate.interp1d(A[:, 0], A[:, 1], kind='linear', 
                                      fill_value='extrapolate')
    B = interp_func(I0[:, 0])
    
    # 3. 构建背底矩阵C
    C = np.column_stack((I0[:, 0], B))
    
    # 4. 创建I0的副本I1并减去背底
    I1 = I0.copy()
    I1[:, 1] = I1[:, 1] - C[:, 1]
    
    # 5. 计算峰中心(质心法)
    # 获取减去背底后的坐标和强度
    x = I1[:, 0]
    y = I1[:, 1]
    
    # 应用质心算法：Σ(x_i * y_i) / Σ(y_i)
    weighted_sum = np.sum(x * y)
    total_intensity = np.sum(y)
    
    # 避免除以0
    if total_intensity > 0:
        peak_center = weighted_sum / total_intensity
    else:
        # 如果强度总和为0，回退到最大值位置
        peak_idx = np.argmax(y)
        peak_center = x[peak_idx]
    
    # 6. 计算FWHM
    # 找到峰值高度
    max_intensity = np.max(y)
    half_max = max_intensity / 2
    
    # 找到左侧边界
    # 先找到峰值位置附近的点
    peak_idx = np.argmin(np.abs(x - peak_center))
    
    left_idx = peak_idx
    while left_idx > 0 and y[left_idx] > half_max:
        left_idx -= 1
    
    # 线性插值计算精确的左侧边界
    if left_idx < peak_idx and y[left_idx] <= half_max:
        x_left = x[left_idx] + (x[left_idx+1] - x[left_idx]) * \
                (half_max - y[left_idx]) / (y[left_idx+1] - y[left_idx])
    else:
        x_left = x[0]  # 如果找不到，使用第一个点
    
    # 找到右侧边界
    right_idx = peak_idx
    while right_idx < len(x)-1 and y[right_idx] > half_max:
        right_idx += 1
    
    # 线性插值计算精确的右侧边界
    if right_idx > peak_idx and y[right_idx] <= half_max:
        x_right = x[right_idx-1] + (x[right_idx] - x[right_idx-1]) * \
                 (half_max - y[right_idx-1]) / (y[right_idx] - y[right_idx-1])
    else:
        x_right = x[-1]  # 如果找不到，使用最后一个点
    
    # 计算FWHM
    FWHM = x_right - x_left
    max_intensity = np.max(I0[:, 1])
    peak_center = I0[np.argmax(I0[:, 1]), 0]
    
    return I1, peak_center, FWHM, max_intensity


# %% 22. 方位角积分峰范围的精细化确定
def azimuth_refine_peak_boundaries(q, intensity, peak_q_value, left_bound, right_bound):
    """
    优化函数：基于强度变化率重新精修衍射峰边界，排除干扰峰影响
    # q = I_azimuth_fillgap[:, 0]; intensity = I_smooth; peak_q_value = 315; left_bound = 255; right_bound = 374
    参数:
    q : numpy.ndarray - q值数组
    intensity : numpy.ndarray - 强度值
    peak_q_value : float - 目标峰位的q值
    left_bound, right_bound : float - 初始边界
    
    返回:
    new_left : float - 精修后的左侧边界
    new_right : float - 精修后的右侧边界
    """
    # 1. 创建数据矩阵并提取区间数据
    data_matrix = np.column_stack((q, intensity))
    mask = (q >= left_bound) & (q <= right_bound)
    segment_data = data_matrix[mask]
    
    # 2. 计算强度变化率（差分绝对值）
    intensity_values = segment_data[:, 1]
    smoothed = savgol_filter(intensity_values, window_length=5, polyorder=2)  # 先平滑
    
    diff_values = np.abs(np.diff(smoothed))
    diff_values = np.insert(diff_values, 0, 0)  # 保持长度一致
    segment_data = np.column_stack((segment_data, diff_values))
    
    # 3. 定位峰中心在区间中的位置
    peak_idx = np.argmin(np.abs(segment_data[:, 0] - peak_q_value))
    
    # 4. 精修左侧边界
    # 找到左侧最大变化率点
    left_section = segment_data[:peak_idx + 1]
    max_diff_idx_left = np.argmax(left_section[:, 2])
    left_section = np.delete(left_section, 1, axis=1)
    
    consecutive_threshold = 10
    
    # 从最大变化率点向左搜索
    current_min = left_section[max_diff_idx_left, 1]
    consecutive_rise = 0
    new_left_idx = max_diff_idx_left
    
    for i in range(max_diff_idx_left - 1, -1, -1):
        if left_section[i, 1] < current_min:
            current_min = left_section[i, 1]
            new_left_idx = i
            consecutive_rise = 0
        else:
            consecutive_rise += 1
            if consecutive_rise >= consecutive_threshold:
                break
        current_min = min(current_min, left_section[i, 1])
    
    new_left = left_section[new_left_idx, 0]
    
    # 5. 精修右侧边界
    # 找到右侧最大变化率点
    right_section = segment_data[peak_idx:]
    max_diff_idx_right = np.argmax(right_section[:, 2])
    right_section = np.delete(right_section, 1, axis=1)
    
    # 从最大变化率点向右搜索
    current_min = right_section[max_diff_idx_right, 1]
    consecutive_rise = 0
    new_right_idx = max_diff_idx_right
    
    for i in range(max_diff_idx_right + 1, len(right_section)):
        if right_section[i, 1] < current_min:
            current_min = right_section[i, 1]
            new_right_idx = i
            consecutive_rise = 0
        else:
            consecutive_rise += 1
            if consecutive_rise >= consecutive_threshold:
                break
        current_min = min(current_min, right_section[i, 1])
    
    new_right = right_section[new_right_idx, 0]
    
    mask = (q >= new_left) & (q <= new_right)
    baseI = intensity[mask]
    max_base = np.max(baseI) - np.min(baseI)
    base0 = np.abs(baseI[-1]-baseI[0]); coeff = base0/max_base
    if coeff > 0.4:
        new_left = left_bound; new_right = right_bound
    
    return new_left, new_right


# %% 23. 径向积分自动剔除gap
def radial_cut_gap(A):  
    isnan = np.isnan(A[:, 1])
    gap_regions = np.where(np.diff(np.concatenate([[False], isnan, [False]])))[0].reshape(-1, 2)
    B = A.copy()  
    for start, end in gap_regions:  
        B[max(0, start - 5):start, 1] = np.nan  # 前扩5点  
        B[end:min(len(B), end + 5), 1] = np.nan  # 后扩5点  
    return B  


# %% 24. 径向积分中的背底基线扣除
def radial_deduct_background(I):
    # I = I_azimuth_fillgap_0.copy()
    length_I = len(I)
    A = I[0, :]
    min_E_i = float('inf')  # 最小负值占比 (初始设为无穷大)
    best_B_i = None        # 最优B_i点
    best_D = None          # 最优D数组
    
    for idx in range(length_I - 1, 10, -1):  # range(start, stop, step)
        # 步骤2: 提取当前右侧点B_i
        B_i = I[idx, :]
        
        # 步骤3: 构建2×2矩阵C
        # C = [[A_x, A_y],
        #      [B_x, B_y]]
        C = np.array([A, B_i])
        
        # 步骤4: 线性插值处理
        # 获取插值的x坐标 (起点和终点)
        x_points = C[:, 0]
        
        # 获取插值的y坐标 (起点和终点)
        y_points = C[:, 1]
        
        # 在整个x范围(I[:,0])上进行线性插值
        # 插值公式: y = y0 + (y1 - y0) * (x - x0)/(x1 - x0)
        # 避免除零错误
        denominator = x_points[1] - x_points[0]
        if abs(denominator) < 1e-10:  # 处理x0≈x1的情况
            denominator = 1e-10
            
        # 计算插值结果
        slope = (y_points[1] - y_points[0]) / denominator
        interpolated = y_points[0] + slope * (I[:, 0] - x_points[0])
        
        # 将插值结果添加到矩阵I (临时创建扩展矩阵)
        I_temp = np.hstack((I, interpolated.reshape(-1, 1)))
        
        # 步骤5: 计算D数组和负值占比E_i
        D = I_temp[:, 1] - I_temp[:, 2]  # 原始N值 - 插值结果
        negative_count = np.sum(D < 0)
        E_i = negative_count / length_I  # 负值点占比
        
        # 步骤7: 更新最佳结果 (寻找最小E_i)
        if E_i < min_E_i:
            min_E_i = E_i
            best_B_i = B_i
            best_D = D.copy()
            best_idx = idx  # 记录最优点的行索引
    
    # 步骤8: 构建基线修正矩阵I_deduct_baseline
    I_deduct_baseline = np.column_stack((I[:, 0], best_D))
    
    # 步骤9: 截断矩阵 (保留到best_B_i对应行)
    # 注意: 行索引从0开始，因此保留行数为best_idx+1
    I_result = I[: best_idx + 1, :]
    
    # 步骤10: 返回最终结果
    return I_result


# %% 25. 径向积分中的最佳多项式拟合
def best_fit_order(C, order):
    # import pandas as pd
    # df = pd.read_clipboard(header=None, sep='\t'); C = df.values;  
    
    C = C[~np.isnan(C[:, 1]), :]
    x, y = C[:, 0], C[:, 1]
    
    # 3. 遍历拟合次数
    E = []; 
    for k in range(1, order + 1):
        try:
            # 多项式拟合
            coeffs = np.polyfit(x, y, k)
            y_pred = np.polyval(coeffs, x)
            # 计算R²
            SS_res = np.sum((y - y_pred)**2)
            SS_tot = np.sum((y - np.mean(y))**2)
            R2 = 1 - SS_res / (SS_tot + 1e-10)
            value_1 = np.mean(np.abs(y - y_pred))
            E.append([k, R2, value_1])
            
        except:
            continue  # 跳过失败的高次拟合
    
    # 4. 按R²降序排序
    E = np.array(E)
    if len(E) > 0:
        E = E[E[:, 1].argsort()[::-1]]  # 降序索引
    return E


# %% 26. 找到连续的nan值区域
def find_continuous_nan_regions(data):
    """
    查找连续NaN区域并返回起始和终止点
    
    参数:
    data -- 输入数据数组 (1D)
    
    返回:
    regions -- 连续NaN区域的起始和终止点数组 (N×2)
    """
    # 标记NaN位置
    is_nan = np.isnan(data[:, 1])
    
    # 在数组前后添加False，确保能检测到边界
    padded = np.concatenate(([False], is_nan, [False]))
    
    # 找到状态变化点
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]   # False -> True (起始点)
    ends = np.where(changes == -1)[0] - 1  # True -> False (终止点)
    A = np.column_stack((starts, ends))
    # 组合结果
    return A


# %% 27. 绘制和调整峰位
def plot_change_peaks(Im, peaks):
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)  # 增加底部空间以容纳更多按钮
    
    # 绘制原始数据 - 使用单对数坐标
    ax.semilogy(Im[:, 0], Im[:, 1], 'b-', label='X-ray Diffraction')
    ax.set_xlabel('Pixel (mm)')
    ax.set_ylabel('Intensity')
    ax.set_title('Draggable Peak Positions')
    ax.grid(True)
    
    # 存储垂直线对象的列表
    vlines = []
    
    # 初始绘制所有峰位线
    for peak in peaks:
        vline = ax.axvline(x=peak, color='r', alpha=0.7, picker=5)
        vlines.append(vline)
    
    # 存储当前拖动的线索引
    current_line = None
    # 编辑状态标志
    editing_mode = False
    
    # 鼠标按下事件处理
    def on_pick(event):
        nonlocal current_line
        if not editing_mode or event.artist not in vlines:
            return
        current_line = vlines.index(event.artist)
        fig.canvas.draw()
    
    # 鼠标移动事件处理
    def on_motion(event):
        nonlocal current_line
        if not editing_mode or current_line is None or event.inaxes != ax or event.xdata is None:
            return
        
        # 更新垂直线位置
        new_x = event.xdata
        vlines[current_line].set_xdata([new_x, new_x])
        
        # 更新peaks数组
        peaks[current_line] = new_x
        
        fig.canvas.draw()
    
    # 鼠标释放事件处理
    def on_release(event):
        nonlocal current_line
        current_line = None
    
    # 开始编辑按钮回调函数
    def start_editing(event):
        nonlocal editing_mode
        editing_mode = True
        btn_start_edit.color = 'lightgreen'  # 改变按钮颜色表示激活状态
        btn_end_edit.color = '0.85'  # 恢复结束按钮颜色
        btn_save.color = '0.85'  # 保存按钮不可用状态
        ax.texts[0].set_text("Editing mode: ON - You can now drag the red lines")
        ax.texts[0].set_bbox(dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))
        fig.canvas.draw()
        print("Editing mode activated")
    
    # 结束编辑按钮回调函数
    def end_editing(event):
        nonlocal editing_mode
        editing_mode = False
        btn_start_edit.color = '0.85'  # 恢复开始按钮颜色
        btn_end_edit.color = 'lightcoral'  # 改变结束按钮颜色表示状态
        btn_save.color = 'lightgoldenrodyellow'  # 激活保存按钮
        ax.texts[0].set_text("Editing mode: OFF - Click 'Adjust Peaks' to enable dragging")
        ax.texts[0].set_bbox(dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
        fig.canvas.draw()
        print("Editing mode deactivated")
    
    # 保存按钮回调函数
    def save_peaks(event):
        if editing_mode:
            # 如果仍在编辑模式，不允许保存
            ax.texts[0].set_text("Please click 'Finish Adjustment' before saving")
            ax.texts[0].set_bbox(dict(boxstyle="round", facecolor="lightcoral", alpha=0.8))
            fig.canvas.draw()
            return
        
        print("Updated peaks:", peaks)
        # 在图表上添加保存成功的文本提示
        if len(ax.texts) > 1:
            ax.texts[1].remove()
        ax.text(0.02, 0.98, f"Peaks saved! {len(peaks)} peaks updated.", 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        fig.canvas.draw()
        
        # 这里可以添加保存到文件的代码
        # np.savetxt('updated_peaks.txt', peaks)
    
    # 连接事件处理器
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    
    # 添加操作说明文本
    ax.text(0.02, 0.02, "Editing mode: OFF - Click 'Adjust Peaks' to enable dragging", 
            transform=ax.transAxes, fontsize=10, style='italic',
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
    
    # 添加"微调峰位"按钮
    ax_start_edit = plt.axes([0.1, 0.02, 0.2, 0.06])
    btn_start_edit = Button(ax_start_edit, 'Adjust Peaks', color='0.85')
    btn_start_edit.on_clicked(start_editing)
    
    # 添加"微调完毕"按钮
    ax_end_edit = plt.axes([0.4, 0.02, 0.2, 0.06])
    btn_end_edit = Button(ax_end_edit, 'Finish Adjustment', color='0.85')
    btn_end_edit.on_clicked(end_editing)
    
    # 添加保存按钮
    ax_save = plt.axes([0.7, 0.02, 0.2, 0.06])
    btn_save = Button(ax_save, 'Save Peaks', color='0.85')
    btn_save.on_clicked(save_peaks)
    
    plt.show()
    return peaks



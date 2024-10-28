from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from typing import Tuple, Optional
import cv2
import os
from natsort import natsorted

# 使用归一化相机矩阵
camera_matrix = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

from enum import Enum, auto
from scipy.linalg import block_diag

class MotionState(Enum):
    """运动状态枚举"""
    FORWARD = auto()    # 前进
    BACKWARD = auto()   # 后退
    TURN_LEFT = auto()  # 左转
    TURN_RIGHT = auto() # 右转
    IDLE = auto()       # 静止

class StateMachineKalmanFilter:
    """
    状态机卡尔曼滤波器
    状态向量: [x, y, θ, dx/dt, dy/dt, ω]
    测量值: [dx/dt, dy/dt, ω]
    """
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.dim_state = 6
        self.dim_measure = 3
        
        # 初始化卡尔曼滤波器
        self.kf = KalmanFilter(dim_x=self.dim_state, dim_z=self.dim_measure)
        
        # 基础状态转移矩阵
        self.base_F = np.array([
            [1, 0, 0, dt, 0, 0],  # x = x + dx/dt * dt
            [0, 1, 0, 0, dt, 0],  # y = y + dy/dt * dt
            [0, 0, 1, 0, 0, dt],  # θ = θ + ω * dt
            [0, 0, 0, 1, 0, 0],   # dx/dt = dx/dt
            [0, 0, 0, 0, 1, 0],   # dy/dt = dy/dt
            [0, 0, 0, 0, 0, 1]    # ω = ω
        ])
        
        # 测量矩阵保持不变
        self.kf.H = np.array([
            [0, 0, 0, 1, 0, 0],  # 测量 dx/dt
            [0, 0, 0, 0, 1, 0],  # 测量 dy/dt
            [0, 0, 0, 0, 0, 1]   # 测量 ω
        ])
        
        # 初始状态
        self.kf.x = np.zeros(6)
        self.current_state = MotionState.IDLE
        
        # 为不同状态设置参数
        self.setup_motion_parameters()
        
    def setup_motion_parameters(self):
        """为不同运动状态设置参数"""
        # 过程噪声参数
        self.Q_params = {
            MotionState.FORWARD: {
                'pos': 0.1,      # 位置不确定性
                'angle': 0.01,   # 角度不确定性
                'vel': 0.1,      # 速度不确定性
                'omega': 0.01    # 角速度不确定性
            },
            MotionState.BACKWARD: {
                'pos': 0.1,
                'angle': 0.01,
                'vel': 0.1,
                'omega': 0.01
            },
            MotionState.TURN_LEFT: {
                'pos': 0.01,     # 转弯时位置变化小
                'angle': 0.1,    # 角度变化大
                'vel': 0.01,     # 速度变化小
                'omega': 0.1     # 角速度变化大
            },
            MotionState.TURN_RIGHT: {
                'pos': 0.01,
                'angle': 0.1,
                'vel': 0.01,
                'omega': 0.1
            },
            MotionState.IDLE: {
                'pos': 0.01,
                'angle': 0.01,
                'vel': 0.01,
                'omega': 0.01
            }
        }
        
        # 测量噪声参数
        self.R_params = {
            MotionState.FORWARD: np.diag([0.1, 0.1, 0.01]),    # 前进时速度测量较准
            MotionState.BACKWARD: np.diag([0.1, 0.1, 0.01]),   # 后退时速度测量较准
            MotionState.TURN_LEFT: np.diag([0.2, 0.2, 0.1]),   # 转弯时速度测量不太准
            MotionState.TURN_RIGHT: np.diag([0.2, 0.2, 0.1]),
            MotionState.IDLE: np.diag([0.3, 0.3, 0.3])         # 静止时测量很不准
        }
        
        # 运动约束
        self.motion_constraints = {
            MotionState.FORWARD: {
                'vel_ratio': 0.1,    # 横向速度与前向速度的比值上限
                'omega_max': 0.1     # 最大角速度
            },
            MotionState.BACKWARD: {
                'vel_ratio': 0.1,
                'omega_max': 0.1
            },
            MotionState.TURN_LEFT: {
                'vel_max': 0.1,      # 最大线速度
                'omega_min': 0.1     # 最小角速度
            },
            MotionState.TURN_RIGHT: {
                'vel_max': 0.1,
                'omega_min': 0.1
            }
        }
        
    def update_motion_state(self, velocity_measurement: np.ndarray) -> MotionState:
        """
        根据速度测量更新运动状态
        velocity_measurement: [dx/dt, dy/dt, ω]
        """
        dx_dt, dy_dt, omega = velocity_measurement
        
        # 计算速度大小和方向
        vel_magnitude = np.sqrt(dx_dt**2 + dy_dt**2)
        
        # 状态判断阈值
        VEL_THRESHOLD = 0.1
        OMEGA_THRESHOLD = 0.1
        
        if abs(omega) > OMEGA_THRESHOLD:
            # 转弯状态
            return MotionState.TURN_LEFT if omega > 0 else MotionState.TURN_RIGHT
        elif vel_magnitude > VEL_THRESHOLD:
            # 前进或后退状态
            # 假设dx_dt为前向速度
            return MotionState.FORWARD if dx_dt > 0 else MotionState.BACKWARD
        else:
            return MotionState.IDLE
            
    def apply_motion_constraints(self, state: np.ndarray, motion_state: MotionState) -> np.ndarray:
        """应用运动约束"""
        constrained_state = state.copy()
        
        if motion_state in [MotionState.FORWARD, MotionState.BACKWARD]:
            # 限制横向速度
            max_side_vel = abs(state[3]) * self.motion_constraints[motion_state]['vel_ratio']
            constrained_state[4] = np.clip(state[4], -max_side_vel, max_side_vel)
            
            # 限制角速度
            omega_max = self.motion_constraints[motion_state]['omega_max']
            constrained_state[5] = np.clip(state[5], -omega_max, omega_max)
            
        elif motion_state in [MotionState.TURN_LEFT, MotionState.TURN_RIGHT]:
            # 限制线速度
            vel_max = self.motion_constraints[motion_state]['vel_max']
            constrained_state[3] = np.clip(state[3], -vel_max, vel_max)
            constrained_state[4] = np.clip(state[4], -vel_max, vel_max)
            
            # 确保最小角速度
            omega_min = self.motion_constraints[motion_state]['omega_min']
            if motion_state == MotionState.TURN_LEFT:
                constrained_state[5] = max(state[5], omega_min)
            else:
                constrained_state[5] = min(state[5], -omega_min)
                
        return constrained_state
        
    def update_filter_parameters(self, motion_state: MotionState):
        """更新滤波器参数"""
        # 更新过程噪声协方差
        Q_param = self.Q_params[motion_state]
        pos_var = np.diag([Q_param['pos']] * 2 + [Q_param['angle']])
        vel_var = np.diag([Q_param['vel']] * 2 + [Q_param['omega']])
        self.kf.Q = block_diag(pos_var, vel_var)
        
        # 更新测量噪声协方差
        self.kf.R = self.R_params[motion_state]
        
    def predict(self) -> np.ndarray:
        """预测下一状态"""
        self.kf.predict()
        # 应用运动约束
        self.kf.x = self.apply_motion_constraints(self.kf.x, self.current_state)
        return self.get_state()
        
    def update(self, velocity_measurement: Optional[np.ndarray] = None):
        """使用速度测量更新状态"""
        if velocity_measurement is not None:
            # 更新运动状态
            new_state = self.update_motion_state(velocity_measurement)
            if new_state != self.current_state:
                self.current_state = new_state
                self.update_filter_parameters(new_state)
            
            # 创建测量掩码
            valid_measurements = ~np.isnan(velocity_measurement)
            
            if np.any(valid_measurements):
                # 只使用有效的测量值
                H_valid = self.kf.H[valid_measurements]
                R_valid = self.kf.R[valid_measurements][:, valid_measurements]
                z_valid = velocity_measurement[valid_measurements]
                
                # 更新状态
                self.kf.update(z_valid, H_valid, R_valid)
                
                # 应用运动约束
                self.kf.x = self.apply_motion_constraints(self.kf.x, self.current_state)
                
                # 标准化角度
                self.kf.x[2] = self.normalize_angle(self.kf.x[2])
                
    def get_state(self) -> Tuple[float, float, float]:
        """获取当前状态估计 (x, y, θ)"""
        return tuple(self.kf.x[:3])
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        将角度标准化到 [-π, π] 范围内
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
class FeatureTracker:
    """
    FeatureTracker类用于检测和跟踪图像中的特征点。
    属性:
        max_corners (int): 最大特征点数。
        quality_level (float): FAST角点质量阈值。
        min_distance (int): 特征点最小距离。
        block_size (int): 角点检测块大小。
        win_size (tuple): 金字塔窗口大小。
        max_level (int): 金字塔层数。
        criteria (tuple): 终止条件。
    方法:
        detect_features(frame):
            检测新的特征点。
            参数:
                frame (numpy.ndarray): 输入图像帧。
            返回:
                numpy.ndarray: 检测到的特征点。
        track_features(old_frame, new_frame, old_points):
            使用光流法跟踪特征点。
            参数:
                old_frame (numpy.ndarray): 前一帧图像。
                new_frame (numpy.ndarray): 当前帧图像。
                old_points (numpy.ndarray): 前一帧中的特征点。
            返回:
                tuple: 包含新特征点、状态和误差的元组。
    """
    def __init__(self):
        self.max_corners = 200      # 最大特征点数
        self.quality_level = 0.2    # FAST角点质量阈值
        self.min_distance = 15       # 特征点最小距离
        self.block_size = 9         # 角点检测块大小
        
        # 光流参数
        self.win_size = (21,21)     # 金字塔窗口大小
        self.max_level = 3          # 金字塔层数
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    def detect_features(self, frame):
        """
        检测新的特征点
        参数:
            frame (numpy.ndarray): 输入图像帧。
        返回:
            numpy.ndarray: 检测到的特征点。
        """
        return cv2.goodFeaturesToTrack(frame, maxCorners=self.max_corners,
                                     qualityLevel=self.quality_level,
                                     minDistance=self.min_distance,
                                     blockSize=self.block_size)

    def track_features(self, old_frame, new_frame, old_points):
        """
        使用光流法跟踪特征点
        参数:
                old_frame (numpy.ndarray): 前一帧图像。
                new_frame (numpy.ndarray): 当前帧图像。
                old_points (numpy.ndarray): 前一帧中的特征点。
        返回:
            tuple: 包含新特征点、状态和误差的元组。
        """
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            old_frame, new_frame, old_points, None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria)
        return new_points, status, error
    
class VelocityMotionEstimator:
    def __init__(self):
        self.min_inliers = 8
        self.ransac_threshold = 1.0
        self.dt = 1.0  # 时间间隔，需要根据实际帧率调整
        
    def estimate_velocity(self, points1, points2):
        """
        估计相对运动，直接返回速度和角速度
        """
        # 估计基础矩阵
        F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC,
                                        self.ransac_threshold)
        if F is None:
            return None, None, None, None
            
        # 从基础矩阵恢复相对运动
        E = camera_matrix.T @ F @ camera_matrix
        _, R, t, _ = cv2.recoverPose(E, points1, points2, camera_matrix)
        
        # 提取角速度（绕z轴的旋转）
        omega = np.arctan2(R[1,0], R[0,0]) / self.dt
        
        # 提取线速度（注意要除以时间间隔得到速度）
        dx_dt = t[0,0] / self.dt
        dy_dt = t[1,0] / self.dt
        
        return dx_dt, dy_dt, omega, mask
    
class VisualOdometry:
    def __init__(self):
        self.feature_tracker = FeatureTracker()
        self.motion_estimator = VelocityMotionEstimator()
        self.kalman_filter = StateMachineKalmanFilter()
        self.trajectory = []
        
        self.prev_frame = None
        self.prev_points = None
        
    def process_frame(self, frame):
        """处理每一帧"""
        if self.prev_frame is None:
            # 第一帧
            self.prev_frame = frame
            self.prev_points = self.feature_tracker.detect_features(frame)
            return
            
        # 跟踪特征点
        curr_points, status, _ = self.feature_tracker.track_features(
            self.prev_frame, frame, self.prev_points)
            
        # 估计运动
        dx, dy, d_theta, mask = self.motion_estimator.estimate_velocity(
            self.prev_points, curr_points)
            
        # 卡尔曼滤波更新
        measurement = np.array([dx, dy, d_theta])
        self.kalman_filter.predict()
        self.kalman_filter.update(measurement)
        
        # 记录轨迹
        filtered_state = self.kalman_filter.get_state()
        self.trajectory.append(*filtered_state)
        
        # 更新特征点
        if len(curr_points) < self.feature_tracker.max_corners // 2:
            self.prev_points = self.feature_tracker.detect_features(frame)
        else:
            self.prev_points = curr_points[status == 1]
        
        self.prev_frame = frame

class VisualOdometryTest:
    def __init__(self):
        self.feature_tracker = FeatureTracker()
        self.motion_estimator = VelocityMotionEstimator()
        self.kalman_filter = StateMachineKalmanFilter()
        self.trajectory = []
        
        self.prev_frame = None
        self.prev_points = None
        
    def process_frame(self, frame):
        """处理每一帧"""
        if self.prev_frame is None:
            # 第一帧
            self.prev_frame = frame
            self.prev_points = self.feature_tracker.detect_features(frame)
            return
            
        # 跟踪特征点
        curr_points, status, _ = self.feature_tracker.track_features(
            self.prev_frame, frame, self.prev_points)
            
        # 估计运动
        if len(curr_points) < self.motion_estimator.min_inliers:
            print('Not enough inliers')
            return None, None, None
        dx, dy, omega, mask = self.motion_estimator.estimate_velocity(
            self.prev_points, curr_points)
        
        print(f'dx: {dx}, dy: {dy}, omega: {omega}')
        side_by_side = np.hstack((self.prev_frame, frame))
        cv2.imshow('Frame', side_by_side)
        cv2.waitKey(80)
        
        if len(curr_points) < self.feature_tracker.max_corners // 2:
            self.prev_points = self.feature_tracker.detect_features(frame)
        else:
            self.prev_points = curr_points[status == 1]
        self.prev_frame = frame

        return dx, dy, omega
    
# 测试代码
if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'images')
    imgs = natsorted(os.listdir(data_path))[120:140]
    vo = VisualOdometryTest()
    for img in imgs:
        frame = cv2.imread(os.path.join(data_path, img))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vo.process_frame(frame)
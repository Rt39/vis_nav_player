from dataclasses import dataclass
import vis_nav_game
from enum import Enum
from typing import Dict, List, Tuple
import cv2
import os
import pickle
import numpy as np
import networkx as nx
from natsort import natsorted
import pygame
from tqdm import tqdm
from player import KeyboardPlayerPyGame

from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

data_path = os.path.join(os.path.dirname(__file__), 'data', 'images_subsample')

class ImageFeatureExtractor:
    """
    ImageFeatureExtractor 是一个使用 SIFT 提取图像特征并使用 VLAD（局部聚合描述符向量）对其进行编码的类。
    属性:
        sift (cv2.SIFT): SIFT 特征检测器的实例。
        n_clusters (int): 用于 KMeans 聚类的聚类数量。
        codebook (KMeans): 由 KMeans 聚类生成的码本。
    方法:
        __init__(n_clusters=128):
            使用指定的聚类数量初始化 ImageFeatureExtractor。
        extract_sift(img):
            从单个图像中提取 SIFT 特征。
        extract_sift_batch(data_path):
            从指定目录中的所有图像中提取 SIFT 特征。
        train_codebook(descriptors):
            使用提供的描述符通过 KMeans 聚类训练 VLAD 码本。
        compute_vlad_encoding(descriptors, centroids):
            计算给定描述符和质心的 VLAD 编码。
        normalize_vlad(vlad_vector):
            使用幂归一化和 L2 归一化对 VLAD 向量进行归一化。
        compute_image_descriptor(img):
            计算单个图像的 VLAD 描述符。
    """
    def __init__(self, n_clusters=128):
        self.sift = cv2.SIFT_create()
        self.n_clusters = n_clusters
        self.codebook = None
    
    def extract_sift(self, img):
        """单个图像的SIFT特征提取"""
        _, descriptors = self.sift.detectAndCompute(img, None)
        return descriptors
    
    def extract_sift_batch(self, images):
        """批量提取SIFT特征
        Args:
            images: List[np.ndarray] 图像列表
        """
        all_descriptors = []
        for img in tqdm(images, desc="Extracting SIFT features"):
            descriptors = self.extract_sift(img)
            all_descriptors.extend(descriptors)
        return np.asarray(all_descriptors)
    
    def train_codebook(self, descriptors):
        """训练VLAD编码的码本"""
        self.codebook = KMeans(
            n_clusters=self.n_clusters, 
            init='k-means++', 
            n_init=5, 
            verbose=0
        ).fit(descriptors)
        return self.codebook
    
    def compute_vlad_encoding(self, descriptors, centroids):
        """计算VLAD编码"""
        # 预测每个描述符属于哪个聚类
        pred_labels = self.codebook.predict(descriptors)
        
        # 初始化VLAD特征向量
        k = self.codebook.n_clusters
        vlad_vector = np.zeros([k, descriptors.shape[1]])
        
        # 计算每个聚类的残差和
        for i in range(k):
            if np.sum(pred_labels == i) > 0:
                vlad_vector[i] = np.sum(
                    descriptors[pred_labels==i, :] - centroids[i], 
                    axis=0
                )
                
        return vlad_vector.flatten()
    
    def normalize_vlad(self, vlad_vector):
        """VLAD向量的归一化"""
        # Power归一化
        vlad_vector = np.sign(vlad_vector) * np.sqrt(np.abs(vlad_vector))
        # L2归一化
        vlad_vector = vlad_vector / np.linalg.norm(vlad_vector)
        return vlad_vector
    
    def compute_image_descriptor(self, img):
        """计算单个图像的VLAD描述子"""
        if self.codebook is None:
            raise ValueError("Codebook not trained! Call train_codebook first.")
            
        # 1. 提取SIFT特征
        descriptors = self.extract_sift(img)
        
        # 2. 计算VLAD编码
        vlad_vector = self.compute_vlad_encoding(
            descriptors, 
            self.codebook.cluster_centers_
        )
        
        # 3. 归一化
        normalized_vlad = self.normalize_vlad(vlad_vector)
        
        return normalized_vlad

class ActionType(Enum):
    """Possible actions in the maze"""
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    LEFT_TURN = "LEFT_TURN"
    RIGHT_TURN = "RIGHT_TURN"
    UNKNOWN = "UNKNOWN"

@dataclass
class Node:
    """Represents a location in the maze"""
    id: int
    features: np.ndarray  # VLAD features of this location
    image: np.ndarray    # Representative image of this location
    neighbors: Dict[ActionType, int] = None  # Mapping of action to neighbor node ids
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = {}

class MazeMapper:
    def __init__(self, feature_params: dict = None):
        # Initialize basic structures
        self.nodes: Dict[int, Node] = {}
        self.graph = nx.Graph()
        
        # Initialize feature extractor
        if feature_params is None:
            feature_params = {'n_clusters': 128}
        self.feature_extractor = ImageFeatureExtractor(**feature_params)
        
        # Initialize similarity search structure
        self.feature_tree = None
        self.similarity_threshold = 0.85  # Threshold for considering locations similar
        
    def build_maze_map(self, exploration_images: List[np.ndarray]) -> None:
        """Build maze map from exploration data"""
        print("Building maze map...")
        
        # 1. Extract SIFT features from all images and train codebook
        print("Training feature extractor...")
        all_descriptors = self.feature_extractor.extract_sift_batch(exploration_images)
        self.feature_extractor.train_codebook(all_descriptors)
        
        # 2. Process each image and create initial nodes
        print("Creating initial nodes...")
        features_list = []
        for idx, img in enumerate(tqdm(exploration_images, desc="Processing images")):
            # Compute VLAD features
            features = self.feature_extractor.compute_image_descriptor(img)
            
            # Create new node
            node = Node(id=idx, features=features, image=img)
            self.nodes[idx] = node
            features_list.append(features)
            
        # 3. Build feature tree for similarity search
        self.feature_tree = BallTree(np.array(features_list))
        
        # 4. Connect similar locations
        self._connect_similar_locations()
        
        # 5. Analyze sequential connections
        self._analyze_sequential_connections(exploration_images)
        
        print("Maze map building completed.")

    def _connect_similar_locations(self) -> None:
        """Connect nodes that represent similar locations"""
        print("Connecting similar locations...")
        
        features_array = np.array([node.features for node in self.nodes.values()])
        
        # Find similar locations using BallTree
        distances, indices = self.feature_tree.query(features_array, k=5)
        
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            for dist, idx in zip(dists[1:], idxs[1:]):  # Skip first (self)
                if dist < self.similarity_threshold:
                    self.graph.add_edge(i, int(idx))
    
    def _analyze_sequential_connections(self, exploration_images: List[np.ndarray]) -> None:
        """Analyze connections between sequential frames"""
        print("Analyzing sequential connections...")
        
        for i in range(len(exploration_images) - 1):
            curr_node = self.nodes[i]
            next_node = self.nodes[i + 1]
            
            # Determine action type based on image analysis
            action = self._determine_action(curr_node.image, next_node.image)
            
            # Add connection to graph
            self.graph.add_edge(curr_node.id, next_node.id, action=action)
            
            # Update node neighbors
            curr_node.neighbors[action] = next_node.id
            next_node.neighbors[self._get_reverse_action(action)] = curr_node.id
    
    def _determine_action(self, img1: np.ndarray, img2: np.ndarray) -> ActionType:
        """Determine the action between two sequential frames"""
        # TODO: Implement more sophisticated action detection
        # Currently returns UNKNOWN as placeholder
        return ActionType.UNKNOWN
    
    def _get_reverse_action(self, action: ActionType) -> ActionType:
        """Get the reverse of an action"""
        action_pairs = {
            ActionType.FORWARD: ActionType.BACKWARD,
            ActionType.BACKWARD: ActionType.FORWARD,
            ActionType.LEFT_TURN: ActionType.RIGHT_TURN,
            ActionType.RIGHT_TURN: ActionType.LEFT_TURN,
            ActionType.UNKNOWN: ActionType.UNKNOWN
        }
        return action_pairs[action]
    
    def find_similar_location(self, query_image: np.ndarray, k: int = 1) -> List[Tuple[int, float]]:
        """Find k most similar locations to query image"""
        # Extract features from query image
        query_features = self.feature_extractor.compute_image_descriptor(query_image)
        
        # Find nearest neighbors
        distances, indices = self.feature_tree.query(query_features.reshape(1, -1), k=k)
        
        # Return list of (node_id, similarity_score) pairs
        return [(int(idx), 1 - dist) for dist, idx in zip(distances[0], indices[0])]
    
    def get_node_neighbors(self, node_id: int) -> Dict[ActionType, int]:
        """Get neighbors of a node"""
        return self.nodes[node_id].neighbors
    
    def visualize_map(self, output_path: str = "maze_map.png") -> None:
        """Visualize the maze map"""
        # TODO: Implement visualization
        # Could use networkx's drawing functions or custom visualization
        pass

class SolutionPlayer(KeyboardPlayerPyGame):
    def __init__(self):
        super(KeyboardPlayerPyGame, self).__init__()

        self.count = 0  # 保存图像的计数

        # 新增的组件
        self.mapper = MazeMapper()
        self.feature_extractor = ImageFeatureExtractor()
        self.feature_database = []

    # 重写pre_navigation方法
    # Override
    def pre_navigation(self):
        pass    # TODO: 在导航阶段之前执行的计算

    # 重写see方法
    # Override
    def see(self, fpv):
        """
        Set the first-person view input
        """

        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        # This allows subsequent rendering of the first-person view image onto the pygame screen
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == vis_nav_game.Phase.EXPLORATION:
                # TODO: 在实际应用中，探索数据会被给出，这里我们自行探索数据

                # Get full absolute save path
                save_dir_full = os.path.join(os.path.dirname(__file__), data_path)
                save_path = save_dir_full + str(self.count) + ".jpg"
                # Save current FPV
                cv2.imwrite(save_path, fpv)

                # Get VLAD embedding for current FPV and add it to the database
                VLAD = self.feature_extractor.compute_image_descriptor(fpv)
                self.feature_database.append(VLAD)
                self.count += 1
            # If in navigation stage
            elif self._state[1] == vis_nav_game.Phase.NAVIGATION:
                # TODO: could you do something else, something smarter than simply getting the image closest to the current FPV?
                
                # Key the state of the keys
                keys = pygame.key.get_pressed()
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q]:
                    pass    # TODO: 实现导航功能

        # Display the first-person view image on the pygame screen
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

    def _load_exploration_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """加载探索数据
        Returns:
            images: List[np.ndarray] 图像列表
            image_ids: List[int] 图像ID列表
        """
        image_files = natsorted([x for x in os.listdir(data_path) if x.endswith('.jpg')])
        images = []
        image_ids = []
        
        for img_file in image_files:
            img = cv2.imread(os.path.join(data_path, img_file))
            if img is not None:
                images.append(img)
                image_ids.append(int(img_file.split('.')[0]))
                
        return images, image_ids
    

if __name__ == "__main__":
    os.makedirs(data_path, exist_ok=True)
    player = SolutionPlayer()
    vis_nav_game.play(player)
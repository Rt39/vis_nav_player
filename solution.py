import cv2
import os
import pickle
import numpy as np
from natsort import natsorted
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
    
class ImageDatabase:
    """
    ImageDatabase类用于存储图像特征并进行快速近邻搜索。
    方法:
        __init__():
            初始化ImageDatabase对象，创建一个空的特征列表和一个空的搜索树。
        add_feature(feature):
            添加一个特征到数据库。
        build_search_tree(leaf_size=64):
            构建搜索树以便进行快速近邻搜索。
        find_nearest(query_feature, k=1):
            找到最相似的k个特征的索引。
    """
    def __init__(self):
        self.features = []  # 存储所有图像的特征
        self.tree = None    # 用于快速近邻搜索
        
    def add_feature(self, feature):
        """
        添加一个特征到数据库

        参数:
        feature: 要添加到数据库的特征
        """
        self.features.append(feature)
        
    def build_search_tree(self, leaf_size=64):
        """
        构建搜索树

        参数:
        leaf_size (int): 每个叶子节点的最大样本数。默认值为64。
        """
        self.tree = BallTree(self.features, leaf_size=leaf_size)
        
    def find_nearest(self, query_feature, k=1):
        """
        找到与查询特征最相似的k个特征的索引。

        参数:
        query_feature (array-like): 查询的特征向量。
        k (int): 要找到的最相似特征的数量，默认为1。

        返回:
        tuple: 包含两个元素的元组，第一个元素是最相似特征的索引数组，第二个元素是对应的距离数组。

        异常:
        ValueError: 如果搜索树尚未构建，则抛出此异常。
        """
        if self.tree is None:
            raise ValueError("Search tree not built! Call build_search_tree first.")
        distances, indices = self.tree.query([query_feature], k=k)
        return indices[0], distances[0]

class TopoNode:
    """
    TopoNode类表示拓扑图中的一个节点。

    属性:
        id (int): 节点的唯一标识符。
        img_feature (Any): 节点的图像特征。
        neighbors (dict): 邻接表，表示与该节点相邻的节点及其边的权重。

    方法:
        __init__(id, img_feature): 初始化TopoNode类的新实例。
    """
    def __init__(self, id, img_feature):
        self.id = id
        self.img_feature = img_feature  # 图像特征
        self.neighbors = {}             # 邻接表 {node_id: edge_weight}

class TopoMap:
    """
    TopoMap 类表示一个拓扑地图，其中包含节点和边。

    属性:
        nodes (dict): 存储节点的字典，键为节点 ID，值为 TopoNode 对象。
        current_node_id (int 或 None): 当前节点的 ID。

    方法:
        __init__(): 初始化 TopoMap 对象。
        add_node(id, img_feature): 添加一个节点到拓扑地图中。
        add_edge(id1, id2, weight): 在两个节点之间添加一条边。
    """
    def __init__(self):
        self.nodes = {}                 # {node_id: TopoNode}
        self.current_node_id = None     # 当前节点

    def add_node(self, id, img_feature):
        self.nodes[id] = TopoNode(id, img_feature)

    def add_edge(self, id1, id2, weight):
        if id1 not in self.nodes or id2 not in self.nodes:
            return
        self.nodes[id1].neighbors[id2] = weight
        self.nodes[id2].neighbors[id1] = weight

class SolutionPlayer(KeyboardPlayerPyGame):
    def __init__(self):
        super(KeyboardPlayerPyGame, self).__init__()

        # 新增的组件
        self.topo_map = TopoMap()
        self.feature_extractor = ImageFeatureExtractor()
        self.image_database = ImageDatabase()

    # 重写pre_navigation方法
    # Override
    def pre_navigation(self):
        print("Starting pre-navigation setup...")

        # 1. 加载探索数据
        images, image_ids = self._load_exploration_data()

        # 2. 提取图像特征
        print("Extracting SIFT features...")
        all_descriptors = self.feature_extractor.extract_sift_batch(images)
        self.feature_extractor.train_codebook(all_descriptors)

        # 3. 构建图像数据库和拓扑地图
        print("Building database and topological map...")
        for i, img in enumerate(tqdm(images, desc="Processing images")):
            # 提取特征
            feature = self.feature_extractor.compute_image_descriptor(img)
            # 添加到数据库
            self.database.add_feature(feature)
            # 添加到拓扑图
            self.topo_map.add_node(i, feature)
            
            # 添加时序相邻的边
            if i > 0:
                weight = np.linalg.norm(feature - self.database.features[i-1])
                self.topo_map.add_edge(i, i-1, weight)

        # 4. 构建搜索树
        print("Building search tree...")
        self.database.build_search_tree()

    def _load_exploration_data(self):
        """加载探索数据
        Returns:
            images: List[np.ndarray] 图像列表
            image_ids: List[int] 图像ID列表
        """
        save_dir = "data/images_subsample/"
        image_files = natsorted([x for x in os.listdir(save_dir) if x.endswith('.jpg')])
        images = []
        image_ids = []
        
        for img_file in image_files:
            img = cv2.imread(os.path.join(save_dir, img_file))
            if img is not None:
                images.append(img)
                image_ids.append(int(img_file.split('.')[0]))
                
        return images, image_ids
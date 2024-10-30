# import necessary libraries and modules
import threading
from vis_nav_game import Player, Action, Phase
import pygame
import cv2

import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import BallTree
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from natsort import natsorted
from tqdm import tqdm


# Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        super(KeyboardPlayerPyGame, self).__init__()
        
        # Variables for saving data
        self.count = 0  # Counter for saving images
        self.save_dir = "data/images/"  # Directory to save images to

        # Initialize SIFT detector
        # SIFT stands for Scale-Invariant Feature Transform
        self.sift = cv2.SIFT_create()
        # Load pre-trained codebook for VLAD encoding
        # If you do not have this codebook comment the following line
        # You can explore the maze once and generate the codebook (refer line 181 onwards for more)
        self.codebook = None
        # Initialize database for storing VLAD descriptors of FPV
        self.database = None
        self.sift_descriptors = None  # SIFT descriptors of images in the database
        self.tree = None  # BallTree for nearest neighbor search
        self.goal = None  # Goal ID

        # 降维并可视化特征
        self.embedded_coords = None     # 降维后的坐标
        self.cluster_labels = None      # 聚类标签
        self.nav_figure = None          # matplotlib图像
        self.current_position = None    # 当前位置
        self.goal_position = None       # 目标位置
        self.visulization_lock = threading.Lock()  # 线程锁

        # 参数
        self.n_neighbors = 15           # 邻居数量
        self.n_clusters = 64            # 聚类数量

        self.current_rotation = 0       # 当前旋转角度
        self.DEGREE_PER_ROTATION = 2.5  # 每次旋转的角度

    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            #  Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                else:
                    # If a key is pressed that is not mapped to an action, then display target images
                    self.show_target_images()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def display_img_from_id(self, id, window_name):
        """
        Display image from database based on its ID using OpenCV
        """
        path = self.save_dir + str(id) + ".jpg"
        img = cv2.imread(path)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)

    def compute_sift_features(self):
        """
        Compute SIFT features for images in the data directory
        """
        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
        sift_descriptors = list()
        for img in tqdm(files, desc="Processing images"):
            img = cv2.imread(os.path.join(self.save_dir, img))
            # Pass the image to sift detector and get keypoints + descriptions
            # We only need the descriptors
            # These descriptors represent local features extracted from the image.
            _, des = self.sift.detectAndCompute(img, None)
            # Extend the sift_descriptors list with descriptors of the current image
            sift_descriptors.extend(des)
        return np.asarray(sift_descriptors)
    
    def get_VLAD(self, img):
        """
        Compute VLAD (Vector of Locally Aggregated Descriptors) descriptor for a given image
        """
        # We use a SIFT in combination with VLAD as a feature extractor as it offers several benefits
        # 1. SIFT features are invariant to scale and rotation changes in the image
        # 2. SIFT features are designed to capture local patterns which makes them more robust against noise
        # 3. VLAD aggregates local SIFT descriptors into a single compact representation for each image
        # 4. VLAD descriptors typically require less memory storage compared to storing the original set of SIFT
        # descriptors for each image. It is more practical for storing and retrieving large image databases efficicently.

        # Pass the image to sift detector and get keypoints + descriptions
        # Again we only need the descriptors
        _, des = self.sift.detectAndCompute(img, None)
        # We then predict the cluster labels using the pre-trained codebook
        # Each descriptor is assigned to a cluster, and the predicted cluster label is returned
        pred_labels = self.codebook.predict(des)
        # Get number of clusters that each descriptor belongs to
        centroids = self.codebook.cluster_centers_
        # Get the number of clusters from the codebook
        k = self.codebook.n_clusters
        VLAD_feature = np.zeros([k, des.shape[1]])

        # Loop over the clusters
        for i in range(k):
            # If the current cluster label matches the predicted one
            if np.sum(pred_labels == i) > 0:
                # Then, sum the residual vectors (difference between descriptors and cluster centroids)
                # for all the descriptors assigned to that clusters
                # axis=0 indicates summing along the rows (each row represents a descriptor)
                # This way we compute the VLAD vector for the current cluster i
                # This operation captures not only the presence of features but also their spatial distribution within the image
                VLAD_feature[i] = np.sum(des[pred_labels==i, :] - centroids[i], axis=0)
        VLAD_feature = VLAD_feature.flatten()
        # Apply power normalization to the VLAD feature vector
        # It takes the element-wise square root of the absolute values of the VLAD feature vector and then multiplies 
        # it by the element-wise sign of the VLAD feature vector
        # This makes the resulting descriptor robust to noice and variations in illumination which helps improve the 
        # robustness of VPR systems
        VLAD_feature = np.sign(VLAD_feature)*np.sqrt(np.abs(VLAD_feature))
        # Finally, the VLAD feature vector is normalized by dividing it by its L2 norm, ensuring that it has unit length
        VLAD_feature = VLAD_feature/np.linalg.norm(VLAD_feature)

        return VLAD_feature

    def get_neighbor(self, img):
        """
        Find the nearest neighbor in the database based on VLAD descriptor
        """
        # Get the VLAD feature of the image
        q_VLAD = self.get_VLAD(img).reshape(1, -1)
        # This function returns the index of the closest match of the provided VLAD feature from the database the tree was created
        # The '1' indicates the we want 1 nearest neighbor
        _, index = self.tree.query(q_VLAD, 1)
        return index[0][0]

    def pre_nav_compute(self):
        """
        Build BallTree for nearest neighbor search and find the goal ID
        """
        # If this function is called after the game has started

        # Compute sift features for images in the database
        if self.sift_descriptors is None:
            if os.path.exists("sift_descriptors.npy"):
                print("Loaded SIFT features from sift_descriptors.npy")
                self.sift_descriptors = np.load("sift_descriptors.npy")
            else:
                print("Computing SIFT features...")
                self.sift_descriptors = self.compute_sift_features()
                np.save("sift_descriptors.npy", self.sift_descriptors)
        else:
            pass
            # print("Loaded SIFT features from sift_descriptors.npy")

            # KMeans clustering algorithm is used to create a visual vocabulary, also known as a codebook,
            # from the computed SIFT descriptors.
            # n_clusters = 64: Specifies the number of clusters (visual words) to be created in the codebook. In this case, 64 clusters are being used.
            # init='k-means++': This specifies the method for initializing centroids. 'k-means++' is a smart initialization technique that selects initial 
            # cluster centers in a way that speeds up convergence.
            # n_init=10: Specifies the number of times the KMeans algorithm will be run with different initial centroid seeds. The final result will be 
            # the best output of n_init consecutive runs in terms of inertia (sum of squared distances).
            # The fit() method of KMeans is then called with sift_descriptors as input data. 
            # This fits the KMeans model to the SIFT descriptors, clustering them into n_clusters clusters based on their feature vectors

            # TODO: try tuning the function parameters for better performance
        if self.codebook is None:
            if os.path.exists("codebook.pkl"):
                self.codebook = pickle.load(open("codebook.pkl", "rb"))
            else:
                print("Computing codebook...")
                self.codebook = KMeans(n_clusters=128, init='k-means++', n_init=5, verbose=1).fit(self.sift_descriptors)
                pickle.dump(self.codebook, open("codebook.pkl", "wb"))
        else:
            pass
            # print("Loaded codebook from codebook.pkl")
        
        # get VLAD emvedding for each image in the exploration phase
        if self.database is None:
            if os.path.exists("database.npy"):
                print("Loaded database from database.npy")
                self.database = np.load("database.npy")
            else:
                self.database = []
                print("Computing VLAD embeddings...")
                exploration_observation = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
                for img in tqdm(exploration_observation, desc="Processing images"):
                    img = cv2.imread(os.path.join(self.save_dir, img))
                    VLAD = self.get_VLAD(img)
                    self.database.append(VLAD)
                self.database = np.array(self.database)
                np.save("database.npy", self.database)

            # Build a BallTree for fast nearest neighbor search
            # We create this tree to efficiently perform nearest neighbor searches later on which will help us navigate and reach the target location
            
            # TODO: try tuning the leaf size for better performance
        if self.tree is None:
            if os.path.exists("tree.pkl"):
                self.tree = pickle.load(open("tree.pkl", "rb"))
            else:
                print("Building BallTree...")
                tree = BallTree(self.database, leaf_size=60)
                self.tree = tree
                pickle.dump(self.tree, open("tree.pkl", "wb"))

        # Display the navigation map
        print("Creating navigation map...")
        self.create_navigation_map()

    def pre_navigation(self):
        """
        Computations to perform before entering navigation and after exiting exploration
        """
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.pre_nav_compute()
        
    def display_next_best_view(self):
        """
        Display the next best view based on the current first-person view
        """

        # TODO: could you write this function in a smarter way to not simply display the image that closely 
        # matches the current FPV but the image that can efficiently help you reach the target?

        # Get the neighbor of current FPV
        # In other words, get the image from the database that closely matches current FPV
        index = self.get_neighbor(self.fpv)

        # 更新当前位置
        self.current_position = self.embedded_coords[index]
        self.update_navigation_visualization()

        # 原先的代码
        # # Display the image 5 frames ahead of the neighbor, so that next best view is not exactly same as current FPV
        # self.display_img_from_id(index+5, f'Next Best View')
        # # Display the next best view id along with the goal id to understand how close/far we are from the goal
        print(f'View ID: {index} || Goal ID: {self.goal}')

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
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:
                # TODO: could you employ any technique to strategically perform exploration instead of random exploration
                # to improve performance (reach target location faster)?

                pass
            # If in navigation stage
            elif self._state[1] == Phase.NAVIGATION:
                # TODO: could you do something else, something smarter than simply getting the image closest to the current FPV?

                if self.goal is None:
                    # Get the neighbor nearest to the front view of the target image and set it as goal
                    targets = self.get_target_images()
                    index = self.get_neighbor(targets[0])
                    self.goal = index
                    self.goal_position = self.embedded_coords[index]
                    print(f'Goal ID: {self.goal}')
                
                # Key the state of the keys
                keys = pygame.key.get_pressed()
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q]:
                    self.display_next_best_view()

                if keys[pygame.K_LEFT]:
                    self.current_rotation -= self.DEGREE_PER_ROTATION
                if keys[pygame.K_RIGHT]:
                    self.current_rotation += self.DEGREE_PER_ROTATION

        # Display the first-person view image on the pygame screen
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

    def create_navigation_map(self):
        """
        创建导航地图
        """
        X = np.array(self.database) # 数据库

        # 降维
        if self.embedded_coords is None:
            if os.path.exists("embedded_coords.pkl"):
                print("Loaded embedded coordinates from embedded_coords.pkl")
                self.embedded_coords = pickle.load(open("embedded_coords.pkl", "rb"))
            else:
                print("Computing TSNE embedding...")
                tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
                self.embedded_coords = tsne.fit_transform(X)
                pickle.dump(self.embedded_coords, open("embedded_coords.pkl", "wb"))

        # 聚类
        if self.cluster_labels is None:
            if os.path.exists("cluster_labels.pkl"):
                print("Loaded cluster labels from cluster_labels.pkl")
                self.cluster_labels = pickle.load(open("cluster_labels.pkl", "rb"))
            else:
                print("Clustering...")
                km = KMeans(n_clusters=self.n_clusters, random_state=0)
                self.cluster_labels = km.fit_predict(X)
                pickle.dump(self.cluster_labels, open("cluster_labels.pkl", "wb"))

        # 可视化
        self.nav_figure = plt.figure(figsize=(12, 9))
        self.update_navigation_visualization()

    def update_navigation_visualization(self):
        with self.visulization_lock:
            plt.clf()
            
            # 创建子图网格,左边是导航地图,右边是指南针
            gs = plt.GridSpec(1, 4, figure=self.nav_figure)
            ax_map = plt.subplot(gs[0, :3])  # 导航地图占3/4
            ax_compass = plt.subplot(gs[0, 3])  # 指南针占1/4
            
            # 1. 绘制导航地图部分
            scatter = ax_map.scatter(self.embedded_coords[:, 0], 
                                self.embedded_coords[:, 1],
                                c=self.cluster_labels, 
                                cmap='tab10', 
                                alpha=0.6, 
                                s=2)
            
            # 绘制当前位置
            if self.current_position is not None:
                ax_map.scatter(self.current_position[0], 
                            self.current_position[1], 
                            c='r', 
                            marker='x', 
                            s=150, 
                            label='Current Position')
                
            # 绘制目标位置
            if self.goal is not None:
                goal_pos = self.embedded_coords[self.goal]
                ax_map.scatter(goal_pos[0], 
                            goal_pos[1], 
                            c='r', 
                            marker='^', 
                            s=150, 
                            label='Goal Position')
                
            ax_map.legend()
            ax_map.set_title('Navigation Map')
            
            # 2. 绘制指南针部分
            ax_compass.set_aspect('equal')
            
            # 绘制指南针圆圈
            circle = plt.Circle((0, 0), 1, color='lightgrey', fill=True)
            ax_compass.add_artist(circle)
            
            # 绘制方向标记
            directions = ['N', 'E', 'S', 'W']
            angles = [90, 0, 270, 180]
            for direction, angle in zip(directions, angles):
                rad = np.deg2rad(angle)
                x, y = np.cos(rad), np.sin(rad)
                ax_compass.text(x * 1.1, y * 1.1, direction, 
                            ha='center', va='center', 
                            fontsize=12, fontweight='bold')
            
            # 绘制指南针指针
            rotation_rad = np.deg2rad(90 - self.current_rotation)
            needle_x, needle_y = np.cos(rotation_rad), np.sin(rotation_rad)
            ax_compass.plot([0, needle_x], [0, needle_y], 
                        color='red', linewidth=3)
            
            # 设置指南针的显示范围和样式
            ax_compass.set_xlim(-1.5, 1.5)
            ax_compass.set_ylim(-1.5, 1.5)
            ax_compass.axis('off')
            ax_compass.set_title('Compass')
            
            # 3. 调整布局
            plt.tight_layout()
            
            # 4. 转换并显示
            canvas = FigureCanvasAgg(self.nav_figure)
            canvas.draw()
            nav_map = np.asarray(canvas.buffer_rgba())
            nav_map = cv2.cvtColor(nav_map, cv2.COLOR_RGBA2BGR)
            cv2.imshow('Navigation Map', nav_map)
            cv2.waitKey(1)


if __name__ == "__main__":
    import vis_nav_game
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())

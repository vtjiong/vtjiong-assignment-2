import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime

class Manual():
    def __init__(self, data,k,centroids):
        self.data = data
        self.centroids=centroids
        self.k = k
        self.assignment = [-1 for _ in range(len(data))]
        self.snaps = []
        self.timestamp = datetime.now().strftime("%H%M%S")
        

    def snap(self, centers):
        TEMPFILE = f"manual{self.timestamp}.png"
        # fig, ax = plt.subplots()
        # ax.scatter(self.data[:, 0], self.data[:, 1], c=self.assignment)
        # ax.scatter(centers[:,0], centers[:, 1], c='r')
        plt.figure(figsize=(8, 6))  # Match the figure size
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.assignment, cmap='viridis')  # Use colormap for clarity
        if len(centers)>0:
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100,
                        label='Centers')  # Plot centers with distinct style
        plt.xlim(-11, 11)  # Set limits to match the original plot
        plt.ylim(-11, 11)
        plt.title('KMeans Clustering Data')  # Title
        plt.xticks(np.arange(-10, 11, 5))  # X-ticks
        plt.yticks(np.arange(-10, 11, 5))  # Y-ticks
        plt.grid(True)  # Grid
        plt.axhline(0, color='black', linewidth=1)  # Horizontal line at y=0
        plt.axvline(0, color='black', linewidth=1)  # Vertical line at x=0
        for spine in plt.gca().spines.values():
            spine.set_visible(False)  # Hide spines
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        filename = f'static/{TEMPFILE}'  
        plt.savefig(filename)
        self.snaps.append(im.open(img_bytes))
        plt.close()

    def create_gif(self, output_filename):
        static_dir = os.path.join(os.getcwd(), 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        # # Check if there are multiple frames
        full_output_path = os.path.join(static_dir, output_filename)
        if len(self.snaps) > 1:
            # Save the collected snapshots as a GIF
            self.snaps[0].save(
                full_output_path,
                save_all=True,
                append_images=self.snaps[1:],
                duration=500,  # Duration between frames in milliseconds
            )
            print(f"GIF saved as {full_output_path}")
        else:
            print("Not enough frames to create a GIF. Only one image was generated.")

    def isunassigned(self, i):
        return self.assignment[i] == -1

    def initialize(self):
        return self.data[np.random.choice(len(self.data) - 1, size=self.k, replace=False)]

    def make_clusters(self, centers):
        for i in range(len(self.assignment)):
            for j in range(self.k):
                if self.isunassigned(i):
                    self.assignment[i] = j
                    dist = self.dist(centers[j], self.data[i])
                else:
                    new_dist = self.dist(centers[j], self.data[i])
                    if new_dist < dist:
                        self.assignment[i] = j
                        dist = new_dist

    def compute_centers(self):
        centers = []
        for i in range(self.k):
            cluster = []
            for j in range(len(self.assignment)):
                if self.assignment[j] == i:
                    cluster.append(self.data[j])
            centers.append(np.mean(np.array(cluster), axis=0))
        return np.array(centers)

    def unassign(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers):
        for i in range(self.k):
            if self.dist(centers[i], new_centers[i]) != 0:
                return True
        return False

    def dist(self, x, y):
        # Euclidean distance
        return sum((x - y) ** 2) ** (1 / 2)

    def lloyds(self):
        centers = self.centroids
        self.snap(centers)
        self.make_clusters(centers)
        new_centers = self.compute_centers()
        self.snap(new_centers)
        while self.are_diff(centers, new_centers):
            self.unassign()
            centers = new_centers
            self.make_clusters(centers)
            new_centers = self.compute_centers()
            self.snap(new_centers)
        self.snap(new_centers)
        print('success')
        return
    
    def reset (self):
        self.unassign()
        self.snap(centers=[])
        return self.snap

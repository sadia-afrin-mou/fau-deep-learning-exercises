import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path = file_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        
        # load labels from JSON file
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
            
        self.image_file_names = sorted([f for f in os.listdir(file_path) if f.endswith('.npy')])
        self.images = [np.load(os.path.join(file_path, f)) for f in self.image_file_names]
        self.num_images = len(self.images)
        
        # index array and current position
        self.indices = np.arange(self.num_images)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_position = 0
        self._current_epoch = 0
        
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 
                          5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        images = []
        labels = []
        
        for _ in range(self.batch_size):
            if self.current_position >= self.num_images:
                self.current_position = 0
                self._current_epoch += 1
                if self.shuffle:
                    np.random.shuffle(self.indices)
            
            # image and label
            image = self.images[self.indices[self.current_position]].copy()
            image_file_name = self.image_file_names[self.indices[self.current_position]]
            label = int(self.labels[image_file_name.split('.')[0]])
            
            # resize image if needed
            if image.shape != tuple(self.image_size):
                image = resize(image, self.image_size)
            
            # apply augmentations
            image = self.augment(image)
            
            images.append(image)
            labels.append(label)
            self.current_position += 1
        
        return np.array(images), np.array(labels)

    def augment(self, image):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring:
            r = np.random.random()  
            if r < 0.25:
                image = np.fliplr(image)
            elif r < 0.5:
                image = np.flipud(image)
        
        if self.rotation:
            k = np.random.randint(0, 4)  # 1=90°, 2=180°, 3=270°
            image = np.rot90(image, k)
        
        return image

    def current_epoch(self):
        return self._current_epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images, labels = self.next()
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, (image, label) in enumerate(zip(images, labels)):
            axes[i].imshow(image)
            axes[i].set_title(f'{self.class_name(label)}')
            axes[i].axis('off')
        
        for i in range(len(images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


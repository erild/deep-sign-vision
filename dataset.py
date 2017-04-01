import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

class Dataset:
  def __init__(self, path):
    self.path = path
    print(self.path)
    self.classcontent = {'l': [], 'e': [], 't': [], 'r': [], 'o': [], 'i': [], 'c': [], 'p': [], 'n': [], 'v': [], 'm': [], 'w': [], 's': [], 'x': [], 'q': [], 'b': [], 'd': [], 'f': [], 'y': [], 'k': [], 'h': [], 'a': [], 'g': [], 'u': []}
    self.classcorresp = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12, 'o': 13, 'p': 14, 'q': 15, 'r': 16, 's': 17, 't': 18, 'u': 19, 'v': 20, 'w': 21, 'x': 22, 'y': 23}
    self.classCursor = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'k': 0, 'l': 0, 'm': 0, 'n': 0, 'o': 0, 'p': 0, 'q': 0, 'r': 0, 's': 0, 't': 0, 'u': 0, 'v': 0, 'w': 0, 'x': 0, 'y': 0}
    self.classCursor_test = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'k': 0, 'l': 0, 'm': 0, 'n': 0, 'o': 0, 'p': 0, 'q': 0, 'r': 0, 's': 0, 't': 0, 'u': 0, 'v': 0, 'w': 0, 'x': 0, 'y': 0}
    with os.scandir(path) as it:
      for entry in it:
        if not entry.name.startswith('.') and entry.is_dir():
          with os.scandir(path+entry.name+'/') as it2:
            for entry2 in it2:
              if not entry2.name.startswith('.') and entry2.is_file():
                self.classcontent[entry.name].append(entry2.name)

    self.classSize = {}
    self.totalSize = 0
    for x in self.classcontent:
      self.classSize[x] = len(self.classcontent[x])
      self.totalSize += self.classSize[x]
      random.shuffle(self.classcontent[x])

    # Init params

    self.train_size = 3*self.totalSize//5 # size of train set
    self.test_size = 2*self.totalSize//5 # size of test set
    self.scale_size = 227

    self.n_classes = 24

    # self.meanImg = cv2.imread('meanImg.png', cv2.IMREAD_COLOR)


    for x in self.classCursor_test:
      self.classCursor_test[x] = int(self.train_size/self.n_classes)

  def next_batch(self, batch_size, phase):
    # Get next batch of images and labels
    imagesLinks = []
    imagesLabels = np.zeros((batch_size, self.n_classes))
    images = np.ndarray([batch_size, self.scale_size, self.scale_size, 3])
    if phase == 'train':
      i = 0
      while i < batch_size:
        for x in self.classcontent:
          if (i < batch_size) :
            imagesLinks.append(x+"/"+self.classcontent[x][self.classCursor[x]])
            imagesLabels[i][self.classcorresp[x]] = 1
            self.classCursor[x] += 1
            if self.classCursor[x] >= self.classSize[x] - self.train_size/self.n_classes:
              self.classCursor[x] = 0
            i +=1
      for i in range(0,len(imagesLinks)):
        img = cv2.imread(self.path+imagesLinks[i], cv2.IMREAD_COLOR)
        images[i] = img
    elif phase == 'test':
      i = 0
      while i < batch_size:
        for x in self.classcontent:
          if (i < batch_size) :
            imagesLinks.append(x+"/"+self.classcontent[x][self.classCursor_test[x]])
            imagesLabels[i][self.classcorresp[x]] = 1
            self.classCursor_test[x] += 1
            if self.classCursor_test[x] >= self.classSize[x]:
              self.classCursor_test[x] = int(self.train_size/self.n_classes)
            i +=1
      for i in range(0,len(imagesLinks)):
        img = cv2.imread(self.path+imagesLinks[i], cv2.IMREAD_COLOR)
        images[i] = img
    else:
      return None, None
    return np.array(images), imagesLabels




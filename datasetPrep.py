import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Process
import sys, argparse, os
import cv2


class DatasetPrep():
  """docstring for DatasetPrep"""

  def loadImage(self, src):
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    height = img.shape[0]
    width = img.shape[1]
    return img, height, width

  # following condition from http://pesona.mmu.edu.my/~johnsee/research/papers/files/rgbhcbcr_m2usic06.pdf
  # adapted tobe sure to have all features
  def skin_rule_1(self, R, G, B):
    cond1 = (B > 20 and G > 40 and R > 95 and R > G + 15 and R > B)
    cond2 = (R > 220 and G > 210 and B > 170 and abs(int(R)-int(G)) <= 15 and R>B and G>B)
    return (cond1 or cond2)
    # return (cond1)

  def skin_rule_2(self, Y, Cr, Cb):
    cond3 = Cr <= 1.5862*Cb+20
    cond4 = Cr >= 0.3448*Cb+76.2069
    cond5 = Cr >= -4.5652*Cb+234.5652
    cond6 = Cr <= -1.15*Cb+301.75
    cond7 = Cr <= -2.2857*Cb+432.85
    return (cond3 and cond4 and cond5 and cond6 and cond7)

  def skin_rule_3(self, H):
    return (H<25 or H>230)

  def ispeau_1(self, I, height, width):
    mask = np.zeros((height,width,1), np.uint8)
    # YCrCb_I = cv2.cvtColor(I, cv2.COLOR_BGR2YCrCb)
    # HSV_I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    for x in range(0,I.shape[0]):
      for y in range(0,I.shape[1]):
        # if self.skin_rule_1(I[x][y][2], I[x][y][1], I[x][y][0]) and self.skin_rule_2(YCrCb_I[x][y][0], YCrCb_I[x][y][1], YCrCb_I[x][y][2]) and self.skin_rule_3(HSV_I[x][y][0]):
        if self.skin_rule_1(I[x][y][2], I[x][y][1], I[x][y][0]):
          mask[x][y] = 255
        else:
          pass
          mask[x][y] = 0
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    return mask

  def histEqualisation(self, img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

  def meanFilter(self, img, size):
    img = cv2.medianBlur(img, size)
    return img

  def keepSkinOnly(self, img, height, width):
    mask = self.ispeau_1(img, height, width)
    masked_data = cv2.bitwise_and(img, img, mask=mask)
    return masked_data

  def applyToDir(self, path, outpath):
    targetHeight = 227
    targetWidth = 227
    with os.scandir(path) as it:
      for entry in it:
        if not entry.name.startswith('.') and entry.is_file():
          img, height, width = self.loadImage(path+entry.name);
          cv2.imshow('image', img)
          cv2.waitKey(0)
          cv2.destroyAllWindows()
          img = self.meanFilter(img, 3)
          cv2.imshow('image', img)
          cv2.waitKey(0)
          cv2.destroyAllWindows()
          img = self.histEqualisation(img)
          cv2.imshow('image', img)
          cv2.waitKey(0)
          cv2.destroyAllWindows()
          img = self.meanFilter(img, 3)
          cv2.imshow('image', img)
          cv2.waitKey(0)
          cv2.destroyAllWindows()
          img = self.keepSkinOnly(img, height, width)
          cv2.imshow('image', img)
          cv2.waitKey(0)
          cv2.destroyAllWindows()
          if height > width:
            r = targetHeight / height
            newW = int(width*r)
            resized = cv2.resize(img, (newW, targetHeight), interpolation = cv2.INTER_AREA)
            Wdiff = targetWidth - newW
            resizedImg = cv2.copyMakeBorder(resized,0,0,Wdiff//2 + Wdiff%2,Wdiff//2,cv2.BORDER_CONSTANT,value=[0,0,0])
            cv2.imshow('image', resizedImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(outpath+entry.name, resizedImg)
          else:
            r = targetWidth / width
            newH = int(height*r)
            resized = cv2.resize(img, (targetWidth, newH), interpolation = cv2.INTER_AREA)
            Hdiff = targetHeight - newH
            resizedImg = cv2.copyMakeBorder(resized,Hdiff//2 + Hdiff%2,Hdiff//2,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
            cv2.imwrite(outpath+entry.name, resizedImg)
        if not entry.name.startswith('.') and entry.is_dir():
          try:
            os.mkdir(outpath+entry.name)
          except Exception as e:
            pass
          print(path+entry.name)
          # p = Process(target=self.applyToDir, args=(path+entry.name+'/', outpath+entry.name+'/',))
          # p.start()
          self.applyToDir(path+entry.name+'/', outpath+entry.name+'/')

  def adjust_gamma(self, image, gamma):
     invGamma = 1.0 / gamma
     table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
     return cv2.LUT(image, table)

  def augmentData(self, path, outpath):
    value1 = 20
    value2 = 40
    with os.scandir(path) as it:
      for entry in it:
        if not entry.name.startswith('.') and entry.is_file():
          img, height, width = self.loadImage(path+entry.name)
          cv2.imwrite(outpath+entry.name[:-4]+"_0.png", img)

          adjusted = adjust_gamma(img, gamma=1.2)
          cv2.imwrite(outpath+"1_"+entry.name[:-4]+"_1.png", adjusted)

          adjusted = adjust_gamma(img, gamma=1.5)
          cv2.imwrite(outpath+"2_"+entry.name[:-4]+"_2.png", adjusted)

          adjusted = adjust_gamma(img, gamma=0.7)
          cv2.imwrite(outpath+"3_"+entry.name[:-4]+"_3.png", adjusted)

          adjusted = adjust_gamma(img, gamma=0.5)
          cv2.imwrite(outpath+"4_"+entry.name[:-4]+"_4.png", adjusted)
        if not entry.name.startswith('.') and entry.is_dir():
          try:
            os.mkdir(outpath+entry.name)
          except Exception as e:
            pass
          print(path+entry.name)
          self.augmentData(path+entry.name+'/', outpath+entry.name+'/')

  def resizeDir(self, path, outpath):
    with os.scandir(path) as it:
      for entry in it:
        if not entry.name.startswith('.') and entry.is_file():
          img, height, width = self.loadImage(path+entry.name);
          if height > width:
            r = targetHeight / height
            newW = int(width*r)
            resized = cv2.resize(img, (newW, targetHeight), interpolation = cv2.INTER_AREA)
            Wdiff = targetWidth - newW
            resizedImg = cv2.copyMakeBorder(resized,0,0,Wdiff//2 + Wdiff%2,Wdiff//2,cv2.BORDER_CONSTANT,value=[0,0,0])
            cv2.imwrite(outpath+entry.name, resizedImg)
          else:
            r = targetWidth / width
            newH = int(height*r)
            resized = cv2.resize(img, (targetWidth, newH), interpolation = cv2.INTER_AREA)
            Hdiff = targetHeight - newH
            resizedImg = cv2.copyMakeBorder(resized,Hdiff//2 + Hdiff%2,Hdiff//2,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
            cv2.imwr
        if not entry.name.startswith('.') and entry.is_dir():
          try:
            os.mkdir(outpath+entry.name)
          except Exception as e:
            pass
          self.resizeDir(path+entry.name+'/', outpath+entry.name+'/')

  def averageDir(self, path, outpath):
    meanImg, height, width = loadImage('meanImg.png');
    with os.scandir(path) as it:
      for entry in it:
        if not entry.name.startswith('.') and entry.is_file():
          img, height, width = loadImage(path+entry.name);
          newImg = img - meanImg
          cv2.imwrite(outpath+entry.name, newImg)
        if not entry.name.startswith('.') and entry.is_dir():
          try:
            os.mkdir(outpath+entry.name)
          except Exception as e:
            pass
          self.averageDir(path+entry.name+'/', outpath+entry.name+'/')

  def getChannelTotal(self, path):
    img_total = np.zeros((227,227,3), np.uint32)
    img_count = 0
    with os.scandir(path) as it:
      for entry in it:
        if not entry.name.startswith('.') and entry.is_file():
          img, height, width = self.loadImage(path+entry.name);
          # print(entry.name)
          # print(img.shape)
          # print(img_total.shape)
          img_total += img
          img_count += 1
          # cv2.imwrite(outpath+entry.name, img)
        if not entry.name.startswith('.') and entry.is_dir():
          print(path+entry.name)
          sub_img_total, sub_img_count = self.getChannelTotal(path+entry.name+"/")
          img_total += sub_img_total
          img_count += sub_img_count
          # p = Process(target=applyToDir, args=(path+entry.name+'/', outpath+entry.name+'/',))
          # p.start()
    return img_total, img_count

  def getCountPerClass(self, path):
    count = 0
    countDict = {}
    with os.scandir(path) as it:
      for entry in it:
        if not entry.name.startswith('.') and entry.is_file():
          count +=1
        if not entry.name.startswith('.') and entry.is_dir():
          dirCount = self.getCountPerClass(path+entry.name+'/')
          countDict[entry.name] = dirCount
      # print ("Sample for "+path+" : "+str(count))
      if len(countDict):
        min_value = min(countDict.values())
        total_value = sum(countDict.values())
        print(min_value)
        print(total_value)
      return count

  def generateMeanImage(self, path, output):
    meanImg = np.zeros((227,227,3), np.uint8)
    I, nbr = self.getChannelTotal(path)
    for x in range(0,227):
        for y in range(0,227):
          meanImg[x][y] = I[x][y]/nbr
    cv2.imwrite(output, meanImg)

if __name__ == '__main__':
  # Get command line argument
  parser = argparse.ArgumentParser(description='Do data preparation for sign recognition dataset')
  parser.add_argument("-d", dest="input", required=True, help='Input directory')
  parser.add_argument("-o", dest="output", required=True, help='output directory/file')
  parser.add_argument('-a',dest='prep', default='prep', help='To do pre treatment on dir: prep or mean for mean image generation')

  args = parser.parse_args()

  datasetPrep = DatasetPrep()
  if args.prep == 'prep':
    datasetPrep.applyToDir(args.input, args.output)
  elif args.mean == 'mean':
    datasetPrep.generateMeanImage(args.input, args.output)


  # augmentData(sys.argv[1], sys.argv[2])
  #
  # meanImg = img_total = np.zeros((227,227,3), np.uint8)
  # I, nbr = getChannelTotal(sys.argv[1])
  # for x in range(0,227):
  #     for y in range(0,227):
  #       meanImg[x][y] = I[x][y]/nbr
  # cv2.imwrite('meanImg.png', meanImg)

  # averageDir(sys.argv[1], sys.argv[2])


  # getCountPerClass(sys.argv[1])

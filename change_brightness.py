import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def a(img,minimum_brightness = 0.7):
   cols, rows = img.shape[:2]
   brightness = np.sum(img) / (255 * cols * rows)
   ratio = brightness / minimum_brightness
   # if ratio >= 1:
   #    print("Image already bright enough")
   #    return img
   return cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)

x = r'C:\Users\doank\PycharmProjects\Face-Verification\static\doananh\10.jpg'  #location of the image
original = cv2.imread(x, 1)
cv2.imshow('original',original)

gamma = 1.5                                   # change the value here to get different result
# adjusted = adjust_gamma(original,gamma= gamma)
adjusted = a(original)
cv2.imshow("gammam image 1", adjusted)

cv2.waitKey(0)
cv2.destroyAllWindows()
# from deepface.DeepFace import find
# img = r'C:\Users\doank\PycharmProjects\Face-Verification\static\doan1\0.jpg'
# db = r'C:\Users\doank\PycharmProjects\Face-Verification\static'
# result = find(img,db,delta=0.8666)
# #print(result)
# for rs,i in zip(result['identity'],result['VGG-Face_euclidean_l2']):
#     print(rs.split('\\')[6],i)
import cv2
img = r'C:\Users\doank\PycharmProjects\Face-Verification\static\doan1\0.jpg'
img2 = r'C:\Users\doank\PycharmProjects\Face-Verification\static\doan\z2617619904513_2bf5ca97633c310abd01408e923af5e3.jpg'
image = cv2.imread(img2)
image = cv2.resize(image,(480,640))
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',image_gray)
cv2.waitKey(0)
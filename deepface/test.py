import DeepFace
import pandas as pd
import cv2
# DeepFace.stream(r'C:\Users\doank\PycharmProjects\Face-Verification-2\static\doan', model_name='VGG-Face', distance_metric='euclidean_l2',
#                 time_threshold=3, frame_threshold=10, delta=0.8)
# # model = DeepFace.build_model('VGG-Face')
# rs = DeepFace.find(r'C:\Users\doank\PycharmProjects\deepface\data\doan\6.jpg',
#                    r"C:\Users\doank\PycharmProjects\deepface\data\vien", distance_metric='euclidean_l2', model=model,
#                    detector_backend='mtcnn', delta=0.9)
# count = 0
# for i in rs['identity']:
#     i = i.split('\\')[6].split('/')[0]
#     if i == 'doan':
#         count = count + 1
#     print(i)
# print(len(rs['identity']))
# print('precison:', count / (len(rs['identity'])+10-count))
# print('recal:',count / (len(rs['identity'])+0.0000000001))
# detected_face = DeepFace.detectFace(r'C:\Users\doank\PycharmProjects\Face-Verification-2\static\doan\7.jpg', detector_backend = 'facenet')
# img = cv2.cvtColor(detected_face,cv2.COLOR_BGR2RGB)
# cv2.imshow('hj',img)
# cv2.waitKey(0)

# from deepface.Face_recognition_vjpro import get_df,add_employee
# df1 = add_employee(7749,r'C:\Users\doank\PycharmProjects\deepface\static\hh')
# print(len(df1))
# anal(df1,model_name='VGG-Face',distance_metric='euclidean_l2',delta = 0.8)
# df = pd.read_csv(r'C:\Users\doank\PycharmProjects\deepface\deepface\hjhj.csv')
# print(df[df['name']=='hieu'])
# def register(vid):
#     vidcap = cv2.VideoCapture(r'C:\Users\doank\PycharmProjects\deepface\static\big_buck_bunny_720p_5mb.mp4')
#     success, image = vidcap.read()
#     count = 0
#     while success:
#         cv2.imwrite("static/frame/frame%d.jpg" % count, image)  # save frame as JPEG file
#         success, image = vidcap.read()
#         print('Read a new frame: ', success)
#         count += 1
import cv2
scale = 0.25
img = cv2.imread(r'C:\Users\doank\PycharmProjects\Face-Verification-2\static\doan\10.jpg')
print(img.shape)
# img = cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)),interpolation = cv2.INTER_AREA)
# print(img.shape)
if img.shape[0] < img.shape[1]:
    img = cv2.resize(img,(640, 480),interpolation = cv2.INTER_AREA)
else:
    img = cv2.resize(img,(480, 640),interpolation = cv2.INTER_AREA)
print(img.shape)
#cv2.imwrite('test2.jpg',img)
cv2.imshow('hjhj',img)
cv2.waitKey(0)
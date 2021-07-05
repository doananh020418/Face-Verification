import DeepFace
import pandas as pd
DeepFace.stream(r'C:\Users\doank\PycharmProjects\deepface\data', model_name='VGG-Face', distance_metric='euclidean_l2',
                time_threshold=2, frame_threshold=10, delta=0.8)
# model = DeepFace.build_model('VGG-Face')
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
# # detected_face = DeepFace.detectFace(r'C:\Users\doank\PycharmProjects\deepface\data\doan\doan1.jpg', detector_backend = 'mtcnn')
# # img = cv2.cvtColor(detected_face,cv2.COLOR_BGR2RGB)
# # cv2.imshow('hj',detected_face)
# # cv2.waitKey(0)

# from deepface.Face_recognition_vjpro import get_df,anal
# df1 = get_df(r'C:\Users\doank\PycharmProjects\deepface\data',model_name='VGG-Face',distance_metric='euclidean_l2')
#
# anal(df1,model_name='VGG-Face',distance_metric='euclidean_l2',delta = 0.8)
# df = pd.read_csv('df.csv')
# print(df['distance_metric'].values[0])

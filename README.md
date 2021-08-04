##Serve
This code performs a server side, which process incoming frame and return a face verification.
##Setup
'''
pip install -r requirements.txt
'''
##Run
python socket_io_flask_api.py
###Feature
-Đăng ký: kết nối qua event handler: 'face_register',đầu vào là chuỗi frame và id, trả về trạng thái đăng ký và frame hiển thị khuôn mặt được phát hiện
-Chấm công: kết nối qua 'face_verify',đầu vào là chuỗi frame và id, trả về kết quả xác minh khuôn mặt và frame chứa khuôn mặt chấm công
###Note
Ảnh truyền và nhận đều ở dạng base64
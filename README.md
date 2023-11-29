# traffic-sign-detection

Traffic Sign Detection là một bài toán ứng dụng các thuật toán liên quán đến Object Detection
để phát hiện các biển báo giao thông trên đường. Các mô hình Traffic Sign Detection thường được sử
dụng rất nhiều trong các bài toán lớn như Self-driving Cars, Advanced Driver Assistance Systems...
Một chương trình Traffic Sign Detection thường bao gồm hai giai đoạn là xác định vị trí của biển báo
và nhận diện tên biển báo. Vì vậy, một chương trình có độ chính xác cao cần xây dựng tốt cả hai thành
phần này
<div align="center">
  <img width="400" alt="image" src="https://github.com/minh190902/traffic-sign-detection/assets/128236164/b0f3af8f-2091-4043-b245-f7c306d8fb3c">
</div>
Trong project này, mình sẽ xây dựng một chương trình Traffic Sign Detection sử dụng mô hình
Support Vector Machine (SVM). Input và output của chương trình như sau:

  - Input: Một bức ảnh có chứa biển báo giao thông.
  - Output: Vị trí tọa độ và tên (class) của các biển báo có trong ảnh.

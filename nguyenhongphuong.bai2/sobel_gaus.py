import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_images(images, titles, cmap='gray'):
    plt.figure(figsize=(15, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.show()
#Hàm này tạo một hình có năm ô phụ và hiển thị các hình ảnh được cung cấp cùng với tiêu đề tương ứng của chúng. 
# cmap='gray' đặt bản đồ màu thành thang độ xám để hiển thị phù hợp.

image_path = 'n1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Sobel Kernels
sobel_x = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1], 
                    [ 0,  0,  0], 
                    [ 1,  2,  1]])
#Đây là các ma trận 3x3 được sử dụng cho tích chập để phát hiện các cạnh ngang (sobel_x) và dọc (sobel_y). 
# Các giá trị biểu thị trọng số được áp dụng cho các pixel lân cận.

sobel_x_edges = cv2.filter2D(image, -1, sobel_x)
sobel_y_edges = cv2.filter2D(image, -1, sobel_y)
#tính toán độ dốc theo hướng x và y. Độ sâu âm (-1) cho biết độ sâu của bộ lọc giống với độ sâu của hình ảnh.

# Magnitude of gradient
sobel_edges = np.sqrt(sobel_x_edges**2 + sobel_y_edges**2)
sobel_edges = np.uint8(np.clip(sobel_edges, 0, 255))

# Step 3: Laplacian of Gaussian (LoG) Edge Detection
log_kernel = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]])
#Việc làm mờ Gaussian ban đầu giúp giảm nhiễu trước khi phát hiện cạnh.
# sau đó làm nổi bật các vùng có cường độ thay đổi cao (các cạnh).
log_edges = cv2.filter2D(image, -1, log_kernel)
 
display_images([image, sobel_x_edges, sobel_y_edges, sobel_edges, log_edges],
               ['Original Image', 'Sobel X', 'Sobel Y', 'Sobel Edges', 'LoG Edges'])

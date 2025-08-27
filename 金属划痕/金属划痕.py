# 金属划痕识别

import cv2
import numpy as np
import os

# 图片文件夹路径
IMAGE_FOLDER = 'metal_scratches'

# 检查是否存在
if not os.path.exists(IMAGE_FOLDER):
    print(f"错误：文件夹 '{IMAGE_FOLDER}' 不存在！")
    print(f"请在脚本旁边创建一个名为 '{IMAGE_FOLDER}' 的文件夹，并将你的.jpg图片放入其中。")
    os.makedirs(IMAGE_FOLDER)

# 获取文件夹中所有jpg图片的文件名
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.JPG', '.png'))]

if not image_files:
    print(f"警告：在文件夹 '{IMAGE_FOLDER}' 中没有找到任何图片。")

for image_file in image_files:
    image_path = os.path.join(IMAGE_FOLDER, image_file)

    # 1.加载图片
    try:
        original_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if original_image is None:
            print(f"警告：文件 '{image_file}' 已损坏或格式无效，跳过。")
            continue
    except Exception as e:
        print(f"读取图片 '{image_file}' 时出错: {e}")
        continue

    # 2.预处理
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # 3. 梯度计算:Sobel算子
    # 分别计算x和y方向的梯度
    grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # 合并梯度
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # 将梯度归一化到0-255
    gradient_image = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 4. 梯度图二值化
    # 使用OTSU自动阈值来分割边缘
    _, binary_gradient = cv2.threshold(gradient_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 形态学滤波:提取线性结构
    # 设计一个水平的长条形内核，用于连接和增强水平方向的划痕
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)) # 内核宽度25，高度1
    # 设计一个垂直的长条形内核，用于连接和增强垂直方向的划痕
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)) # 内核宽度1，高度25

    # 应用水平内核进行闭运算
    horizontal_morph = cv2.morphologyEx(binary_gradient, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)
    # 应用垂直内核进行闭运算
    vertical_morph = cv2.morphologyEx(binary_gradient, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)

    # 将水平和垂直方向提取出的划痕合并起来
    scratch_mask = cv2.add(horizontal_morph, vertical_morph)

    # 6. 缺陷定位与标记
    contours, _ = cv2.findContours(scratch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = original_image.copy()
    defect_count = 0

    for contour in contours:
        # 对划痕进行面积和形状筛选
        area = cv2.contourArea(contour)
        if area > 100:                   # 过滤掉太小的噪声
            defect_count += 1
            # 用一个旋转的最小外接矩形来更精确地框出划痕
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int_(box)
            cv2.drawContours(result_image, [box], 0, (0, 0, 255), 2)

    cv2.putText(result_image, f"Scratches Found: {defect_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 7. 结果
    window_title_suffix = f" - {image_file}"
    cv2.imshow('Original Image' + window_title_suffix, original_image)
    cv2.imshow('Binary Gradient' + window_title_suffix, binary_gradient)
    cv2.imshow('Scratch Mask' + window_title_suffix, scratch_mask)
    cv2.imshow('Final Result with Scratches Marked' + window_title_suffix, result_image)

    print(f"处理完成: '{image_file}'. 按任意键查看下一张图片...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("所有图片均已处理完毕。")
# 焊接气孔识别

# 考虑每张图的光照、对比度和噪声的不同，进行修改
# 修改参量：缺陷识别的参数“100”的全局阈值。考虑每张图片不同的环境量的“自适应参数”，分割图片，根据图片的区域计算得出合适的阈值

import cv2
import numpy as np
import os

# 图片文件夹路径
IMAGE_FOLDER = 'welding porosity'

# 检查是否存在
if not os.path.exists(IMAGE_FOLDER):
    print(f"错误：文件夹 '{IMAGE_FOLDER}' 不存在！")
    print(f"请在脚本旁边创建一个名为 '{IMAGE_FOLDER}' 的文件夹，并将你的.jpg图片放入其中。")
    # 创建一个空的文件夹，避免程序报错
    os.makedirs(IMAGE_FOLDER)

# 获取所有jpg图片的文件名
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.jpg') or f.endswith('.JPG')] # 增加了对大写JPG的支持

if not image_files:
    print(f"警告：在文件夹 '{IMAGE_FOLDER}' 中没有找到任何 .jpg 图片。")

# 循环处理每一张图片
for image_file in image_files:
    # 构建完整路径
    image_path = os.path.join(IMAGE_FOLDER, image_file)

    # 1.加载图片
    try:
        # 从构建的路径加载图片
        original_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if original_image is None:
            # 提供更详细的error
            print(f"警告：文件 '{image_file}' 已损坏或不是有效的图片格式，跳过此图片。")
            continue
    except Exception as e:
        print(f"读取或解码图片 '{image_file}' 时发生错误: {e}")
        continue

    # 2.预处理
    # 转换为灰度图
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊，去除噪声
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0) # 相较V1，将ksize调大

    # 3.缺陷分割

    # 使用自适应阈值分割（二值化）来分离缺陷，以应对光照不均的问题
    # 邻域大小 (blockSize): 必须是奇数，表示计算阈值的区域大小。这个值需要根据图片中缺陷的大小来调整。
    # 常数C (C): 从平均值或加权平均值中减去的常数。可以为正、零或负。

    # 可调
    blockSize = 21
    C = 10

    '''
    调整参数后效果也仍然不好
    最后调为21 - 10
    '''

    # 使用高斯自适应阈值
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, blockSize, C)


    # 使用形态学操作去除小的噪声点
    # 定义一个结构元素
    kernel = np.ones((3, 3), np.uint8)
    # 开运算：先腐蚀后膨胀，去除小的白色噪声点
    morph_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel, iterations=2)


    # 4.缺陷定位与标记
    # 在形态学处理后的图像上寻找轮廓
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个原始图片的copy显示结果
    result_image = original_image.copy()

    # 计算找到的缺陷数量
    defect_count = 0

    # 遍历所有找到的轮廓
    for contour in contours:
        # 过滤掉面积过小的轮廓，避免将噪声标记为缺陷
        if cv2.contourArea(contour) > 50:  # 50 面积阈值，相较V1变成了50
            # 统计缺陷数量
            defect_count += 1

            # 获取轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)

            # 用红色矩形框出缺陷
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 矩形边写 "Defect"
            cv2.putText(result_image, "Defect", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 左上角显示总数
    cv2.putText(result_image, f"Defects Found: {defect_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # 5.结果

    # 每个窗口的标题：当前处理的文件名
    window_title_suffix = f" - {image_file}"

    # 显示处理过程中的各个图像
    cv2.imshow('Original Image' + window_title_suffix, original_image)
    cv2.imshow('Adaptive Threshold' + window_title_suffix, thresh_image)
    cv2.imshow('Final Result with Defects Marked' + window_title_suffix, result_image)
    print(f"处理完成: '{image_file}'. 按任意键查看下一张图片...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("所有图片均已处理完毕。")
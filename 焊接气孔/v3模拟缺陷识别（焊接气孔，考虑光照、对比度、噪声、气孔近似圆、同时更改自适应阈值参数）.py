# 焊接气孔识别

'''
考虑每张图的光照、对比度和噪声的不同，但是仍然效果不好
核心问题：仅使用自适应阈值，会将大量焊缝自身的纹理（其灰度值也较低）误判为缺陷。
解决方案：在找到所有“疑似”缺陷的轮廓后，增加一个几何约束筛选。

具体：
1. 假设真实的焊接气孔在形状上更接近于圆形。
2. 计算每个检测出的轮廓的“圆度”。圆度：衡量一个形状接近圆的程度的指标，完美圆的圆度为1，细长、不规则形状的圆度远小于1。
3. 设定一个圆度阈值，只保留那些“足够圆”的轮廓作为最终的缺陷。
4. 由焊缝纹理产生的细长、不规则的轮廓就会因为圆度太低而被过滤掉，从而大大减少误检。
'''

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
        # 增加基于几何特征的筛选
        # 1.按面积过滤掉太小的噪声
        area = cv2.contourArea(contour)
        if area > 50:  # 50 面积阈值，相较V2没变

            # 2.计算轮廓周长
            perimeter = cv2.arcLength(contour, True)

            # 3. 计算圆度
            # check避免周长为零导致除法错误
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
            else:
                circularity = 0

            # 4. 根据圆度进行筛选
            # 圆度的阈值可以调整，通常0.6-0.9之间效果较好
            if circularity > 0.6:

                '''
                调整参数，但是大了效果过于保守，连原来能检测出来的也检测不出来
                小了效果和原先一样不准确，会识别很多噪声
                '''

                # 只有同时满足面积和圆度条件的轮廓，才被认为是真正的缺陷
                defect_count += 1

                # 获取轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)

                # 用红色矩形框出缺陷
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # 为了避免画面混乱，暂时不显示文字“defeect”

                # cv2.putText(result_image, "Defect", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


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
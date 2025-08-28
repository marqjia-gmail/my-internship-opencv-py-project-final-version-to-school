# 焊接气孔识别

# 核心问题：所有基于灰度阈值的分割方法，都无法有效区分颜色、纹理复杂的焊缝背景和真实缺陷。
#             手动设计的几何特征（圆度）效果并不好，反而使得效果更加保守。
# 新的解决路径：利用颜色特征进行分割，代替灰度阈值。并且放弃“圆度”，引入“高宽比”。

# 具体策略：
# 1. 颜色空间转换：将图片从RGB转换到HSV。HSV空间对光照变化不敏感，能更稳定地描述颜色。
# 2. 基于颜色的分割：假设焊接气孔的颜色主要是“黑色”或“深灰色”，这意味着其明度/亮度值（V）非常低。
# 3. 定义颜色范围：设定一个HSV颜色范围来定义“缺陷颜色”。对于黑色，我们主要限制明度（V）的上限，同时放宽色相（H）和饱和度（S）的范围。
# 4. 创建掩码：使用cv2.inRange()函数，生成一个二值掩码。图片中颜色在预设范围内的像素为白色（255），其余为黑色。
# 5. 后处理与筛选：在生成的掩码上，进行形态学操作和我们之前用过的几何特征筛选，得到最终结果。

import cv2
import numpy as np
import os

# 图片文件夹路径
IMAGE_FOLDER = 'welding porosity'

# 检查是否存在
if not os.path.exists(IMAGE_FOLDER):
    print(f"错误：文件夹 '{IMAGE_FOLDER}' 不存在！")
    print(f"请在脚本旁边创建一个名为 '{IMAGE_FOLDER}' 的文件夹，并将你的.jpg图片放入其中。")
    os.makedirs(IMAGE_FOLDER)

# 获取所有jpg图片的文件名
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.jpg') or f.endswith('.JPG')]

if not image_files:
    print(f"警告：在文件夹 '{IMAGE_FOLDER}' 中没有找到任何 .jpg 图片。")

# 循环处理每一张图片
for image_file in image_files:
    # 构建完整路径
    image_path = os.path.join(IMAGE_FOLDER, image_file)

    # 1.加载图片
    try:
        original_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if original_image is None:
            # 提供更详细的error
            print(f"警告：文件 '{image_file}' 已损坏或不是有效的图片格式，跳过此图片。")
            continue
    except Exception as e:
        print(f"读取或解码图片 '{image_file}' 时发生错误: {e}")
        continue

    # 2.预处理
    # 高斯模糊，去除噪声
    blurred_image = cv2.GaussianBlur(original_image, (7, 7), 0)

    # 3. 颜色空间转换
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # 4. 基于颜色的缺陷分割
    # 定义黑色的HSV颜色范围
    # 对于黑色，主要限制明度（V）的值。色相（H）和饱和度（S）可以范围大一些。
    # H: 0-180, S: 0-255, V: 0-255
    lower_black = np.array([0, 0, 0])      # HSV下限，一般不调整
    upper_black = np.array([180, 255, 60]) # HSV上限，主要调整V值(最后一个数字)

    # 根据颜色范围创建掩码
    mask = cv2.inRange(hsv_image, lower_black, upper_black)

    # 5. 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    # 使用闭运算填充缺陷内部的小洞
    morph_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 使用开运算去除小的噪声点
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_OPEN, kernel, iterations=1)


    # 6. 缺陷定位与筛选
    contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = original_image.copy()
    defect_count = 0

    for contour in contours:
        # 特征筛选
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0

        # 设定筛选阈值
        min_area = 20
        max_area = 5000
        min_aspect_ratio = 0.2
        max_aspect_ratio = 2.5 # 适当放宽宽高比

        if min_area < area < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            defect_count += 1
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)


    cv2.putText(result_image, f"Defects Found: {defect_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 7. 结果

    window_title_suffix = f" - {image_file}"

    # 显示处理过程中的各个图像
    cv2.imshow('Original Image' + window_title_suffix, original_image)
    cv2.imshow('Color-based Mask' + window_title_suffix, morph_mask)
    cv2.imshow('Final Result with Color Segmentation' + window_title_suffix, result_image)
    print(f"处理完成: '{image_file}'. 按任意键查看下一张图片...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("所有图片均已处理完毕。")
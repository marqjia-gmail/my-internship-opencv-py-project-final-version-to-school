# 焊接气孔识别
# 固定面积阈值，未考虑每张图的光照、对比度和噪声的不同

import cv2
import numpy as np
import os

# 图片文件夹路径
IMAGE_FOLDER = 'welding porosity'

# 检查文件是否存在
if not os.path.exists(IMAGE_FOLDER):
    print(f"错误：文件夹 '{IMAGE_FOLDER}' 不存在！")
    print(f"请在脚本旁边创建一个名为 '{IMAGE_FOLDER}' 的文件夹，并将你的.jpg图片放入其中。")
    # 创建一个空的文件夹，避免报错
    os.makedirs(IMAGE_FOLDER)

# 获取所有jpg图片的文件名
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.jpg')]

if not image_files:
    print(f"警告：在文件夹 '{IMAGE_FOLDER}' 中没有找到任何 .jpg 图片。")

# 循环处理每一张图片
for image_file in image_files:
    # 构建完整的路径
    image_path = os.path.join(IMAGE_FOLDER, image_file)

    # 1.加载图片
    try:
        # 从构建的路径加载图片
        original_image = cv2.imread(image_path)
        if original_image is None:
            # 提供更详细的error
            print(f"警告：无法正确读取图片 '{image_file}'，跳过此图片。")
            continue      # 跳过无法读取的，继续下一张
    except Exception as e:
        print(f"读取图片 '{image_file}' 时发生错误: {e}")
        continue

    # 2.预处理
    # 转换为灰度图
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊，去除噪声
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 3.缺陷分割
    # 使用阈值分割（二值化）来分离缺陷
    _, thresh_image = cv2.threshold(blurred_image, 50, 255, cv2.THRESH_BINARY_INV)

    '''
    参数从100上下调整，效果都不是很好，调整的尝试范围：50-128
    目前最好的：50
    '''

    # 使用形态学操作去除小的噪声点
    # 定义一个结构元素
    kernel = np.ones((5, 5), np.uint8)
    # 开运算：先腐蚀后膨胀，去除小的白色噪声点
    morph_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)

    # 4.缺陷定位与标记
    # 在形态学处理后的图像上寻找轮廓
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个原始图片的copy显示结果
    result_image = original_image.copy()

    # 计算缺陷数量
    defect_count = 0

    for contour in contours:
        # 过滤掉面积过小的轮廓，避免将噪声标记为缺陷
        if cv2.contourArea(contour) > 100:  #  100为 面积阈值
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

    # 显示处理过程中各个图像
    cv2.imshow('Original Image' + window_title_suffix, original_image)
    cv2.imshow('Final Result with Defects Marked' + window_title_suffix, result_image)
    print(f"处理完成: '{image_file}'. 按任意键查看下一张图片...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("所有图片均已处理完毕。")
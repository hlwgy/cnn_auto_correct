# 透视变换的例子源码 https://juejin.cn/post/7016096411301183502
import numpy as np
import cv2

# 固定点透视变换
def trans1():
    img = cv2.imread('img4_0_perspective.jpg')
    img_size = (img.shape[1], img.shape[0])
    # 左上，左下，右下，右上 
    src = np.float32([[92,105],[92,382],[447,350],[447,140]])
    dst = np.float32([[92,105],[92,382],[447,382],[447,105]])

    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, img_size, borderValue=(255,255,255))

    cv2.imshow('output', img)
    cv2.waitKey(0)

# 自动找点的透视变换
def trans2():
    img = cv2.imread('img4_0_perspective.jpg')
    # 转为灰度单通道 [[255 255],[255 255]]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化图像
    ret,img_b=cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
    # 图像出来内核大小，相当于PS的画笔粗细   
    kernel=np.ones((5,5),np.uint8)
    # 图像膨胀
    img_dilate=cv2.dilate(img_b,kernel,iterations=8)
    # 图像腐蚀
    img_erode=cv2.erode(img_dilate,kernel,iterations=3)
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(img_erode,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    # 绘制轮廓
    cv2.drawContours(img,contours,-1,(255,0,255),1)  

    # 一般会找到多个轮廓，这里因为我们处理成只有一个大轮廓
    contour = contours[0]
    # 每个轮廓进行多边形拟合
    approx = cv2.approxPolyDP(contour, 150, True)
    # 绘制拟合结果，这里返回的点的顺序是：左上，左下，右下，右上 
    cv2.polylines(img, [approx], True, (0, 255, 0), 2)

    # 寻找最小面积矩形
    rect = cv2.minAreaRect(contour)
    # 转化为四个点，这里四个点顺序是：左上，右上，右下，左下
    box = np.int0(cv2.boxPoints(rect))
    # 绘制矩形结果
    cv2.drawContours(img, [box], 0, (0, 66, 255), 2)

    img_size = (img.shape[1], img.shape[0])
    # 同一成一个顺序：左上，左下，右下，右上 
    src = np.float32(approx)
    dst = np.float32([box[0],box[3],box[2],box[1]])

    # 获取透视变换矩阵，进行转换
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, img_size, borderValue=(255,255,255))

    cv2.imshow('output', img)
    cv2.waitKey(0)
# %%
if __name__ == '__main__':

    print("请选择需要的代码解除注释，查看效果")
    # 固定点透视变换
    #trans1()
    # 自动找点的透视变换
    trans2()
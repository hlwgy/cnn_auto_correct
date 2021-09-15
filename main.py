# %%
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import PIL
import matplotlib.pyplot as plt
import os
import shutil
from numpy.core.records import array
from numpy.core.shape_base import block
import time
from collections import Counter
import cnn

# %%
# 整幅图片的Y轴投影
def img_y_shadow(img_b):
    ### 计算投影 ###
    (h,w)=img_b.shape
    # 初始化一个跟图像高一样长度的数组，用于记录每一行的黑点个数
    a=[0 for z in range(0,h)]
    # 遍历每一列，记录下这一列包含多少有效像素点
    for i in range(0,h):          
        for j in range(0,w):      
            if img_b[i,j]==255:     
                a[i]+=1  

    return a

# 图片获取文字块，传入投影列表，返回标记的数组区域坐标[[左，上，右，下]]
def img2rows(a,w,h):
    
    ### 根据投影切分图块 ### 
    inLine = False # 是否已经开始切分
    start = 0 # 某次切分的起始索引
    mark_boxs = []
    for i in range(0,len(a)):        
        if inLine == False and a[i] > 10:
            inLine = True
            start = i
        # 记录这次选中的区域[左，上，右，下]，上下就是图片，左右是start到当前
        elif i-start >5 and a[i] < 10 and inLine:
            inLine = False
            if i-start > 10:
                top = max(start-1, 0)
                bottom = min(h, i+1)
                box = [0, top, w, bottom]
                mark_boxs.append(box) 
                
    return mark_boxs

# 一行图片的X轴投影
def img_x_shadow(img_b):
    ### 计算投影 ###
    (h,w)=img_b.shape
    #初始化一个跟图像宽一样长度的数组，用于记录每一列的像素数量
    a =[0 for z in range(0,w)]
    # 遍历每一列，记录下这一列包含多少有效像素点
    for i in range(0,w):           
        for j in range(0,h):      
            if img_b[j,i]==255:
                a[i]+=1          
    return a

# 图片获取文字块，传入图片路径，返回标记的数组区域坐标[[左，上，右，下]]
def row2blocks(a, w, h):

    ### 根据投影切分图块 ### 
    inLine = False # 是否已经开始切分
    start = 0 # 某次切分的起始索引
    block_mark_boxs = [] # 切分的矩形区域坐标[左，上，右，下]

    for i in range(0,len(a)):
        # 如果还没有开始切，并且这列有效像素超过2个        
        if inLine == False and a[i] > 2:
            inLine = True # 标记为现在开始切块
            start = i # 标记这次切块的位置索引
        # 如果在切，并且已经超过10个，并且这次低于2个有效像素，说明遇到空白了
        elif i-start >10 and a[i] < 2 and inLine: 
            inLine = False # 标记不切了
            # 记录这次选中的区域[左，上，右，下]，上下就是图片，左右是start到当前
            left = max(start-1, 0)
            right = min(w, i+1)
            box = [left, 0, right, h]
            block_mark_boxs.append(box)  

    return block_mark_boxs


# 图片获取文字块，传入图片路径，返回标记的数组区域坐标[[左，上，右，下]]
def block2chars(a, w, h,row_top,block_left):

    ### 根据投影切分图块 ### 
    inLine = False # 是否已经开始切分
    start = 0 # 某次切分的起始索引
    char_mark_boxs = [] # 切分的矩形区域坐标[左，上，右，下]
    abs_char_mark_boxs = [] # 切分的矩形区域坐标[左，上，右，下]

    for i in range(0,len(a)):
        # 如果还没有开始切，并且这列有效像素超过1个        
        if inLine == False and a[i] > 0:
            inLine = True # 标记为现在开始切块
            start = i # 标记这次切块的位置索引
        # 如果在切，并且已经超过5个，并且这次低于2个有效像素，说明遇到空白了
        elif i-start >5 and a[i] < 1 and inLine: 
            inLine = False # 标记不切了
            # 记录这次选中的区域[左，上，右，下]，上下就是图片，左右是start到当前
            left = max(start-1, 0)
            right = min(w, i+1)
            box = [left, 0, right, h]
            char_mark_boxs.append(box)
            ads_box = [block_left+left, row_top,block_left+right, row_top+h]  
            abs_char_mark_boxs.append(ads_box)  

    return char_mark_boxs,abs_char_mark_boxs

# 裁剪图片
def cut_img(img, mark_boxs, is_square = False):

    img_items = []
    for i in range(0,len(mark_boxs)):
        img_org = img.copy()
        box = mark_boxs[i]
        img_item = img_org[box[1]:box[3], box[0]:box[2]]

        if is_square: # 是否转化为方形
            img_item = get_square_img(img_item)
        img_items.append(img_item)
    return img_items

# 展示投影图
def show_shadow(arr, direction = 'x'):

    a_max = max(arr)
    if direction == 'x': # x轴方向的投影
        a_shadow = np.zeros((a_max, len(arr)), dtype=int)
        for i in range(0,len(arr)):
            if arr[i] == 0:
                continue
            for j in range(0, arr[i]):
                a_shadow[j][i] = 255
    elif direction == 'y': # y轴方向的投影
        a_shadow = np.zeros((len(arr),a_max), dtype=int)
        for i in range(0,len(arr)):
            if arr[i] == 0:
                continue
            for j in range(0, arr[i]):
                a_shadow[i][j] = 255

    img_show_array(a_shadow)

# 展示图片，路径展示方式
def img_show_path(img_path):
    pil_im = Image.open(img_path)
    plt.imshow(pil_im)
    plt.show()

# 展示图片，数组展示方式
def img_show_array(a):
    plt.imshow(a)
    plt.show()

# 保存图片
def save_imgs(dir_name, imgs):
 
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name) 
    if not os.path.exists(dir_name):    
        os.makedirs(dir_name)

    img_paths = []
    for i in range(0,len(imgs)):
        file_path = dir_name+'/part_'+str(i)+'.png'
        cv2.imwrite(file_path,imgs[i])
        img_paths.append(file_path)
    
    return img_paths

# %%
def get_square_img(image):
    
    x, y, w, h = cv2.boundingRect(image)
    image = image[y:y+h, x:x+w]

    max_size = 18
    max_size_and_border = 24

    if w > max_size or h > max_size: # 有超过宽高的情况
        if w>=h: # 宽比高长，压缩宽
            times = max_size/w
            w = max_size
            h = int(h*times)
        else: # 高比宽长，压缩高
            times = max_size/h
            h = max_size
            w = int(w*times)
        # 保存图片大小
        image = cv2.resize(image, (w, h))


    xw = image.shape[0]
    xh = image.shape[1]

    xwLeftNum = int((max_size_and_border-xw)/2)
    xwRightNum = (max_size_and_border-xw) - xwLeftNum

    xhLeftNum = int((max_size_and_border-xh)/2)
    xhRightNum = (max_size_and_border-xh) - xhLeftNum
        
    img_large=np.pad(image,((xwLeftNum,xwRightNum),(xhLeftNum,xhRightNum)),'constant', constant_values=(0,0)) 
    
    return img_large

def divImg(img_path, save_file = False):

    thresh = 200

    img_o=cv2.imread(img_path,1) 

    # 读入图片
    img=cv2.imread(img_path,0) 
    (img_h,img_w)=img.shape
    # 二值化整个图，用于分行
    ret,img_b=cv2.threshold(img,thresh,255,cv2.THRESH_BINARY_INV) 

    # 计算投影，并截取整个图片的行
    img_y_shadow_a = img_y_shadow(img_b)
    row_mark_boxs = img2rows(img_y_shadow_a,img_w,img_h)
    # 切行的图片，切的是原图
    row_imgs = cut_img(img, row_mark_boxs)
    all_mark_boxs = []
    all_char_imgs = []
    # ===============从行切块======================
    for i in range(0,len(row_imgs)):
        row_img = row_imgs[i]
        (row_img_h,row_img_w)=row_img.shape
        # 二值化一行的图，用于切块
        ret,row_img_b=cv2.threshold(row_img,thresh,255,cv2.THRESH_BINARY_INV)
        kernel=np.ones((3,3),np.uint8)
        #图像膨胀6次
        row_img_b_d=cv2.dilate(row_img_b,kernel,iterations=6)
        img_x_shadow_a = img_x_shadow(row_img_b_d)
        block_mark_boxs = row2blocks(img_x_shadow_a, row_img_w, row_img_h)
        row_char_boxs = []
        row_char_imgs = []
        # 切块的图，切的是原图
        block_imgs = cut_img(row_img, block_mark_boxs)
        if save_file:
            b_imgs = save_imgs('imgs/cuts/row_'+str(i), block_imgs) # 如果要保存切图
            #print(b_imgs)
        # =============从块切字====================
        for j in range(0,len(block_imgs)):
            block_img = block_imgs[j]
            (block_img_h,block_img_w)=block_img.shape
            # 二值化块,因为要切字符图片了
            ret,block_img_b=cv2.threshold(block_img,thresh,255,cv2.THRESH_BINARY_INV)
            block_img_x_shadow_a = img_x_shadow(block_img_b)
            row_top = row_mark_boxs[i][1]
            block_left = block_mark_boxs[j][0]
            char_mark_boxs,abs_char_mark_boxs = block2chars(block_img_x_shadow_a, block_img_w, block_img_h,row_top,block_left)
            row_char_boxs.append(abs_char_mark_boxs)
            # 切的是二值化的图
            char_imgs = cut_img(block_img_b, char_mark_boxs, True)
            row_char_imgs.append(char_imgs)
            if save_file:
                c_imgs = save_imgs('imgs/cuts/row_'+str(i)+'/blocks_'+str(j), char_imgs) # 如果要保存切图
                #print(c_imgs)
        all_mark_boxs.append(row_char_boxs)
        all_char_imgs.append(row_char_imgs)


    return all_mark_boxs,all_char_imgs,img_o

# 计算数值并返回结果
def calculation(chars):
    cstr = ''.join(chars)
    
    result = ''

    if("=" in cstr): # 有等号
        str_arr = cstr.split('=')
        c_str = str_arr[0]
        r_str = str_arr[1]
        c_str = c_str.replace("×","*")
        c_str = c_str.replace("÷","/") 
        try:
            c_r = int(eval(c_str))
        except Exception as e:
            print("Exception",e)

        if r_str == "":
            result = c_r
        else:
            if str(c_r) == str(r_str):
                result = "√"
            else:
                result = "×"

    return result

# 绘制文本
def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("fonts/fangzheng_shusong.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# %%
def main(path, save = False):
    # 获取切图标注，切图图片，原图图图片
    all_mark_boxs,all_char_imgs,img_o = divImg(path,save)
    # 恢复模型，用于图片识别
    model = cnn.create_model()
    model.load_weights('checkpoint/char_checkpoint')
    class_name = np.load('checkpoint/class_name.npy')

    #遍历行
    for i in range(0,len(all_char_imgs)):
        row_imgs = all_char_imgs[i]
        # 遍历块
        for j in range(0,len(row_imgs)):
            block_imgs = row_imgs[j]
            block_imgs = np.array(block_imgs)
            # 图片识别
            results = cnn.predict(model, block_imgs, class_name)
            print('recognize result:',results)
            # 计算结果
            result = calculation(results)
            print('calculate result:',result)
            # 获取块的标注坐标
            block_mark = all_mark_boxs[i][j]
            # 获取结果的坐标，写在块的最后一个字
            answer_box = block_mark[-1]
            # 计算最后一个字的位置
            x = answer_box[2] 
            y = answer_box[3]
            iw = answer_box[2] - answer_box[0]
            ih = answer_box[3] - answer_box[1]
            # 计算字体大小
            textSize =  max(iw,ih)
            # 根据结果设置字体颜色
            if str(result) == "√":
                color = (0, 255, 0)
            elif str(result) == "×":
                color = (255, 0, 0)
            else:
                color = (192, 192,192)
            # 将结果写到原图上
            img_o = cv2ImgAddText(img_o, str(result), answer_box[2],  answer_box[1],color, textSize)
    # 将写满结果的原图保存
    cv2.imwrite('imgs/question_result.png', img_o)

# %%
if __name__ == '__main__':

    t = time.time()
    main('imgs/question.png', True)
    print(f'all take time:{time.time() - t:.4f}s')
# %%

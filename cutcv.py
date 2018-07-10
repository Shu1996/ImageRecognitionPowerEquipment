import os
import sys
import cv2
import numpy as np

def split_image(img_path, side, step, dstpath):
    # try:
    mPath, ext = os.path.splitext(img_path)
    if astrcmp(ext, ".jpeg") or astrcmp(ext, ".jpg"):
        img = cv2.imread(img_path)
        w, h, channel = img.shape

        s = os.path.split(img_path)
        if dstpath == '':
            dstpath = s[0]
        fn = s[1].split('.')
        basename = fn[0]
        ext = fn[-1]

        num = 0
        times = (w - side) // step + 1
        for r in range(times):
            for c in range(times):
                # box = (c * step, r * step, step*c+side, step*r+side)
                crop_img = img[r*step:r*step+side, c*step:c*step+side]
                cv2.imwrite(os.path.join(dstpath, basename + '_' + str(num) + '.' + ext), crop_img)
                # crop_img.save(os.path.join(dstpath, basename + '_' + str(num) + '.' + ext), ext)
                num = num + 1

def astrcmp(str1,str2):
    return str1.lower()==str2.lower()

if __name__ == '__main__':
    path = 'E:\\GradProject\\data\\unfit'
    dstpath = 'E:\\GradProject\\data\\unfit100_100'
    # side = int(input('请输入方块边长：'))
    # step = int(input('请输入滑动步长：'))
    side = 100
    step = 25
    for filename in os.listdir(path):
        split_image(filename, side, step, dstpath)

    print("*********图片处理完毕*********")

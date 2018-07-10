#-*- coding: UTF-8 -*-
import numpy as np
from PIL import Image
import os

def ImageToMatrix(filename):      #把图像转化为矩阵
    im=Image.open(filename)
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype='float32')
    return data

def getAllfile(dir,name):     #对dir下所有文件进行矩阵转换
    result=[]
    i=0
    for dirpath,dirs,files in os.walk(dir):
        for f in files:
            filename=os.path.join(dirpath,f)
            result.append(ImageToMatrix(filename))
            if i%500 == 0:
                print(i)
            i+=1
    result = np.array(result)
    print(result.shape)
    np.save(name+".npy",result)

dir = 'E:\\GradProject\\data\\insulators100_100'
name = 'Insulator'
getAllfile(dir,name)

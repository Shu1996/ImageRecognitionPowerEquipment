import os
import sys
from PIL import Image

def resize_image(img_path):
    # try:
    mPath, ext = os.path.splitext(img_path)
    if astrcmp(ext, ".jpeg") or astrcmp(ext, ".jpg"):
        img = Image.open(img_path)
        # (width, height) = img.size
        new_height = new_width

        if (width != new_width or height != new_height):
            out = img.resize((new_width, new_height), Image.ANTIALIAS)
            new_file_name = '%s%s' % (mPath, ext)
            out.save(new_file_name, quality=100)
            print("图片尺寸已修改")
        else:
            print("图片尺寸正确，未修改")
    else:
        print("非图片格式")
    # except Exception as e:
    #     print(e)

def astrcmp(str1,str2):
    return str1.lower()==str2.lower()

# def Py_Log(_string):
#     print("----"+_string.decode('utf-8')+"----")

def printFile(dirPath):
    print("file: " + dirPath)
    resize_image(dirPath)
    return True

if __name__ == '__main__':
    # path = "1.JPG"
    # path = 'E:\\python\\pics'
    path = 'E:\\GradProject\\data\\unfit'
    new_width = 300
    for filename in os.listdir(path):
        try:
            b = printFile(filename)
        except:
            print("Unexpected error:", sys.exc_info())
        print("*********图片处理完毕*********")

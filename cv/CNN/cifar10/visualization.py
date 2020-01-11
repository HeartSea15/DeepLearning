"""
    作者：Heart Sea
    功能：实现cifar10数据集的图片可视化
    版本：1.0
    日期：10/12/2019

"""
import numpy as np
import imageio
import os

# 为后面存放图片先建立文件夹
path_train = './训练集图片'
path_test = './测试集图片'
if not os.path.exists(path_train):
    os.makedirs(path_train)
if not os.path.exists(path_test):
    os.makedirs(path_test)

# 生成训练集5个batch图片,训练集5万张32*32*3的图片，每个batch1万张
# 生成测试集1个batch图片,测试集1万张32*32*3的图片，batch1万张
for j in range(1, 7):
    if j != 6:
        filename = 'd:/tmp/cifar10_data/cifar-10-batches-bin/data_batch_'+str(j)+'.bin'
    else:
        filename = 'd:/tmp/cifar10_data/cifar-10-batches-bin/test_batch.bin'

    bytestream = open(filename, "rb")   # rb 以二进制读模式打开

    # 1是占位字节（存放标签），一行（每个图片的记录长度）为32*32*3+1（存放像素）
    # read函数，从文件中读取的所有字节数，返回从字符串中读取的字节
    buf = bytestream.read(10000 * (1 + 32 * 32 * 3))
    bytestream.close()

    # numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)
    # buffer 任何暴露缓冲区借口的对象
    # dtype 返回数组的数据类型，默认为float
    # count 需要读取的数据数量，默认为-1，读取所有数据
    # offset 需要读取的起始位置，默认为0
    data = np.frombuffer(buf, dtype=np.uint8)    # 带有\n格式的一维数组

    data = data.reshape(10000, 1 + 32 * 32 * 3)  # 二维数组
    labels_images = np.hsplit(data, [1])   # 将data分成两个array,一组是data的第一列，另一组是剩下的所有列
    labels = labels_images[0].reshape(10000)             # 第一组为标签
    images = labels_images[1].reshape(10000, 32, 32, 3)  # 第二组为图像

    for i in range(0, 5):  # 输出一个batch的前5个图片，可以自行修改
        img = np.reshape(images[i], (3, 32, 32))  # 导出第一幅图
        img = img.transpose(1, 2, 0)
        if j != 6:
            picname = 'train'+str(i+(j-1)*10)+'.jpg'
            imageio.imwrite('训练集图片/' + picname, img)  # 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
        else:
            picname = 'test' + str(i + (j - 1) * 10) + '.jpg'
            imageio.imwrite('测试集图片/' + picname, img)
print("photo loaded.")
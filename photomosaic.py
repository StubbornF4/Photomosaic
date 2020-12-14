"""

为图片打上马赛克

"""

from PIL import Image
import numpy as np
import os
import argparse


def crop_image(image, dim):
    """
    裁剪图片
    image：输入的图片， dim：裁剪规格，裁剪成dim[0]*dim[1]块
    """
    m, n = dim[0], dim[1]
    width, height = image.size[0], image.size[1]
    # 剪裁过后小碎片的长和高
    width_crop, height_crop = int(width / n), int(height / m)
    images = []
    for row in range(m):
        for col in range(n):
            images.append(image.crop((col * width_crop, row * height_crop,
                                      (col + 1) * width_crop, (row + 1) * height_crop)))
    return images


def get_sub_images(directory):
    # 显示文件夹中的所有文件
    files = os.listdir(directory)
    images = []
    for file in files:
        try:
            filepath = os.path.abspath(os.path.join(directory + file))
            with open(filepath, 'rb') as fp:
                image = Image.open(fp)
                images.append(image)
                # force load image ?
                image.load()
        except:
            print('invalid file')
    return images


def resize_image(sub_images, input_image, input_dim):
    """裁剪替换图片"""
    images_cropped = []
    dim = (
        # 输入图片宽除以分割的列数（N）， 得到分割后碎片的宽
        int(input_image.size[0] / input_dim[1]),
        int(input_image.size[1] / input_dim[0])
    )

    for image in sub_images:
        image_cropped = image.crop((0, 0, dim[0], dim[1]))
        images_cropped.append(image_cropped)

    return images_cropped


def get_image_rgb(image):
    """

    输入PIL.Image, 将他转换成数组， 再转换成 （w*h，3）的数组求平均值
    返回求得平均值即为RGB值

    """
    image_array = np.array(image)
    # d=3
    w, h, d = image_array.shape
    # 转换成 w*h行3列的数组，再求平均值
    reshape_image = image_array.reshape(w * h, d)
    average_rgb = np.average(reshape_image, axis=0)
    return tuple(average_rgb)


def get_dif_rgb(rgb1, rgb2):
    """计算两个rgb的不同"""
    dif = (
            (rgb1[0] - rgb2[0]) * (rgb1[0] - rgb2[0]) +
            (rgb1[1] - rgb2[1]) * (rgb1[1] - rgb2[1]) +
            (rgb1[2] - rgb2[2]) * (rgb1[2] - rgb2[2])
    )
    return dif


def get_sub_images_final(input_images, sub_images):
    """传入输入图片裁剪后的图片， 替换图片rgb值，选择匹配的rgb，返回一个匹配索引列表"""
    # 得到替换图片的RGB
    sub_images_rgb = []
    for image in sub_images:
        sub_images_rgb.append(get_image_rgb(image))

    sub_images_final = []

    # 替换的图片碎片
    for image in input_images:
        image_rgb = get_image_rgb(image)
        index, min_index = 0, 0
        # dif 初始化为正无穷
        min_dif = float('inf')
        # image与 sub_images比较，得到最匹配的那个
        for sub_image_rgb in sub_images_rgb:
            dif = get_dif_rgb(image_rgb, sub_image_rgb)
            if dif < min_dif:
                min_dif = dif
                min_index = index
            index = index + 1

        sub_images_final.append(sub_images[min_index])

    return sub_images_final


def create_mosaic(images, dim):
    """输入替换的图片块，和规格得到马赛克图片"""
    # 图片有 m * n 块
    m, n = dim[0], dim[1]
    assert len(images) == m * n

    # 假设图片块之间大小不相等，求出最大的宽和高
    width = max([image.size[0] for image in images])
    height = max([image.size[1] for image in images])
    output_image = Image.new('RGB', (n*width, n*height))

    for index in range(len(images)):
        # index除以n列，得到此时的行数
        row = int(index/n)
        col = int(index - n*row)
        output_image.paste(images[index], (col*width, row*height))
    return output_image


def args_init():
    # parse arguments
    parser = argparse.ArgumentParser(description='Creates a photomosaic from input images')
    # add arguments
    parser.add_argument('--image', dest='target_image', required=True)
    parser.add_argument('--outfile', dest='outfile', required=False)

    args = parser.parse_args()
    return args


def main():
    sub_images_directory = './sub_images/'

    args = args_init()

    input_image = Image.open(args.target_image)
    input_dim = (128, 128)

    # 将输入的图片裁剪成dim（m * n）个小块
    images = crop_image(input_image, input_dim)
    sub_images = get_sub_images(sub_images_directory)

    # 将替换的图片 根据 images的尺寸 调整尺寸
    sub_images = resize_image(sub_images, input_image, input_dim)

    # 遍历输入图片小块的RGB，找到和替换图片最接近的,得到替换的索引
    sub_images_final = get_sub_images_final(images, sub_images)

    # 创造马赛克图片
    sub_image = create_mosaic(sub_images_final, input_dim)
    output_name = 'mosaic.png'
    if args.outfile:
        output_name = args.outfile
    sub_image.save(output_name, 'PNG')


if __name__ == '__main__':
    main()

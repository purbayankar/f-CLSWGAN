import torch
import os
from numpy import loadtxt
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


class images_set_train:
    def __init__(self, root, info_path, transform=None, loader=default_loader, class_num=200, split=150):
        image_info = open(info_path)
        images_path = []
        # images_index = []
        images_class_tmp = []
        for line in image_info.readlines():
            line_tmp = line.split()
            # image_index = line_tmp[0]
            image_path = line_tmp[1]
            image_path = os.path.join(root, image_path)
            images_path.append(image_path)
            images_class_tmp.append(os.path.split(image_path)[0])
            # images_index.append(image_index)

        # Get the ont-hot classes list for images.
        images_class = torch.zeros(len(images_class_tmp), class_num)
        col = 0
        for split_loc in range(len(images_class_tmp)):
            if split_loc == 0:
                images_class[split_loc][col] = 1
            else:
                if images_class_tmp[split_loc] == images_class_tmp[split_loc - 1]:
                    images_class[split_loc][col] = 1
                else:
                    if col < split - 1:
                        col += 1
                        images_class[split_loc][col] = 1
                    else:
                        break

        self.images_class_train = images_class[:split_loc, ]
        self.root = root
        self.images_path_train = images_path[:split_loc][:]
        # self.images_index = images_index
        self.transform = transform
        self.loader = loader
        self.split_loc = split_loc

    def get_split_loc(self):
        return self.split_loc

    def __getitem__(self, index):
        """
        :param index: the index of the images.
        :return: first item - index for that image, second item - the image data.
        """
        image_path = self.images_path_train[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, index, self.images_class_train[index]

    def __len__(self):
        return len(self.images_path_train)


class images_set_test:
    def __init__(self, root, info_path, transform=None, loader=default_loader, class_num=200, split=150):
        image_info = open(info_path)
        images_path = []
        # images_index = []
        images_class_tmp = []
        for line in image_info.readlines():
            line_tmp = line.split()
            # image_index = line_tmp[0]
            image_path = line_tmp[1]
            image_path = os.path.join(root, image_path)
            images_path.append(image_path)
            images_class_tmp.append(os.path.split(image_path)[0])
            # images_index.append(image_index)

        # Get the ont-hot classes list for images.
        images_class = torch.zeros(len(images_class_tmp), class_num)
        col = class_num - 1
        for split_loc in range(len(images_class_tmp) - 1, -1, -1):
            if split_loc == len(images_class_tmp) - 1:
                images_class[split_loc][col] = 1
            else:
                if images_class_tmp[split_loc] == images_class_tmp[split_loc + 1]:
                    images_class[split_loc][col] = 1
                else:
                    if col > split:
                        col -= 1
                        images_class[split_loc][col] = 1
                    else:
                        break

        self.images_class_test = images_class[split_loc + 1:][:]
        self.root = root
        self.images_path_test = images_path[split_loc + 1:][:]
        # self.images_index = images_index
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """
        :param index: the index of the images.
        :return: first item - index for that image, second item - the image data.
        """
        image_path = self.images_path_test[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, index, self.images_class_test[index]

    def __len__(self):
        return len(self.images_path_test)


def att_set(att_path, item_num, att_num):
    att_data_ori = loadtxt(att_path)
    att_data_ori = torch.from_numpy(att_data_ori)
    att_data = torch.zeros(item_num, att_num)
    for i in range(item_num):
        for j in range(att_num):
            att_data[i][j] = att_data_ori[i * att_num + j][2]
    return att_data

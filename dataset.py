# -*- coding:utf-8 -*-
from collections import Counter
import numpy as np
import easydict
import cv2
import os
from PIL import Image

FLAGS = easydict.EasyDict({"GT_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/annotations/ijrr_annotations_160523",
                           
                           "GT_mask_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/raw_aug_gray_mask"})

color_map = {(0, 50, 255): 255,
             (0, 0, 255): 255, 
             (255, 50, 0): 128,
             (255, 0, 0): 128}

if __name__ == "__main__":
    total_GT = os.listdir(FLAGS.GT_path)
    total_GT = [FLAGS.GT_path + "/" + data for data in total_GT]

    for i in range(len(total_GT)):
        img = cv2.imread(total_GT[i])
        img_copy = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)

        func = lambda b: b[:, :, 0] == 0 
        func2 = lambda g: g[:, :, 1] == 0 
        func3 = lambda r: r[:, :, 2] == 255

        func4 = lambda b: b[:, :, 0] == 0 
        func5 = lambda g: g[:, :, 1] == 50 
        func6 = lambda r: r[:, :, 2] == 255

        func7 = lambda b: b[:, :, 0] == 255 
        func8 = lambda g: g[:, :, 1] == 0 
        func9 = lambda r: r[:, :, 2] == 0

        func10= lambda b: b[:, :, 0] == 255 
        func11 = lambda g: g[:, :, 1] == 50 
        func12 = lambda r: r[:, :, 2] == 0

        func13 = lambda b: b[:, :, 0] == 0
        func14 = lambda g: g[:, :, 1] == 0 
        func15 = lambda r: r[:, :, 2] == 0

        func_test = lambda b: func(b) & func2(b) & func3(b) \
            | func4(b) & func5(b) & func6(b) \
            | func7(b) & func8(b) & func9(b) \
            | func10(b) & func11(b) & func12(b) \
            | func13(b) & func14(b) & func15(b)

        img_copy = np.where(func(img) & func2(img) & func3(img), 255, 0)
        img_copy = np.where(func4(img) & func5(img) & func6(img), 255, img_copy)
        img_copy = np.where(func7(img) & func8(img) & func9(img), 128, img_copy)
        img_copy = np.where(func10(img) & func11(img) & func12(img), 128, img_copy)
        #cv2.imshow("d", img_copy)
        #cv2.waitKey(0)

        a = np.reshape(img_copy, [img_copy.shape[0]*img_copy.shape[1],])

        if np.bincount(a)[128] == 0:
            print("No weed",total_GT[i].split("/")[-1])
        else:
            img_copy = np.where(func_test(img), img_copy, 128)
            img_copy = np.array(img_copy, dtype=np.uint8)
            img_copy2 = cv2.flip(img_copy, -1)
            name = total_GT[i].split('/')[-1].split("_")[3].split("frame")[1]
            name2 = total_GT[i].split('/')[-1].split("_")[3].split("frame")[1]
            if len(name) == 2:
                name = "rgb_000"+ name + ".png"
                name2 = "rgb_000"+ name2 + "_2" + ".png"
            elif len(name) == 1:
                name = "rgb_0000"+ name + ".png"
                name2 = "rgb_0000"+ name2 + "_2" + ".png"
            else:
                name = "rgb_00"+ name + ".png"
                name2 = "rgb_00"+ name2 + "_2" + ".png"

            cv2.imwrite(FLAGS.GT_mask_path + "/" + name, img_copy)
            cv2.imwrite(FLAGS.GT_mask_path + "/" + name2, img_copy2)

    #ori_img = os.listdir("D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/annotations/original_annotation/")
    #mask_img = os.listdir("D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/raw_rgb_mask/")
    #for i in range(len(mask_img)):
    #    name = mask_img[i].split("_")[3].split("frame")[1]
    #    name2 = mask_img[i].split("_")[3].split("frame")[1]
    #    if len(name) == 2:
    #        name = "rgb_000"+ name + ".png"
    #        name2 = "rgb_000"+ name2 + "_2" + ".png"
    #    elif len(name) == 1:
    #        name = "rgb_0000"+ name + ".png"
    #        name2 = "rgb_0000"+ name2 + "_2" + ".png"
    #    else:
    #        name = "rgb_00"+ name + ".png"
    #        name2 = "rgb_00"+ name2 + "_2" + ".png"

    #    for j in range(len(ori_img)):
    #        ori_name = ori_img[j]
    #        if name == ori_name:
    #            img = cv2.imread("D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/annotations/original_annotation/"+ ori_name)
    #            img2 = cv2.flip(img, -1)
    #            cv2.imwrite("D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/raw_rgb_img/" + name, img)
    #            cv2.imwrite("D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/raw_rgb_img/" + name2, img2)
        
    #data = os.listdir("D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/annotations/new_annotation/")
    #for i in range(len(data)):
    #    img = cv2.imread("D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/annotations/new_annotation/" + data[i], 0)

    #    img = np.array(img, dtype=np.uint8)
    #    img = np.reshape(img, [img.shape[0]*img.shape[1], ])
    #    print(np.all(img==128), data[i])
    #    if np.all(img==128) is True:
    #        print("No weed = {}".format(data[i]))

    #    if np.all(img==255) is True:
    #        print("No crop = {}".format(data[i]))

    

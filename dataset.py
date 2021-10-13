# -*- coding:utf-8 -*-
import numpy as np
import easydict
import cv2
import os

FLAGS = easydict.EasyDict({"GT_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/train_rgb_mask",
                           
                           "GT_mask_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/train_aug_gray_mask"})

color_map = {(0, 50, 255): 255,
             (0, 0, 255): 255, 
             (255, 50, 0): 128,
             (255, 0, 0): 128}

if __name__ == "__main__":
    total_GT = os.listdir(FLAGS.GT_path)
    total_GT = [FLAGS.GT_path + "/" + data for data in total_GT]

    for i in range(len(total_GT)):
        img = cv2.imread(total_GT[i])
        img_copy = np.zeros([img.shape[0], img.shape[1]], dtype=np.long)

        #for y in range(img.shape[0]):
        #    for x in range(img.shape[1]):
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

        img_copy = np.where(func(img) & func2(img) & func3(img), 255, 0)
        img_copy = np.where(func4(img) & func5(img) & func6(img), 255, img_copy)
        img_copy = np.where(func7(img) & func8(img) & func9(img), 128, img_copy)
        img_copy = np.where(func10(img) & func11(img) & func12(img), 128, img_copy)

        #print(np.bincount(np.reshape(img_copy, [img.shape[0]*img.shape[1]])))

        img_copy2 = cv2.flip(img_copy, -1)

        name = total_GT[i].split('/')[-1].split("_")[3].split("frame")[1]
        name2 = total_GT[i].split('/')[-1].split("_")[3].split("frame")[1]
        if len(name) == 2:
            name = "rgb_00"+ name + ".png"
            name2 = "rgb_00"+ name2 + "_2" + ".png"
        elif len(name) == 1:
            name = "rgb_000"+ name + ".png"
            name2 = "rgb_000"+ name2 + "_2" + ".png"
        else:
            name = "rgb_0"+ name + ".png"
            name2 = "rgb_0"+ name2 + "_2" + ".png"

        cv2.imwrite(FLAGS.GT_mask_path + "/" + name, img_copy)
        cv2.imwrite(FLAGS.GT_mask_path + "/" + name2, img_copy2)


        a = 0
    

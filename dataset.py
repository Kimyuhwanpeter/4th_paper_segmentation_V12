# -*- coding:utf-8 -*-
import numpy as np
import easydict
import cv2
import os

FLAGS = easydict.EasyDict({"GT_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/annotations/ijrr_annotations_160523",
                           
                           "GT_mask_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/annotations/new_annotation"})

if __name__ == "__main__":
    total_GT = os.listdir(FLAGS.GT_path)
    total_GT = [FLAGS.GT_path + "/" + data for data in total_GT]

    for i in range(len(total_GT)):
        img = cv2.imread(total_GT[i])
        img_copy = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        #a = np.where(img == [0, 0, 255], 255, img)

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                b = img[y,x][0]
                g = img[y,x][1]
                r = img[y,x][2]

                if b == 0 and g == 0 and r == 255:
                    img_copy[y,x] = 255

                elif b == 0 and g == 0 and r == 0:
                    img_copy[y,x] = 0

                else:
                    img_copy[y,x] = 128
       
        cv2.imwrite(FLAGS.GT_mask_path + "/" + total_GT[i].split('/')[-1], img_copy)


        a = 0
    

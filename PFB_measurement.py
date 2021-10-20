# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os

class Measurement:
    def __init__(self, predict, label, shape, total_classes):
        self.predict = predict
        self.label = label
        self.total_classes = total_classes
        self.shape = shape

    def MIOU(self):

        self.predict = np.reshape(self.predict, self.shape)
        self.label = np.reshape(self.label, self.shape)

        predict_count = np.bincount(self.predict, minlength=self.total_classes)
        label_count = np.bincount(self.label, minlength=self.total_classes)
         
        temp = self.total_classes * np.array(self.label, dtype="int") + np.array(self.predict, dtype="int")  # Get category metrics
    
        temp_count = np.bincount(temp, minlength=self.total_classes * self.total_classes)
        cm = np.reshape(temp_count, [self.total_classes, self.total_classes])
        cm = np.diag(cm)
    
        U = label_count + predict_count - cm
        U = np.delete(U, -1)
        cm_ = np.delete(cm, -1)

        out = np.zeros((2))
        miou = np.divide(cm_, U + 1e-7)
        crop_iou = miou[0]
        weed_iou = miou[1]
        miou = np.nanmean(miou)
        

        if weed_iou == float('NaN'):
            weed_iou = 0.
        if crop_iou == float('NaN'):
            crop_iou = 0.

        return miou, crop_iou, weed_iou

    def F1_score_and_recall(self):  # recall - sensitivity

        self.predict = np.reshape(self.predict, self.shape)
        self.label = np.reshape(self.label, self.shape)

        self.axis1 = np.where(self.label != 2)
        self.predict1 = np.take(self.predict, self.axis1)
        self.label1 = np.take(self.label, self.axis1)

        TP_func1 = lambda predict: predict[:] == 1
        TP_func2 = lambda label,predict: label[:] == predict[:]
        TP = np.where(TP_func1(self.predict1) & TP_func2(self.label1, self.predict1), 1, 0)
        TP = np.sum(TP, dtype=np.int32)

        TN_func1 = lambda predict: predict[:] == 0
        TN_func2 = lambda label,predict: label[:] == predict[:]
        TN = np.where(TN_func1(self.predict1) & TN_func2(self.label1, self.predict1), 1, 0)
        TN = np.sum(TN, dtype=np.int32)

        FN_func1 = lambda predict: predict[:] == 0
        FN_func2 = lambda label,predict: label[:] != predict[:]
        FN = np.where((FN_func1(self.predict1) & FN_func2(self.label1,self.predict1)), 1, 0)
        FN = np.sum(FN, dtype=np.int32)

        FP_func1 = lambda predict: predict[:] == 1
        FP_func2 = lambda label,predict: label[:] != predict[:]
        FP = np.where((FP_func1(self.predict1) & FP_func2(self.label1,self.predict1)), 1, 0)
        FP = np.sum(FP, dtype=np.int32)

        self.axis2 = np.where(self.label == 2)
        self.predict2 = np.take(self.predict, self.axis2)
        self.label2 = np.take(self.label, self.axis2)

        FN_func3 = lambda predict: predict[:] == 2
        FN_func4 = lambda label,predict: label[:] != predict[:]
        FN2 = np.where((FN_func3(self.predict2) & FN_func4(self.label2,self.predict2)), 1, 0)
        FN = np.sum(FN2, dtype=np.int32) + FN

        FP_func3 = (lambda predict: predict[:] == 1)
        FP_func4 = (lambda predict: predict[:] == 0)
        FP_func5 = (FP_func3(self.predict2) | FP_func4(self.predict2))
        FP_func6 = lambda label,predict: label[:] != predict[:]
        FP2 = np.where(FP_func5 & FP_func6(self.label2,self.predict2), 1, 0)
        FP = np.sum(FP2, dtype=np.int32) + FP


        TP_FP = (TP + FP) + 1e-7

        TP_FN = (TP + FN) + 1e-7

        out = np.zeros((1))
        Precision = np.divide(TP, TP_FP)
        Recall = np.divide(TP, TP_FN)

        Pre_Re = (Precision + Recall) + 1e-7

        F1_score = np.divide(2. * (Precision * Recall), Pre_Re)

        return F1_score, Recall

    def TDR(self): # True detection rate

        self.predict = np.reshape(self.predict, self.shape)
        self.label = np.reshape(self.label, self.shape)

        self.axis1 = np.where(self.label != 2)
        self.predict1 = np.take(self.predict, self.axis1)
        self.label1 = np.take(self.label, self.axis1)

        TP_func1 = lambda predict: predict[:] == 1
        TP_func2 = lambda label,predict: label[:] == predict[:]
        TP = np.where(TP_func1(self.predict1) & TP_func2(self.label1, self.predict1), 1, 0)
        TP = np.sum(TP, dtype=np.int32)

        TN_func1 = lambda predict: predict[:] == 0
        TN_func2 = lambda label,predict: label[:] == predict[:]
        TN = np.where(TN_func1(self.predict1) & TN_func2(self.label1, self.predict1), 1, 0)
        TN = np.sum(TN, dtype=np.int32)

        FN_func1 = lambda predict: predict[:] == 0
        FN_func2 = lambda label,predict: label[:] != predict[:]
        FN = np.where((FN_func1(self.predict1) & FN_func2(self.label1,self.predict1)), 1, 0)
        FN = np.sum(FN, dtype=np.int32)

        FP_func1 = lambda predict: predict[:] == 1
        FP_func2 = lambda label,predict: label[:] != predict[:]
        FP = np.where((FP_func1(self.predict1) & FP_func2(self.label1,self.predict1)), 1, 0)
        FP = np.sum(FP, dtype=np.int32)

        self.axis2 = np.where(self.label == 2)
        self.predict2 = np.take(self.predict, self.axis2)
        self.label2 = np.take(self.label, self.axis2)

        FN_func3 = lambda predict: predict[:] == 2
        FN_func4 = lambda label,predict: label[:] != predict[:]
        FN2 = np.where((FN_func3(self.predict2) & FN_func4(self.label2,self.predict2)), 1, 0)
        FN = np.sum(FN2, dtype=np.int32) + FN

        FP_func3 = (lambda predict: predict[:] == 1)
        FP_func4 = (lambda predict: predict[:] == 0)
        FP_func5 = (FP_func3(self.predict2) | FP_func4(self.predict2))
        FP_func6 = lambda label,predict: label[:] != predict[:]
        FP2 = np.where(FP_func5 & FP_func6(self.label2,self.predict2), 1, 0)
        FP = np.sum(FP2, dtype=np.int32) + FP

        TP_FP = (TP + FP) + 1e-7

        out = np.zeros((1))
        TDR = np.divide(FP, TP_FP)

        TDR = 1 - TDR

        return TDR
    
    def show_confusion(self):
        # TP - Red [255, 0, 0] 10, TN - 하늘색 [0, 255, 255] 20, FP - 분홍색 [255, 0, 255] 30, FN - 노랑 [255, 255, 0] 40

        self.predict = np.squeeze(self.predict, -1)

        color_map = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [236, 184, 199]])
        TP_func = lambda func:func[:] == 1
        output = np.where(TP_func(self.predict) & TP_func(self.label), 10, 0)  # get TP

        TN_func = lambda func:func[:] == 0
        output = np.where(TN_func(self.predict) & TN_func(self.label), 20, output)  # get TN
        
        FP_func = lambda func:func[:] == 1
        FP_func2 = lambda func:func[:] == 0
        FP_func3 = lambda func:func[:] == 2
        output = np.where(FP_func(self.predict) & (FP_func2(self.label)|FP_func3(self.label)), 30, output)  # get FP

        FN_func = lambda func:func[:] == 0
        FN_func2 = lambda func:func[:] == 1
        FN_func3 = lambda func:func[:] == 2
        output = np.where(FN_func(self.predict) & (FN_func2(self.label)|FN_func3(self.label)), 40, output)  # get FN

        output = np.expand_dims(output, -1)
        output_3D = np.concatenate([output, output, output], -1)
        temp_output_3D = output_3D
        
        output_3D = np.where(output == np.array([0, 0, 0], dtype=np.uint8), np.array([0, 0, 0], dtype=np.uint8), output_3D).astype(np.uint8)
        output_3D = np.where(output == np.array([10, 10, 10], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), output_3D).astype(np.uint8)
        output_3D = np.where(output == np.array([20, 20, 20], dtype=np.uint8), np.array([0, 255, 255], dtype=np.uint8), output_3D).astype(np.uint8)
        output_3D = np.where(output == np.array([30, 30, 30], dtype=np.uint8), np.array([255, 0, 255], dtype=np.uint8), output_3D).astype(np.uint8)
        output_3D = np.where(output == np.array([40, 40, 40], dtype=np.uint8), np.array([255, 255, 0], dtype=np.uint8), output_3D).astype(np.uint8)


        return output_3D, temp_output_3D

#import matplotlib.pyplot as plt

#if __name__ == "__main__":

    
#    path = os.listdir("D:/[1]DB/[5]4th_paper_DB/other/CamVidtwofold_gray/CamVidtwofold_gray/train/labels")

#    b_buf = []
#    for i in range(len(path)):
#        img = tf.io.read_file("D:/[1]DB/[5]4th_paper_DB/other/CamVidtwofold_gray/CamVidtwofold_gray/train/labels/"+ path[i])
#        img = tf.image.decode_png(img, 1)
#        img = tf.image.resize(img, [513, 513], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#        img = tf.image.convert_image_dtype(img, tf.uint8)
#        img = tf.squeeze(img, -1)
#        #plt.imshow(img, cmap="gray")
#        #plt.show()
#        img = img.numpy()
#        a = np.reshape(img, [513*513, ])
#        print(np.max(a))
#        img = np.array(img, dtype=np.int32) # void클래스가 정말 12 인지 확인해봐야함
#        #img = np.where(img == 0, 255, img)

#        b = np.bincount(np.reshape(img, [img.shape[0]*img.shape[1],]))
#        b_buf.append(len(b))
#        total_classes = len(b)  # 현재 124가 가장 많은 클래스수

#        #miou = MIOU(predict=img, label=img, total_classes=total_classes, shape=[img.shape[0]*img.shape[1],])
#        miou_ = Measurement(predict=img,
#                            label=img, 
#                            shape=[513*513, ], 
#                            total_classes=12).MIOU()
#        print(miou_)

#    print(np.max(b_buf))

# -*- coding:utf-8 -*-
from predefine_segmentation_model import *
from modified_deeplab_V3 import *
from PFB_measurement import Measurement
from random import shuffle, random
from keras_radam.training import RAdamOptimizer

import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 512,
                           
                           "label_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/annotations/new_annotation/",
                           
                           "image_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/annotations/original_annotation/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 200,

                           "total_classes": 3,

                           "ignore_label": 0,

                           "batch_size": 2,

                           "train": True})


total_step = len(os.listdir(FLAGS.image_path)) // FLAGS.batch_size
warmup_step = int(total_step * 0.6)
power = 1.

#lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
#    initial_learning_rate = FLAGS.lr,
#    decay_steps = total_step - warmup_step,
#    end_learning_rate = FLAGS.min_lr,
#    power = power
#)
#lr_schedule = LearningRateScheduler(FLAGS.lr, warmup_step, lr_scheduler)

#optim = RAdamOptimizer(total_steps=total_step*FLAGS.epochs,learning_rate=FLAGS.lr)
optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.9, beta_2=0.99)
color_map = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

def tr_func(image_list, label_list):

    h = tf.random.uniform([1], 1e-2, 30)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 30)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.random_brightness(img, max_delta=50.)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)

    return img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

def cal_loss(model, images, labels, objectiness, class_im_plain, ignore_label):

    with tf.GradientTape() as tape:

        
        batch_labels = tf.reshape(labels, [-1,])
        indices = tf.squeeze(tf.where(tf.not_equal(batch_labels, ignore_label)),1)
        batch_labels = tf.cast(tf.gather(batch_labels, indices), tf.float32)

        logits = run_model(model, images, True)
        raw_logits = tf.reshape(logits, [-1, FLAGS.total_classes-1])
        #print(raw_logits.shape)
        predict = tf.gather(raw_logits, indices)
        #print(predict.shape)

        class_im_plain = tf.reshape(class_im_plain, [-1,])
        class_im_plain = tf.cast(tf.gather(class_im_plain, indices), tf.float32)

        label_objectiness = tf.cast(tf.reshape(objectiness, [-1,]), tf.float32)
        logit_objectiness = raw_logits[:, -1]

        no_obj_indices = tf.squeeze(tf.where(tf.equal(tf.reshape(objectiness, [-1,]), 0)),1)
        no_logit_objectiness = tf.gather(logit_objectiness, no_obj_indices)
        no_obj_labels = tf.cast(tf.gather(label_objectiness, no_obj_indices), tf.float32)
        no_obj_loss = -(1. - no_obj_labels) * tf.math.log(1 - tf.nn.sigmoid(no_logit_objectiness) + 1e-7)
        no_obj_loss = tf.reduce_mean(no_obj_loss)

        obj_indices = tf.squeeze(tf.where(tf.not_equal(tf.reshape(objectiness, [-1,]), 0)),1)
        yes_logit_objectiness = tf.gather(logit_objectiness, obj_indices)
        yes_obj_labels = tf.cast(tf.gather(label_objectiness, obj_indices), tf.float32)
        obj_loss = -yes_obj_labels * tf.math.log(tf.nn.sigmoid(yes_logit_objectiness) + 1e-7)
        obj_loss = tf.reduce_mean(obj_loss)

        seg_loss = dice_loss(batch_labels, tf.squeeze(predict[:, 0:1], -1)) \
            + tf.nn.sigmoid_cross_entropy_with_logits(batch_labels, tf.squeeze(predict[:, 0:1], -1)) * class_im_plain
        seg_loss = tf.reduce_mean(seg_loss)
        
        loss = no_obj_loss + (seg_loss + obj_loss)
        #loss = seg_loss

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# yilog(h(xi;θ))+(1−yi)log(1−h(xi;θ))
def main():
    tf.keras.backend.clear_session()
    # 마지막 plain은 objecttines에 대한 True or False값 즉 (mask값이고), 라벨은 annotation 이미지임 (crop/weed)
    #model = PFB_model(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), OUTPUT_CHANNELS=FLAGS.total_classes-1)
    model = DeepLabV3Plus(FLAGS.img_size, FLAGS.img_size, 34)
    out = model.get_layer("activation_decoder_2_upsample").output
    out = tf.keras.layers.Conv2D(FLAGS.total_classes-1, (1,1), name="output_layer")(out)
    model = tf.keras.Model(inputs=model.input, outputs=out)
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        #elif isinstance(layer, tf.keras.layers.Conv2D):
        #    layer.kernel_regularizer = tf.keras.regularizers.l2(0.0005)

    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!")
    
    if FLAGS.train:
        count = 0
        
        image_dataset = os.listdir(FLAGS.image_path)
        label_dataset = os.listdir(FLAGS.label_path)
        # bonirob_2016-05-23-10-37-10_0_frame23_GroundTruth_color
        # rgb_00023
        train_img_dataset = []
        train_lab_dataset = []
        for i in range(len(image_dataset)):
            for j in range(len(label_dataset)):
                lab_name = label_dataset[j].split("_")[3]
                lab_name = lab_name.split("frame")[1]
                img_name = image_dataset[i].split(".")[0]
                img_name = img_name.split("_")[1]
                if int(img_name) == int(lab_name):
                    train_img_dataset.append(FLAGS.image_path + image_dataset[i])
                    train_lab_dataset.append(FLAGS.label_path + label_dataset[j])

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img_dataset, train_lab_dataset))
            shuffle(A)
            train_img_dataset, train_lab_dataset = zip(*A)
            train_img_dataset, train_lab_dataset = np.array(train_img_dataset), np.array(train_lab_dataset)

            train_ge = tf.data.Dataset.from_tensor_slices((train_img_dataset, train_lab_dataset))
            train_ge = train_ge.shuffle(len(train_img_dataset))
            train_ge = train_ge.map(tr_func)
            train_ge = train_ge.batch(FLAGS.batch_size)
            train_ge = train_ge.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(train_ge)
            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)  
                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == FLAGS.ignore_label, 2, batch_labels)    # 2 is void
                batch_labels = np.where(batch_labels == 255, 0, batch_labels)
                batch_labels = np.where(batch_labels == 128, 1, batch_labels)
                batch_labels = np.squeeze(batch_labels, -1)

                
                class_imbal_labels = batch_labels
                class_imbal_labels_buf = 0.
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels[i]
                    class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=FLAGS.total_classes)
                    class_imbal_labels_buf += count_c_i_lab
                class_imbal_labels_buf /= 2
                class_imbal_labels_buf = class_imbal_labels_buf[0:FLAGS.total_classes-1]
                class_imbal_labels_buf = (np.max(class_imbal_labels_buf / np.sum(class_imbal_labels_buf)) + 1 - (class_imbal_labels_buf / np.sum(class_imbal_labels_buf)))
                class_im_plain = np.where(batch_labels == 0, class_imbal_labels_buf[0], batch_labels)
                class_im_plain = np.where(batch_labels == 1, class_imbal_labels_buf[1], batch_labels)
                #a = np.reshape(class_im_plain, [FLAGS.batch_size*FLAGS.img_size*FLAGS.img_size, ])
                #a = np.array(a, dtype=np.int32)
                #a = np.bincount(a, minlength=3)
                objectiness = np.where(batch_labels == 2, 0, 1)  # 피사체가 있는곳은 1 없는곳은 0으로 만들어준것

                loss = cal_loss(model, batch_images, batch_labels, objectiness, class_im_plain, 2)
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step+1, tr_idx, loss))

                    logits = run_model(model, batch_images, False)
                    images = tf.nn.sigmoid(logits[:, :, :, 0:1])
                    for i in range(FLAGS.batch_size):
                        image = images[i]
                        image = np.where(image.numpy() >= 0.5, 1, 0)
                        
                        pred_mask_color = color_map[image]  # predict 이미지는 만들었음 이어서 코딩해야함!! 기억해!!
                        a = 0
                    

                count += 1

            tr_iter = iter(train_ge)
            miou = 0.
            for i in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)
                batch_labels = tf.squeeze(batch_labels, -1)
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    predict = run_model(model, batch_image, False) # type을 batch label과 같은 type으로 맞춰주어야함
                    predict = tf.nn.sigmoid(predict[0, :, :, 0:1])
                    predict = np.where(predict.numpy() >= 0.5, 1, 0)
                    #predict = tf.argmax(predict, -1)
                    #predict = predict.numpy()

                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)
                    ignore_label_axis = np.where(batch_label==2)   # 출력은 x,y axis로 나옴!
                    predict[ignore_label_axis] = 2

                    miou_ = Measurement(predict=predict,
                                       label=batch_label, 
                                       shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                       total_classes=FLAGS.total_classes).MIOU()

                    miou += miou_
            
            print("Epoch: {}, IoU = {}".format(epoch, miou / len(train_img_dataset)))
            # MIOU도 text로 쓰자 기억해!!!!
            # validataion , test miou도 실시간으로!!!
            # 다른 measurement도 같이!! 기억해!!!!

if __name__ == "__main__":
    main()

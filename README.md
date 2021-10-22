# 4th_paper_segmentation_V2
* Crop/weed 학습중 (아직 현재 최대 train miou 는 0.81 계속 증가하는 추세를 보임)
* 대략 20에폭?(실험상 20~30에폭에서 급격하게 miou가 증가하였음) 에서 learning decay 를 도입해보자 
* Test miou etc..(f1, recall) is state of the art (0.9038) (I can write paper !!!)
* 기본 backbone 모델은 기존과 동일, loss와 output을 완전 다르게 구성하였음 (box detection에 쓰이는 object loss 및 grid 추가)
* 자세한 모델설명은 다음주 월요일에..

## 테스트 결과 샘플 사진 및 성능

### BoniRob dataset
![f1](https://github.com/Kimyuhwanpeter/4th_paper_segmentation_V12/blob/main/Figure/figure1.png)
<br/>

![f2](https://github.com/Kimyuhwanpeter/4th_paper_segmentation_V12/blob/main/Figure/figure2.png)

#### BoniRob dataset with confusion image
![f8](https://github.com/Kimyuhwanpeter/4th_paper_segmentation_V12/blob/main/Figure/figure8.png)
<br/>
===============================================================================================
<br/>

### Rice seedling and weed dataset
![f3](https://github.com/Kimyuhwanpeter/4th_paper_segmentation_V12/blob/main/Figure/figure3.png)
<br/>

![f4](https://github.com/Kimyuhwanpeter/4th_paper_segmentation_V12/blob/main/Figure/figure4.png)

#### Rice seedling weed dataset with confusion image
![f9](https://github.com/Kimyuhwanpeter/4th_paper_segmentation_V12/blob/main/Figure/figure9.png)
<br/>
===============================================================================================
<br/>

### Carrot crop and weed (CWFID) dataset
![f5](https://github.com/Kimyuhwanpeter/4th_paper_segmentation_V12/blob/main/Figure/figure5.png)
<br/>

![f6](https://github.com/Kimyuhwanpeter/4th_paper_segmentation_V12/blob/main/Figure/figure6.png)

#### CWFID dataset with confusion image
![f10](https://github.com/Kimyuhwanpeter/4th_paper_segmentation_V12/blob/main/Figure/figure10.png)
<br/>

## 비교실험 결과 (FCN-8s는 학습중이므로 못넣었음)
![f7](https://github.com/Kimyuhwanpeter/4th_paper_segmentation_V12/blob/main/Figure/figure7.png)
<br/>

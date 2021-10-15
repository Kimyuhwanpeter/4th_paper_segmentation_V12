# 4th_paper_segmentation_V2
* Crop/weed 학습중 (아직 현재 최대 train miou 는 0.81 계속 증가하는 추세를 보임)
* 대략 20에폭?(실험상 20~30에폭에서 급격하게 miou가 증가하였음) 에서 learning decay 를 도입해보자 
* Test miou etc..(f1, recall) is state of the art (0.9056) (I can write paper !!!)
* 기본 backbone 모델은 기존과 동일, loss와 output을 완전 다르게 구성하였음 (box detection에 쓰이는 object loss 및 grid 추가)
* 자세한 모델설명은 다음주 월요일에..

## 테스트 결과 샘플 사진 및 성능
![f1](https://github.com/Kimyuhwanpeter/4th_paper_segmentation_V12/blob/main/Figure/figure1.png)
<br/>

![f2](https://github.com/Kimyuhwanpeter/4th_paper_segmentation_V12/blob/main/Figure/figure2.png)

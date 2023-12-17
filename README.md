# Vision Detection을 활용한 자율주행 야드 트랙터 Centering 솔루션

## 1. Introduction
기존에 있는 야드 트랙터는 크레인이 컨테이너를 운송할 때 트랙터가 정확한 위치에 있지 않으면 컨테이너 적재의 안전상의 문제가 발생함. 이러한 문제점을 해결하고자 트랙터의 위치가 정확한 라인 안에 위치하여 컨테이너가 위치를 정확하게 Centering 하기 위해 Vision Detection을 기반한 자율주행 야드트랙터 기술 개발을 위해 터틀봇3를 이용하여 시연하고자 함.
## 2. Specification
<img width="526" alt="스크린샷 2023-12-17 200408" src="https://github.com/Hseongha/Turtlebot_Centering/assets/145640813/065b97e9-36e2-489d-afeb-917b7cfafc7a">

</br>
    Product : Turtlebot3 Waffle pi
OS : Linux ubuntu 20.04 LTS
ROS : ROS1 noetic
언어 : Python
프레임워크 : Pycharm
 	Line Detection : OpenCV
  <img width="875" alt="스크린샷 2023-12-10 012530" src="https://github.com/Hseongha/Turtlebot_Centering/assets/145640813/7a7256ff-4a80-4d97-9d69-e7e1a5d2cedd">
 	Deep Learninig : YOLOv8
  <img width="748" alt="스크린샷 2023-12-16 160059" src="https://github.com/Hseongha/Turtlebot_Centering/assets/145640813/7cd90a2e-9a59-40a2-a49f-84370c1541d6">

<img width="478" alt="스크린샷 2023-12-16 235730" src="https://github.com/Hseongha/Turtlebot_Centering/assets/145640813/1021e147-e56f-42c3-bd23-486a05a53195">
    

## 3. 사용 기술

    Python, OpenCV, YOLOv8
    - 개발환경 : Colab, PyCharm

</br>


## 4. 구성원 역할
    황성하 - LaneDetection : 좌우 Centering
             YOLOv8n-seg 모델 학습
    황흥기 - Marker Detection : 전후 Centering
            Marker 틀어진 각도 측정
    정용준 - 터틀봇 제어 ROS 코드 개발
             기기 간 연결(PC- 폰 - 터틀봇)
             AI 모델 통합 및 터틀봇 연

## 5. 터틀봇 시뮬레이션 영상
   

https://github.com/Hseongha/Turtlebot_Centering/assets/145640813/fe34ec9f-9929-4b0d-9084-21a507b06dbd
## 6. KOSAIS팀_최종발표ppt.

[KOSAis팀_정용준, 황성하, 황흥기_최종발표자료.pptx](https://github.com/Hseongha/Turtlebot_Centering/files/13697082/KOSAis._._.pptx)







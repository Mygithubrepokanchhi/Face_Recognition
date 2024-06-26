# About:

- The face recognition and attendance-taking project successfully developed a robust system leveraging AI/ML technologies and Python-based frameworks. 
- Through face detection, feature extraction, recognition, attendance tracking, and a user-friendly interface, the system demonstrated high accuracy in identifying individuals and maintaining attendance records. 
- While showcasing significant potential for revolutionizing identity verification and attendance tracking processes, ongoing optimization efforts are needed to address challenges such as lighting variations and 
  data biases. 
- Overall, the project lays a foundation for future advancements in face recognition technology, offering practical solutions for diverse applications while emphasizing the importance of ethical deployment and 
  continual improvement.

1. **Steps to run project:**

- Install Python
   https://www.python.org/downloads/
- Install Pycharm(Python IDE)
   https://www.jetbrains.com/pycharm/download/?section=windows 
- After installing of Pycharm now go to settings of Pycharm
- Select Python Interpreter from settins in Pycharm.
- From there download some packages that is listed below.
   
   ![image](https://github.com/Mygithubrepokanchhi/Face_Recognition/assets/170111682/447384ca-36aa-45fe-81c2-a79e7bb41b3e)

    1. click
    2. cmake
    3. colorama
    4. dlib
    5. face-recognition
    6. face_recognition_models
    7. numpy
    8. opencv-python
    9. pillow
    10. pip
    11. setuptools
        
2. **After downloading all these packages now make two different name as ImageBasics and ImagesAttendence**
    
     ![image](https://github.com/Mygithubrepokanchhi/Face_Recognition/assets/170111682/8388812b-39c3-4c42-b588-e58ec1d13c93)

3. **Add training images in ImagesAttendence**
   
     - You should use your own dataset of images.
       
      ![image](https://github.com/Mygithubrepokanchhi/Face_Recognition/assets/170111682/f68a3e2e-714f-4a9a-9b07-4b5ac46473c6)

4. **Add testing images in ImageBasics**
      
      ![image](https://github.com/Mygithubrepokanchhi/Face_Recognition/assets/170111682/5cfa8b86-3bde-4c44-95d5-e5153d7b7054)

5. **Run Basic.py by just clicking on run arrow in Pycharm.**
    
    ![image](https://github.com/Mygithubrepokanchhi/Face_Recognition/assets/170111682/301cf07c-bc11-4504-ad21-e1b0bfcec8da)

    - In basic.py file after trained the model if we put the training and testing images of same people than it will generate Result True and it will also generate true percentage.

      ![True Result](https://github.com/Mygithubrepokanchhi/Face_Recognition/assets/170111682/2dc18cf3-1bf7-4780-9d5e-01dadad82fe7)
      

    - Or else if we will put the training and testing images of two different people than it will generate Result False and it will also generate False percentage.
   
      ![False result](https://github.com/Mygithubrepokanchhi/Face_Recognition/assets/170111682/065a7fc4-5089-4fda-bd01-49ae68728832)
      


6. **Follow the same process to run the AttendenceProject.py.**
    
    ![image](https://github.com/Mygithubrepokanchhi/Face_Recognition/assets/170111682/1b606ead-9f72-4e8c-b8aa-ce609bd60178)
    
7. **Attendence will be taken by using webcam of the particular device.**
    
    ![image](https://github.com/Mygithubrepokanchhi/Face_Recognition/assets/170111682/a7c04904-820d-4cd3-8bca-6ba4f886c2df)
    

    ![Screenshot 2024-05-03 143958](https://github.com/Mygithubrepokanchhi/Face_Recognition/assets/170111682/85b70417-e285-4a6b-b077-2bbea4f60705)

8. **After takin attendence it will store in Attendence.csv file with person's name and timing.**
    
    ![image](https://github.com/Mygithubrepokanchhi/Face_Recognition/assets/170111682/0a5115fd-076f-498d-bf4c-ac9b58e53b1b)

    ![image](https://github.com/Mygithubrepokanchhi/Face_Recognition/assets/170111682/456020b9-1878-4426-a31c-0b8e32bd4d27) 


    

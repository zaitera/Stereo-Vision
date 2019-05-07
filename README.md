# Stereo-Vision
> Abdullah_Zaiter__Ian_Moura  
      ├── README.md (This file)  
      ├── Abdullah_Zaiter__Ian_Moura.pdf  
      ├── /data  
      │   ├── /Middlebury  
      │   │   ├── /Jadeplant-perfect  
      │   │   │   ├── disparity.pgm  
      │   │   │   └── depth.png  
      │   │   └── /Motorcycle-perfect  
      │   │       ├── disparity.pgm  
      │   │       └── depth.png  
      │   └── /FurukawaPonce  
      ├── /relatorio  
      │   └── Latex source code of the report      
      ├── /src   
      |   └── main.py (principal source code)    
      |   └── utils.py (aux code with several relevant classes and methods)   
      |   └── MouseClick.py (a class for mouse tracking) 

### OpenCV version : 3.4.1
### Python 3

### Python modules used:
     - cv2  
     - numpy  
     - argparse
### *Requisites:*
      1- Calculation of disparity and depth map, and then using them to determine real world coordenates, this can be applied for two group of images, motor and plant.

      2- Calculation of disparity and depth for another group of image where there is a rotation between the two images.

      3- Determing the minimum box dimensions in which the object of requisite 2 can fit, this can be done clicking on the image two times to measure each dimension.
### To evaluate requisite 1 run this command, where i parameter can be m or p (m for the motor images and p for the plant images):
>python3 ./src/main.py --r 1 --i m OR python3 ./src/pd3.py --r 1 --i m

### To evaluate requisite 2 run this command:
>python3 ./src/main.py --r 2

### To evaluate requisite 3 run this command (measuring box size):
>python3 ./src/main.py --r 3



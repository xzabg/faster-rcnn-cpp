## Object Detection
This is an implementation of the Faster R-CNN in C++ on Windows  
  
The project includes:  
- The camera capturing the image  
- Detecting persons in the pictures  
- Saving the pictures and sending them to the server and the smart phone  

### Requirements
- caffe
- OpenCV

### Getting Started
- Create a solution with Visual Studio and include the source code  
- Download the [GeTui APP](http://docs.getui.com/download.html) and use the value in the app to replace the "appId", "appKey", "masterSecret" and "cid" in GETUI.h
- Build the solution
- Run ObjectDetect.cpp

### Results
The speed of the program achieves 13fps on GeForce GTX 1060

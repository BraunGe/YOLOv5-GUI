# YOLOv5-GUI
A GUI for YOLOv5, support all the 11  inference formats that YOLOv5 supports.
The detect program is based on YOLOv5 v6.1.
Run the `detect6.py` to start the GUI.

1. Compared with other YOLOv5's GUI, this one has a unique advantage, this GUI can support all the 11  inference formats that YOLOv5 supports. That will be great for embeded device deploy with GUI. 

| Format |Model  |
|--|--|
| PyTorch | yolov5s. pt |
| TorchScript | yolov5s.torchscript |
| ONNX | yolov5s.onnx |
| OpenVINO | yolov5s_openvino_model/ |
| TensorRT | yolov5s.engine |
| CoreML | yolov5s.mlmodel |
| TensorFlow SavedModel | yolov5s_saved_model/ |
| TensorFlow GraphDef | yolov5s.pb |
| TensorFlow Lite | yolov5s.tflite |
| TensorFlow Edge TPU | yolov5s_edgetpu.tflite |
| TensorFlow.js | yolov5s_web_model/ |
| PaddlePaddle | yolov5s_paddle_model/ |

 2. This GUI can support all the inference source when using any one in the 11  inference formats. 
Support source:
- 
       source 0  # webcam 
              img.jpg  # image
              vid.mp4  # video
              path/  # directory
              path/*.jpg  # glob
              'https://youtu.be/Zgi9g1ksQHc'  # YouTube
              'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

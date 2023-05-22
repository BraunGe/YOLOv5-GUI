# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
from multiprocessing.resource_sharer import stop
import os
import time
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import ctypes

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.Qt import *
from main_win.mainUI5 import Ui_MainWindow
from time import ctime, sleep 

class DetThread(QThread):

    send_img = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.jump_out = False  
        self.is_continue = True  
        self.is_continue2 = True 
        self.is_continue3 = True
        self.detect_picture = True
        self.detect_video = False
        self.weights = ROOT / 'bestn6_nc50.pt'
        self.detect_source = 0
        self.imgsz = [640]
        self.data = ROOT / 'data/TT100K6.yaml'
        self.opt3 = DetThread.parse_opt3(self)

    def parse_opt3(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=self.weights, help='model path(s)')
        parser.add_argument('--source', type=str, default=self.detect_source, help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default=self.data, help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=self.imgsz, help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.25, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        opt3 = parser.parse_args()
        opt3.imgsz *= 2 if len(opt3.imgsz) == 1 else 1  # expand
        print_args(FILE.stem, opt3)
        return opt3
    
    @torch.no_grad()
    def run(self,
            weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
            source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=[640, 640],  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):
        prev_frame_time = 0
        new_frame_time = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            if self.jump_out:
                break

            if not self.is_continue:
                continue

            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
#                        s += f"{n} {names[int(c)]}{'s' * (n > 1)},"  # add to string
                        s += f"\n{n} {names[int(c)]}{'s' * (n > 1)},"  # add to string
                    self.send_statistic.emit(s)

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
#                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                fps = round(fps,2) 
                fps = str(fps)
                FPS = "FPS:"
                FPS0 = FPS+fps
                im0=cv2.resize(im0,(1080,720),interpolation=cv2.INTER_CUBIC)
                cv2.putText(im0, FPS0, (7, 30), font,1, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.waitKey(1)
                self.send_img.emit(im0)
                if self.detect_picture:
                    while True:
                        cv2.waitKey(0)
                        if not self.is_continue3:
                            self.is_continue3 = True
                            break
                if self.detect_video:
                    while True:
                        cv2.waitKey(0)
                        if self.is_continue2:
                            break


    #            CUi_MainWindow.show_camera(im0)

    #            cv2.imshow(str(p), im0)
    #            cv2.waitKey(1)  # 1 millisecond


                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    def start3(self):
        opt3 = DetThread.parse_opt3(self)
        check_requirements(exclude=('tensorboard', 'thop'))
        print(self.detect_source)
        DetThread.run(self,**vars(opt3)) 
    
class CUi_MainWindow(QMainWindow, Ui_MainWindow): #继承于UI父类

    returnSignal = pyqtSignal()
    
    def __init__(self, parent=None):
        super(CUi_MainWindow, self).__init__(parent)
        self.timer_camera = QTimer() #初始化定时器
        self.cap = cv2.VideoCapture() #初始化摄像头
        self.CAM_NUM = 0
        self.setupUi(self)
        self.slot_init()
        global window, count
        window = self.videolabel
        self.count = 0
        self.nextcount = 0

        self.comboBox.clear()
        self.pt_list = os.listdir(ROOT)
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize(x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)

        self.onnx_list = os.listdir(ROOT)
        self.onnx_list = [file for file in self.onnx_list if file.endswith('.onnx')]
        self.onnx_list.sort(key=lambda x: os.path.getsize(x))
        self.comboBox.addItems(self.onnx_list)

        self.comboBox_2.clear()
        self.video_list = os.listdir(ROOT /'data')
        self.video_list = [file for file in self.video_list if file.endswith('.mp4')]
        self.comboBox_2.clear()
        self.comboBox_2.addItems(self.video_list)

        # yolov5 thread
        self.det_thread = DetThread()
        self.det_thread.send_img.connect(lambda x: self.show_image(x))   
        self.det_thread.send_statistic.connect(self.show_statistic)
        
    def slot_init(self):
        self.pushButton.clicked.connect(self.restart)
        self.pushButton_2.clicked.connect(self.run_or_continue)
        self.pushButton_3.clicked.connect(self.run_or_continue2)
        self.pushButton_4.clicked.connect(self.next)
        self.pushButton_5.clicked.connect(self.quit)
        self.pushButton_6.clicked.connect(self.run_or_continue3)
        self.pushButton_7.clicked.connect(self.run_or_continue4)
        self.radioButton_6.setChecked(True)

    #    self.pushButton.clicked.connect(self.btn)  

#    def show_image(self):
#        show = cv2.cvtColor(self, cv2.COLOR_BGR2RGB)
#        showImage = QImage(show.data, show.shape[1],show.shape[0],QImage.Format_RGB888)
#        window.setScaledContents(True)
#        window.setPixmap(QPixmap.fromImage(showImage))

    def show_image(self,img_src):
        frame = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
        img = QImage(frame.data, frame.shape[1],frame.shape[0],QImage.Format_RGB888)
    #    window.setScaledContents(True)
        window.setPixmap(QPixmap.fromImage(img))

    def search_weights(self):
        pt_list = os.listdir(ROOT)
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def choose_weights(self):
        self.det_thread.weights = str(self.comboBox.currentText())
        print(self.det_thread.weights)

    def choose_videos(self):
        self.det_thread.detect_source = (ROOT / 'data'/ str(self.comboBox_2.currentText()))
        return self.det_thread.detect_source

    def imagesize(self):
        if self.count==0:
            if self.radioButton_6.isChecked():
                self.det_thread.imgsz = [640]
            elif self.radioButton_5.isChecked():
                self.det_thread.imgsz = [1280]
            else:
                pass

    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.count ==0:
            if self.pushButton_2.isChecked():
                self.det_thread.is_continue = True
                if not self.det_thread.isRunning():
                    CUi_MainWindow.choose_weights(self)
                    CUi_MainWindow.imagesize(self)
                    self.count = 1
                    self.det_thread.detect_source = 0
                    self.det_thread.detect_picture = False
                    self.det_thread.start3()
                else:
                    self.det_thread.is_continue = False
        else:
            if self.pushButton_2.isChecked():
                self.det_thread.is_continue = True
            else:
                self.det_thread.is_continue = False

    def run_or_continue2(self):
        self.det_thread.jump_out = False
        if self.count ==0:
            if self.pushButton_3.isChecked():
                self.det_thread.is_continue2 = True
                if not self.det_thread.isRunning():
                    CUi_MainWindow.choose_weights(self)
                    CUi_MainWindow.imagesize(self)
                    self.count = 1
                    self.det_thread.detect_source = 'data/images'
                    self.det_thread.detect_picture = True
                    self.det_thread.start3()
                else:
                    self.det_thread.is_continue2 = False
        else:
            if self.pushButton_3.isChecked():
                self.det_thread.is_continue2 = True
            else:
                self.det_thread.is_continue2 = False

    def run_or_continue3(self):
        self.det_thread.jump_out = False
        if self.count ==0:
            if self.pushButton_6.isChecked():
                self.det_thread.is_continue = True
                if not self.det_thread.isRunning():
                    CUi_MainWindow.choose_weights(self)
                    CUi_MainWindow.imagesize(self)
                    self.count = 1
                    self.det_thread.detect_picture = False
                    self.det_thread.detect_source = self.lineEdit.text()
                    self.det_thread.start3()
                else:
                    self.det_thread.is_continue = False
        else:
            if self.pushButton_6.isChecked():
                self.det_thread.is_continue = True
            else:
                self.det_thread.is_continue = False

    def run_or_continue4(self):
        self.det_thread.jump_out = False
        if self.count ==0:
            if self.pushButton_7.isChecked():
                self.det_thread.is_continue2 = True
                if not self.det_thread.isRunning():
                    CUi_MainWindow.choose_weights(self)
                    CUi_MainWindow.imagesize(self)
                    CUi_MainWindow.choose_videos(self)
                    self.count = 1
                    self.det_thread.detect_picture = False
                    self.det_thread.detect_video = True
                    self.det_thread.start3()
                else:
                    self.det_thread.is_continue2 = False
        else:
            if self.pushButton_7.isChecked():
                self.det_thread.is_continue2 = True
            else:
                self.det_thread.is_continue2 = False


    def next(self):
        self.det_thread.is_continue3 = False

    def show_statistic(self, statistic_dic):
        self.label.clear()
        self.label.setText(statistic_dic)

    def restart(self):
        self.timer_camera.stop()
        self.det_thread.jump_out = True
        window.clear()
        self.det_thread.quit()
        p = sys.executable
        os.execl(p,p,*sys.argv)

    def quit(self):
        sys.exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)                           #使用sys新建一个应用（Application）对象
    timer_camera = QTimer()  # 初始化定时器
    MainWindow = CUi_MainWindow()                             #新建一个Qt中QMainWindow()类函数
    #ui = mainUI.Ui_MainWindow()                            #定义ui，与我们设置窗体绑定
    #ui.setupUi(MainWindow)                                 #为MainWindow绑定窗体
    MainWindow.show()                                      #将MainWindow窗体进行显示
    sys.exit(app.exec_())                                  #进入主循环，事件开始处理，接收由窗口触发的事件

import cv2
import sys  
import time
import os  
import numpy as np  
from tqdm import tqdm  
from pathlib import Path  
from glob import glob  
from detect_with_API import detectapi
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,QShortcut
from PyQt5.QtGui import QPixmap, QImage ,QIcon,QKeySequence
from PyQt5.QtCore import Qt,QUrl,QTimer



IMG_FORMATS = '(*.bmp *.dng *.jpeg *.jpg *.mpo *.png *.tif *.tiff *.webp *.pfm)'  # include image suffixes
IMG_END =('.jpg','.png','.jpeg','.JPG','.PNG','.JPEG','.BMP','.DNG','.TIFF','.TIF','.WEBP')  # include image suffixes
VID_FORMATS = '(asf, *avi *gif *m4v *mkv *mov *mp4 *mpeg *mpg *ts *wmv)'  # include video suffixes
ROOT_DIR = Path(__file__).resolve().parent  


class GrayImage:
    def __init__(self) -> None:
        self.save_img = np.full((640,640,3), 114, dtype=np.float32)
        cv2.imwrite(ROOT_DIR /'gray_img.jpg', self.save_img)
        self.gray_img = cv2.imread(ROOT_DIR /'gray_img.jpg')


class MainWindow(QMainWindow):  
    def __init__(self,weight=ROOT_DIR / 'weights/yolov5n.pt'):  
        super().__init__()  
        self.model =detectapi(weight)
        self.frame = []
        self.count_number =0
        self.gray_img =GrayImage().gray_img
        self.__init_window()

    def __init_window(self):  
        self.setWindowTitle("Image Viewer and Detector")  
        self.setGeometry(100, 100, 1000, 600) 
        self.original_image_label = QLabel(self)  
        self.detected_image_label = QLabel(self)  
        self.button_label = QLabel(self)
        # Layouts 
        text_h_layout = QHBoxLayout()  # Horizontal layout for text labels 
        h_layout = QHBoxLayout()  # Horizontal layout for images  
        self.button_label =QHBoxLayout()
        v_layout = QVBoxLayout()  # Vertical layout for passentire window 
        # 设置按钮load_f
        self.load_file_button = QPushButton('加载目录')
        self.before_button = QPushButton('上一张')
        self.next_button = QPushButton('下一张')
        self.load_file_button.setFixedSize(190, 50)  # 设置删除按钮大小
        self.before_button.setFixedSize(190, 50)  # 设置下一张按钮大小
        self.next_button.setFixedSize(190, 50)  # 设置下一张按钮大小

        self.load_file_button.clicked.connect(self.load_file)  
        self.before_button.clicked.connect(self.before_image)  
        self.next_button.clicked.connect(self.next_image)  

        # button(水平布局)
        self.button_label.addWidget(self.load_file_button)
        self.button_label.addWidget(self.before_button)
        self.button_label.addWidget(self.next_button)
        # Add images to horizontal layout  （水平布局）  
        text_h_layout.addWidget(QLabel("原始图", self)) 
        text_h_layout.addWidget(QLabel("检测图", self))   
        h_layout.addWidget(self.original_image_label)  
        h_layout.addWidget(self.detected_image_label)  

        # Add horizontal layout and buttons to vertical layout（最终垂直布局）  
        v_layout.addLayout(text_h_layout)  
        v_layout.addLayout(h_layout)  
        v_layout.addLayout(self.button_label)

        # Container for vertical layout  
        container = QWidget()  
        container.setLayout(v_layout)  
        self.setCentralWidget(container)  
        self.backgroud_img()


    def backgroud_img(self):  
        pixmap =self.show_qt(self.gray_img)
        self.original_image_label.setPixmap(pixmap)
        self.original_image_label.setAlignment(Qt.AlignCenter)  

        self.detected_image_label.setPixmap(pixmap)
        self.detected_image_label.setAlignment(Qt.AlignCenter)

    def load_file(self):  
        file_path = QFileDialog.getExistingDirectory(self, "Open", "")  
        if file_path:  
            file_list =sorted(glob(os.path.join(file_path, '*.*')))
            self.frame =[img for img in file_list if img.endswith(IMG_END)]
            img = cv2.imread(self.frame[self.count_number])
            self.ori_img_show(img)
            self.detect_img_show(img)   

    def before_image(self):  
        if self.count_number <= 0:
            self.count_number = 0
            img = cv2.imread(self.frame[self.count_number])
            self.ori_img_show(img)
            self.detect_img_show(img) 
        else:
            self.count_number -= 1
            img = cv2.imread(self.frame[self.count_number])
            self.ori_img_show(img)
            self.detect_img_show(img)   

    def next_image(self):  
        if self.count_number >= len(self.frame)-1:
            self.count_number = len(self.frame)-1
            img = cv2.imread(self.frame[self.count_number])
            self.ori_img_show(img)
            self.detect_img_show(img) 
        else:
            self.count_number += 1
            img = cv2.imread(self.frame[self.count_number])
            self.ori_img_show(img)
            self.detect_img_show(img)   

    def ori_img_show(self,img):  
        ori_image =self.ResziePadding(img)
        pixmap =self.show_qt(ori_image)
        self.original_image_label.setPixmap(pixmap)
        self.original_image_label.setAlignment(Qt.AlignCenter) 

    def detect_img_show(self,ori_image):  
        s =time.time()
        img = self.detect_img(ori_image)
        img =self.ResziePadding(img)
        fps = int(1/(time.time()-s))
        img =cv2.putText(img,f"FPS:{fps}",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        pixmap =self.show_qt(img)
        self.detected_image_label.setPixmap(pixmap)
        self.detected_image_label.setAlignment(Qt.AlignCenter) 

    def ResziePadding(self,img, fixed_side=640):	

        h, w = img.shape[:2]
        scale = max(w, h)/float(fixed_side)   # 获取缩放比例
        new_w, new_h = int(w/scale), int(h/scale)
        resize_img = cv2.resize(img, (new_w, new_h))    # 按比例缩放
        
        # 计算需要填充的像素长度
        if new_w % 2 != 0 and new_h % 2 == 0:
            top, bottom, left, right = (fixed_side - new_h) // 2, (fixed_side - new_h) // 2, (fixed_side - new_w) // 2 + 1, (
                fixed_side - new_w) // 2
        elif new_w % 2 == 0 and new_h % 2 != 0:
            top, bottom, left, right = (fixed_side - new_h) // 2 + 1, (fixed_side - new_h) // 2, (fixed_side - new_w) // 2, (
                fixed_side - new_w) // 2
        elif new_w % 2 == 0 and new_h % 2 == 0:
            top, bottom, left, right = (fixed_side - new_h) // 2, (fixed_side - new_h) // 2, (fixed_side - new_w) // 2, (
                fixed_side - new_w) // 2
        else:
            top, bottom, left, right = (fixed_side - new_h) // 2 + 1, (fixed_side - new_h) // 2, (fixed_side - new_w) // 2 + 1, (
                fixed_side - new_w) // 2

        # 填充图像
        pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])
        
        return pad_img  
  
    def detect_img(self,img):
        results,names = self.model.detect([img])
        if len(results):
            return results[-1][0]

        else:
            return img

    def show_qt(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        bytes_per_line = 3 * width
        qt_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888) 
        pixmap = QPixmap.fromImage(qt_image)
        return pixmap


    def set_shortcuts(self):  
        # 快捷键 "Ctrl+Left" 用于 "上一张"  
        shortcut_prev = QShortcut(QKeySequence("Left"), self)  
        shortcut_prev.activated.connect(self.before_image)  
  
        # 快捷键 "Ctrl+Right" 用于 "下一张"  
        shortcut_next = QShortcut(QKeySequence("Right"), self)  
        shortcut_next.activated.connect(self.next_image) 

        # copy_file =QShortcut(QKeySequence("Ctrl+S"), self) 
        # copy_file.activated.connect(self.copy_file) 

if __name__ == '__main__':  
    app = QApplication(sys.argv)    
    main_window = MainWindow()  
    main_window.show()  
    sys.exit(app.exec_())
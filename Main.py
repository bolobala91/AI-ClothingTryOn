# main.py
import sys
import os
from dotenv import load_dotenv
import time
import shutil
import threading
import google.generativeai as genai
from PIL import Image

from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                            QTextEdit, QProgressBar, QMessageBox, QInputDialog, QLineEdit,
                            QFrame, QSizePolicy, QScrollArea, QGridLayout)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

# Thư mục để lưu ảnh kết quả
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'results'

# Đảm bảo các thư mục tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def read_api_key_from_file(file_path='api_key.txt'):
    """Đọc API key từ file api_key.txt"""
    try:
        with open(file_path, 'r') as file:
            api_key = file.read().strip()
            return api_key if api_key else None
    except FileNotFoundError:
        print(f"Không tìm thấy file {file_path}")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file API key: {str(e)}")
        return None

class GeminiThread(QThread):
    """
    Lớp xử lý luồng riêng để gọi API Gemini mà không làm đơ giao diện
    """
    finished_signal = pyqtSignal(bool, str, int)  # Thêm int để theo dõi thứ tự kết quả
    progress_signal = pyqtSignal(int, int)  # progress, thread_id
    
    def __init__(self, person_image_path, clothing_image_path, prompt, thread_id, api_key=None):
        super().__init__()
        self.person_image_path = person_image_path
        self.clothing_image_path = clothing_image_path
        self.prompt = prompt
        self.thread_id = thread_id
        self.api_key = api_key
        self.is_cancelled = False
        
    def run(self):
        try:
            if self.is_cancelled:
                return
                
            if not self.api_key:
                raise Exception("Không tìm thấy API key")
            
            # Báo hiệu tiến trình
            self.progress_signal.emit(10, self.thread_id)
            
            # Cấu hình API Gemini
            genai.configure(api_key=self.api_key)
            
            # Tải ảnh
            if self.is_cancelled:
                return
            self.progress_signal.emit(30, self.thread_id)
            
            # Mở ảnh bằng Pillow để đảm bảo tương thích với thư viện
            person_img = Image.open(self.person_image_path)
            clothing_img = Image.open(self.clothing_image_path)
            
            if self.is_cancelled:
                return
            self.progress_signal.emit(50, self.thread_id)
            
            # Khởi tạo mô hình Gemini
            model = genai.GenerativeModel("gemini-2.0-flash-exp-image-generation")
            
            if self.is_cancelled:
                return
            self.progress_signal.emit(60, self.thread_id)
            
            # Tạo biến thể khác nhau cho mỗi kết quả
            temperature = 0.4 + (self.thread_id * 0.05)  # Tăng dần độ sáng tạo
            
            # Cấu hình generation với nhiệt độ biến đổi theo thread_id
            generation_config = {
                "response_modalities": ["TEXT", "IMAGE"],
                "temperature": temperature,
                "top_k": 32,
                "top_p": 1,
                "max_output_tokens": 2048,
            }
            
            # Đợi 2 giây trước khi gọi API để tạo khoảng cách giữa các request
            for i in range(20):  # 2 giây, kiểm tra mỗi 0.1 giây
                if self.is_cancelled:
                    return
                time.sleep(0.1)
            
            if self.is_cancelled:
                return
                
            # Tạo yêu cầu API với prompt và ảnh
            response = model.generate_content(
                [self.prompt, person_img, clothing_img],
                generation_config=generation_config
            )
            
            if self.is_cancelled:
                return
            self.progress_signal.emit(80, self.thread_id)
            
            # Kiểm tra xem có ảnh trong phản hồi không
            result_image_path = None
            
            # Kiểm tra các phần trong phản hồi
            print(f"Xử lý phản hồi từ API cho kết quả {self.thread_id + 1}")
            
            for part in response.candidates[0].content.parts:
                if self.is_cancelled:
                    return
                    
                # Kiểm tra nếu phần này là text
                if hasattr(part, 'text') and part.text:
                    print(f"Phản hồi văn bản từ API (kết quả {self.thread_id + 1}):", part.text)
                
                # Kiểm tra nếu phần này là hình ảnh
                if hasattr(part, 'inline_data'):
                    print(f"Tìm thấy dữ liệu hình ảnh cho kết quả {self.thread_id + 1}")
                    
                    # Lưu ảnh kết quả
                    result_image_path = os.path.join(OUTPUT_FOLDER, f"result_{self.thread_id}_{int(time.time())}.png")
                    with open(result_image_path, "wb") as f:
                        f.write(part.inline_data.data)
                        
                    print(f"Đã lưu ảnh kết quả {self.thread_id + 1} tại: {result_image_path}")
                    break
            
            if result_image_path and not self.is_cancelled:
                self.progress_signal.emit(100, self.thread_id)
                self.finished_signal.emit(True, result_image_path, self.thread_id)
            elif not self.is_cancelled:
                raise Exception(f"API không trả về ảnh kết quả nào cho kết quả {self.thread_id + 1}")
                
        except Exception as e:
            if not self.is_cancelled:
                import traceback
                traceback.print_exc()
                self.finished_signal.emit(False, str(e), self.thread_id)
    
    def cancel(self):
        """Đánh dấu thread này đã bị hủy"""
        self.is_cancelled = True

class ResultWidget(QWidget):
    """
    Widget hiển thị một kết quả thử đồ
    """
    def __init__(self, id, parent=None):
        super().__init__(parent)
        self.id = id
        self.result_image_path = None
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Tiêu đề
        title = QLabel(f"Phiên bản {self.id + 1}")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        
        # Khung ảnh kết quả
        self.image_frame = QFrame()
        self.image_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.image_frame.setFrameShadow(QFrame.Shadow.Sunken)
        self.image_frame.setStyleSheet("background-color: #f5f5f5;")
        self.image_frame.setMinimumSize(250, 350)
        
        frame_layout = QVBoxLayout(self.image_frame)
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(220, 320)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("Đang xử lý...")
        
        frame_layout.addWidget(self.image_label)
        
        # Thanh tiến trình
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        # Nút lưu kết quả
        self.save_btn = QPushButton("Lưu")
        self.save_btn.setEnabled(False)
        
        # Thêm widgets vào layout
        layout.addWidget(title)
        layout.addWidget(self.image_frame)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.save_btn)
        
    def display_image(self, image_path):
        """Hiển thị ảnh kết quả"""
        self.result_image_path = image_path
        pixmap = QPixmap(image_path)
        
        # Giữ tỷ lệ khung hình khi hiển thị
        pixmap = pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(pixmap)
        self.save_btn.setEnabled(True)
        
    def update_progress(self, value):
        """Cập nhật thanh tiến trình"""
        self.progress_bar.setValue(value)
        
    def save_image(self, parent):
        """Lưu ảnh kết quả"""
        if not self.result_image_path:
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            parent, 
            f'Lưu Ảnh Kết Quả {self.id + 1}', 
            f'thudo_result_{self.id + 1}.png', 
            'PNG (*.png);;JPEG (*.jpg)'
        )
        
        if save_path:
            # Sao chép ảnh từ thư mục kết quả sang vị trí mới
            shutil.copy2(self.result_image_path, save_path)
            QMessageBox.information(parent, 'Thành công', f'Đã lưu ảnh vào: {save_path}')

class DuyThuDoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.person_image_path = None
        self.clothing_image_path = None
        self.result_widgets = []
        self.gemini_threads = []
        self.scheduled_timers = []  # Lưu trữ các QTimer đã đặt lịch
        self.init_ui()
        
    def init_ui(self):
        # Thiết lập cửa sổ chính
        self.setWindowTitle('nguyên Liệu làm hoạt hình 2D tại hoathinh2d.com')
        self.setMinimumSize(1400, 800)
        
        # Widget chính
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Panel bên trái - Chọn ảnh
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(350)
        
        # Tiêu đề phần mềm
        title_label = QLabel('Thử quần áo by Tô Đình Duy')
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet('font-size: 16pt; font-weight: bold; margin: 10px;')
        left_layout.addWidget(title_label)
        
        # Khung ảnh người
        person_frame = QFrame()
        person_frame.setFrameShape(QFrame.Shape.StyledPanel)
        person_frame.setFrameShadow(QFrame.Shadow.Sunken)
        person_frame.setLineWidth(2)
        person_frame.setStyleSheet('background-color: #f0f0f0;')
        
        person_frame_layout = QVBoxLayout(person_frame)
        
        person_title = QLabel('Ảnh Người')
        person_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        person_title.setStyleSheet('font-size: 16pt; font-weight: bold; color: black;')  # Thêm color: black
        
        self.person_image_label = QLabel()
        self.person_image_label.setFixedSize(300, 300)
        self.person_image_label.setStyleSheet('border: 2px dashed gray;')
        self.person_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.person_image_label.setText('Chưa có ảnh')
        
        person_upload_btn = QPushButton('Chọn Ảnh Người')
        person_upload_btn.setStyleSheet('font-size: 14pt; padding: 10px; color: black;')
        person_upload_btn.clicked.connect(self.upload_person_image)
        
        person_frame_layout.addWidget(person_title)
        person_frame_layout.addWidget(self.person_image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        person_frame_layout.addWidget(person_upload_btn)
        
        # Khung ảnh quần áo
        clothing_frame = QFrame()
        clothing_frame.setFrameShape(QFrame.Shape.StyledPanel)
        clothing_frame.setFrameShadow(QFrame.Shadow.Sunken)
        clothing_frame.setLineWidth(2)
        clothing_frame.setStyleSheet('background-color: #f0f0f0;')
        
        clothing_frame_layout = QVBoxLayout(clothing_frame)
        
        clothing_title = QLabel('Ảnh Quần Áo')
        clothing_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        clothing_title.setStyleSheet('font-size: 16pt; font-weight: bold;')
        
        self.clothing_image_label = QLabel()
        self.clothing_image_label.setFixedSize(300, 300)
        self.clothing_image_label.setStyleSheet('border: 2px dashed gray;')
        self.clothing_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clothing_image_label.setText('Chưa có ảnh')
        
        clothing_upload_btn = QPushButton('Chọn Ảnh Quần Áo')
        clothing_upload_btn.setStyleSheet('font-size: 14pt; padding: 10px; color: black;')
        clothing_upload_btn.clicked.connect(self.upload_clothing_image)
        
        clothing_frame_layout.addWidget(clothing_title)
        clothing_frame_layout.addWidget(self.clothing_image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        clothing_frame_layout.addWidget(clothing_upload_btn)
        
        # Phần prompt
        prompt_label = QLabel('Prompt tùy chỉnh:')
        prompt_label.setStyleSheet('font-size: 14pt; font-weight: bold;')
        
        self.prompt_text = QTextEdit()
        self.prompt_text.setPlaceholderText('Nhập hướng dẫn thêm cho AI (ví dụ: Làm cho trông thật hơn, phong cách đô thị, v.v.)')
        self.prompt_text.setText('Generate a high-quality virtual try-on image showing the person wearing the clothing from the second image. Preserve all facial features, hairstyle, skin tone, body proportions, pose, and background.')
        self.prompt_text.setMaximumHeight(100)
        
        # Nút tạo ảnh
        self.generate_btn = QPushButton('Tạo 10 Ảnh Thử Đồ')
        self.generate_btn.setStyleSheet('font-size: 16pt; padding: 15px; background-color: #4CAF50; color: white;')
        self.generate_btn.clicked.connect(self.generate_images)
        
        # Thêm các widget vào layout bên trái
        left_layout.addWidget(person_frame)
        left_layout.addWidget(clothing_frame)
        left_layout.addWidget(prompt_label)
        left_layout.addWidget(self.prompt_text)
        left_layout.addWidget(self.generate_btn)
        left_layout.addStretch()
        
        # Panel bên phải - Hiển thị kết quả
        right_panel = QWidget()
        
        # Tạo scroll area để cuộn
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        results_container = QWidget()
        self.results_layout = QGridLayout(results_container)
        
        # Tạo 10 widget kết quả
        NUM_RESULTS = 10
        COLS = 3  # Số cột hiển thị
        
        for i in range(NUM_RESULTS):
            result_widget = ResultWidget(i)
            row = i // COLS
            col = i % COLS
            self.results_layout.addWidget(result_widget, row, col)
            self.result_widgets.append(result_widget)
            
            # Kết nối nút lưu
            result_widget.save_btn.clicked.connect(lambda checked=False, idx=i: self.result_widgets[idx].save_image(self))
        
        scroll_area.setWidget(results_container)
        
        # Layout cho panel bên phải
        right_layout = QVBoxLayout(right_panel)
        
        # Tiêu đề kết quả
        result_title = QLabel('Ảnh Sau khi xử lý hiển thị tại khung này')
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_title.setStyleSheet('font-size: 16pt; font-weight: bold;')
        
        right_layout.addWidget(result_title)
        right_layout.addWidget(scroll_area)
        
        # Thêm các panel vào layout chính
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 3)  # Tỷ lệ 1:3
        
        # Thiết lập widget chính
        self.setCentralWidget(main_widget)
        
    def upload_person_image(self):
        """Mở hộp thoại chọn ảnh người"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Chọn Ảnh Người', '', 'Ảnh (*.png *.jpg *.jpeg *.gif)'
        )
        
        if file_path:
            self.person_image_path = file_path
            self.display_image(self.person_image_label, file_path)
            
    def upload_clothing_image(self):
        """Mở hộp thoại chọn ảnh quần áo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Chọn Ảnh Quần Áo', '', 'Ảnh (*.png *.jpg *.jpeg *.gif)'
        )
        
        if file_path:
            self.clothing_image_path = file_path
            self.display_image(self.clothing_image_label, file_path)
            
    def display_image(self, label, image_path):
        """Hiển thị ảnh được chọn trong QLabel"""
        pixmap = QPixmap(image_path)
        
        # Giữ tỷ lệ khung hình khi hiển thị
        pixmap = pixmap.scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        label.setPixmap(pixmap)

    def cancel_running_threads(self):
        """Hủy tất cả các thread đang chạy và các timer đã đặt lịch"""
        # Hủy tất cả các thread đang chạy
        for thread in self.gemini_threads:
            if thread.isRunning():
                print(f"Đang hủy thread {thread.thread_id + 1}...")
                thread.cancel()
                thread.wait(1000)  # Đợi tối đa 1 giây cho thread kết thúc
                
        # Hủy tất cả các timer đã đặt lịch
        for timer in self.scheduled_timers:
            if timer.isActive():
                timer.stop()
                
        self.gemini_threads.clear()
        self.scheduled_timers.clear()
            
    def generate_images(self):
        """Xử lý tạo nhiều ảnh kết quả"""
        # Hủy các thread đang chạy (nếu có)
        self.cancel_running_threads()
        
        if not self.person_image_path or not self.clothing_image_path:
            QMessageBox.warning(self, 'Cảnh báo', 'Vui lòng chọn cả ảnh người và ảnh quần áo!')
            return
            
        # Đọc API key từ file
        api_key = read_api_key_from_file()
        
        # Kiểm tra API key
        if not api_key:
            # Hiển thị hộp thoại nhập API key
            api_key, ok = QInputDialog.getText(
                self, 'Nhập API key', 
                'Không thể đọc API key từ file api_key.txt.\nVui lòng nhập API key của bạn:',
                QLineEdit.EchoMode.Password
            )
            if not ok or not api_key:
                QMessageBox.critical(self, 'Lỗi', 'Không thể tiếp tục mà không có API key!')
                return
            
            # Lưu API key vào file cho lần sau
            try:
                with open('api_key.txt', 'w') as file:
                    file.write(api_key)
                print("Đã lưu API key vào file api_key.txt")
            except Exception as e:
                print(f"Không thể lưu API key vào file: {str(e)}")
                
        # Lấy prompt từ người dùng
        prompt = self.prompt_text.toPlainText()
        if not prompt:
            prompt = "Generate a high-quality virtual try-on image showing the person wearing the clothing from the second image. Preserve all facial features, hairstyle, skin tone, body proportions, pose, and background."
            
        # Khởi tạo lại các thread và reset widget kết quả
        self.gemini_threads = []
        self.scheduled_timers = []
        
        # Reset các widget kết quả
        for widget in self.result_widgets:
            widget.image_label.setText("Đang xử lý...")
            widget.image_label.setPixmap(QPixmap())  # Xóa ảnh hiện tại nếu có
            widget.progress_bar.setValue(0)
            widget.save_btn.setEnabled(False)
            widget.result_image_path = None
        
        # Vô hiệu hóa nút tạo ảnh
        self.generate_btn.setEnabled(False)
        
        # Kiểm tra và tạo thư mục kết quả nếu chưa tồn tại
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Tạo nhiều thread cho nhiều kết quả, và khởi động từng thread cách nhau 2 giây
        for i in range(10):
            thread = GeminiThread(self.person_image_path, self.clothing_image_path, prompt, i, api_key)
            thread.progress_signal.connect(self.update_progress)
            thread.finished_signal.connect(self.process_result)
            self.gemini_threads.append(thread)
            
            # Đặt lịch khởi động thread sau mỗi 2 giây
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(lambda idx=i: self.start_thread(idx))
            timer.start(i * 2000)
            self.scheduled_timers.append(timer)
            
    def start_thread(self, idx):
        """Khởi động thread thứ idx"""
        if idx < len(self.gemini_threads):
            self.gemini_threads[idx].start()
            
    def update_progress(self, value, thread_id):
        """Cập nhật giá trị thanh tiến trình cho thread cụ thể"""
        if thread_id < len(self.result_widgets):
            self.result_widgets[thread_id].update_progress(value)
        
    def process_result(self, success, message, thread_id):
        """Xử lý kết quả từ API Gemini"""
        if thread_id >= len(self.result_widgets):
            return
            
        if success:
            # Hiển thị ảnh kết quả
            self.result_widgets[thread_id].display_image(message)
        else:
            # Hiển thị thông báo lỗi
            self.result_widgets[thread_id].image_label.setText(f"Lỗi: {message}")
            
        # Nếu tất cả thread đã hoàn thành, kích hoạt lại nút tạo ảnh
        all_done = True
        for thread in self.gemini_threads:
            if thread.isRunning():
                all_done = False
                break
                
        if all_done:
            self.generate_btn.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    
    # Thiết lập font chữ mặc định
    app.setFont(QFont("Arial", 10))
    
    # Thiết lập style sheet chung
    app.setStyleSheet('''
        QMainWindow {
            background-color: white;
        }
        QLabel {
            font-size: 12pt;
        }
        QPushButton {
            font-size: 12pt;
            padding: 8px;
            background-color: #4a86e8;
            color: white;
            border-radius: 5px;
            border: none;
        }
        QPushButton:hover {
            background-color: #3a76d8;
        }
        QPushButton:disabled {
            background-color: #cccccc;
        }
        QTextEdit {
            font-size: 12pt;
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 5px;
        }
        QFrame {
            border-radius: 8px;
        }
        QProgressBar {
            border: 2px solid #cccccc;
            border-radius: 5px;
            text-align: center;
            height: 25px;
            font-size: 12pt;
            font-weight: bold;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            width: 10px;
            margin: 0.5px;
        }
    ''')
    
    window = DuyThuDoApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
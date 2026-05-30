import os
import sys
import time
import pathlib
import shutil
import tempfile
import traceback
import numpy as np
import cv2
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QProgressBar,
    QFileDialog, QFrame, QCheckBox, QTextEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QScrollArea,
    QLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QSettings
from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QIcon, QIntValidator

# Import AI model engine
from realesrgan_ncnn_py import Realesrgan

# Import moviepy for video audio preservation
try:
    from moviepy import VideoFileClip, AudioFileClip
    HAS_MOVIEPY = True
except Exception:
    HAS_MOVIEPY = False

VERSION = "1.0.0"

def get_model_paths():
    """Resolves the models folder path in developmental or compiled EXE mode."""
    # 1. Try relative to the executable directory (for compiled EXE next to models/)
    app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    model_dir = os.path.join(app_dir, "models")
    if os.path.isdir(model_dir):
        return model_dir

    # 2. Try relative to this source script
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_file_dir, "models")
    if os.path.isdir(model_dir):
        return model_dir

    # 3. Try current working directory
    model_dir = os.path.join(os.getcwd(), "models")
    if os.path.isdir(model_dir):
        return model_dir

    return "models"


def detect_gpus():
    """Detects available graphics controllers using PowerShell WMI query on Windows."""
    import subprocess
    gpus = []
    try:
        cmd = "powershell -Command \"Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name\""
        output = subprocess.check_output(cmd, shell=True, text=True)
        gpus = [line.strip() for line in output.split('\n') if line.strip()]
    except Exception:
        try:
            cmd = "wmic path win32_VideoController get name"
            output = subprocess.check_output(cmd, shell=True, text=True)
            gpus = [line.strip() for line in output.split('\n') if line.strip()][1:]
        except Exception:
            pass
    return gpus


class DragDropWidget(QFrame):
    """Custom drag & drop file panel with modern responsive styling."""
    files_dropped = pyqtSignal(list)
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("DragDropZone")
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.icon_label = QLabel("📥", self)
        self.icon_label.setStyleSheet("font-size: 44px; margin-bottom: 4px;")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)
        
        self.text_label = QLabel("Drag & Drop Files here or Click to Browse", self)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #8e8e93;")
        layout.addWidget(self.text_label)
        
        self.setStyleSheet("""
            QFrame#DragDropZone {
                border: 2px dashed #3f3f46;
                border-radius: 12px;
                background-color: #1a1a1e;
                min-height: 100px;
                max-height: 120px;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QFrame#DragDropZone {
                    border: 2px dashed #00f0ff;
                    border-radius: 12px;
                    background-color: #1e293b;
                }
            """)
            self.text_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #00f0ff;")
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame#DragDropZone {
                border: 2px dashed #3f3f46;
                border-radius: 12px;
                background-color: #1a1a1e;
            }
        """)
        self.text_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #8e8e93;")

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("""
            QFrame#DragDropZone {
                border: 2px dashed #3f3f46;
                border-radius: 12px;
                background-color: #1a1a1e;
            }
        """)
        self.text_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #8e8e93;")
        
        urls = event.mimeData().urls()
        if urls:
            file_paths = [url.toLocalFile() for url in urls]
            self.files_dropped.emit(file_paths)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class ImageComparisonDialog(QDialog):
    """Premium interactive side-by-side image comparison window."""
    
    def __init__(self, original_path, upscaled_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Image Comparison — Before vs After")
        self.resize(1100, 750)
        self.setStyleSheet("""
            QDialog {
                background-color: #121214;
            }
            QLabel {
                color: #e4e4e7;
                font-family: "Segoe UI", sans-serif;
            }
            QLabel#Badge {
                font-size: 10px;
                font-weight: bold;
                padding: 2px 6px;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #1e1e24;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: bold;
                color: #e4e4e7;
            }
            QPushButton:hover {
                background-color: #27272a;
                border-color: #52525b;
            }
            QPushButton:checked {
                background-color: #00f0ff;
                color: #0c0c0e;
                border-color: #00f0ff;
            }
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        # Split Side-by-Side Area
        split_layout = QHBoxLayout()
        split_layout.setSpacing(6)
        
        def create_image_view(title, path, badge_style):
            container = QFrame()
            container.setStyleSheet("""
                QFrame {
                    background-color: #16161a;
                    border: 1px solid #27272a;
                    border-radius: 6px;
                }
            """)
            layout = QVBoxLayout(container)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)
            
            # Badge & Resolution Header
            header = QHBoxLayout()
            header.setContentsMargins(2, 2, 2, 2)
            badge = QLabel(title)
            badge.setObjectName("Badge")
            badge.setStyleSheet(badge_style)
            header.addWidget(badge)
            
            res_lbl = QLabel()
            res_lbl.setStyleSheet("color: #71717a; font-size: 11px; font-weight: bold;")
            
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    res_lbl.setText(f"{w} x {h}")
            except Exception:
                pass
                
            header.addStretch()
            header.addWidget(res_lbl)
            layout.addLayout(header)
            
            # Scroll Area for image
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("border: none; background: transparent;")
            
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setStyleSheet("background-color: #0c0c0e; border-radius: 4px;")
            
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                img_label.setPixmap(pixmap)
            
            scroll.setWidget(img_label)
            layout.addWidget(scroll)
            return container, img_label, pixmap

        # Before image (Original)
        before_container, self.before_label, self.before_pix = create_image_view(
            "BEFORE", 
            original_path, 
            "background-color: #27272a; color: #a1a1aa;"
        )
        split_layout.addWidget(before_container, stretch=1)
        
        # After image (Upscaled)
        after_container, self.after_label, self.after_pix = create_image_view(
            "AFTER", 
            upscaled_path, 
            "background-color: #1e293b; color: #00f0ff; border: 1px solid #00f0ff;"
        )
        split_layout.addWidget(after_container, stretch=1)
        
        main_layout.addLayout(split_layout)
        
        # Mini Controls footer
        controls = QHBoxLayout()
        controls.setContentsMargins(4, 2, 4, 2)
        controls.setSpacing(6)
        
        self.btn_fit = QPushButton("Fit to Window")
        self.btn_fit.setCheckable(True)
        self.btn_fit.setChecked(True)
        self.btn_fit.clicked.connect(self.scale_images)
        controls.addWidget(self.btn_fit)
        
        self.btn_original = QPushButton("100% Size")
        self.btn_original.clicked.connect(self.show_original_size)
        controls.addWidget(self.btn_original)
        
        self.btn_fullscreen = QPushButton("⛶ Fullscreen")
        self.btn_fullscreen.setCheckable(True)
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        controls.addWidget(self.btn_fullscreen)
        
        controls.addStretch()
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        controls.addWidget(btn_close)
        
        main_layout.addLayout(controls)
        
        # Initial scaling
        self.scale_images()
        
        # Open maximized by default (now safe after all widgets are fully constructed)
        self.showMaximized()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "btn_fit") and self.btn_fit.isChecked():
            self.scale_images()

    def scale_images(self):
        self.btn_fit.setChecked(True)
        for lbl, pix in [(self.before_label, self.before_pix), (self.after_label, self.after_pix)]:
            if pix and not pix.isNull():
                parent_sz = lbl.parentWidget().size()
                # Subtract small margin for safe scaling without triggering scroll bars
                target_w = max(10, parent_sz.width() - 8)
                target_h = max(10, parent_sz.height() - 8)
                scaled = pix.scaled(
                    QSize(target_w, target_h), 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                lbl.setPixmap(scaled)

    def show_original_size(self):
        self.btn_fit.setChecked(False)
        self.before_label.setPixmap(self.before_pix)
        self.after_label.setPixmap(self.after_pix)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
            self.btn_fullscreen.setChecked(False)
            self.btn_fullscreen.setText("⛶ Fullscreen")
        else:
            self.showFullScreen()
            self.btn_fullscreen.setChecked(True)
            self.btn_fullscreen.setText("🗗 Windowed")


class AboutDialog(QDialog):
    """Premium interactive About window showing author and application information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About IMVI-Upscaler")
        self.setStyleSheet("""
            QDialog {
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 12px;
            }
            QLabel {
                font-family: "Segoe UI", -apple-system, sans-serif;
                color: #e4e4e7;
            }
            QLabel#AppTitle {
                font-size: 18px;
                font-weight: bold;
                color: #00f0ff;
            }
            QLabel#AuthorInfo {
                font-size: 14px;
                color: #a1a1aa;
            }
            QLabel#AuthorName {
                font-size: 14px;
                font-weight: bold;
                color: #10b981;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        
        # Cyberpunk glowing header
        app_title = QLabel("IMVI-Upscaler", self)
        app_title.setObjectName("AppTitle")
        app_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(app_title)
        
        version_lbl = QLabel(f"Version {VERSION}", self)
        version_lbl.setStyleSheet("color: #71717a; font-size: 11px; font-weight: bold;")
        version_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_lbl)
        
        layout.addSpacing(6)
        
        # Author info
        author_layout = QHBoxLayout()
        author_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        author_lbl = QLabel("Author: ", self)
        author_lbl.setObjectName("AuthorInfo")
        
        author_name = QLabel("""TerOl,
        <a href="https://github.com/terol1982/IMVI-Upscaler" style="color: #00f0ff; text-decoration: none;">github.com/terol1982/IMVI-Upscaler</a>""", self)
        author_name.setObjectName("AuthorName")
        author_name.setOpenExternalLinks(True)
        author_name.setTextFormat(Qt.TextFormat.RichText)
        
        author_layout.addWidget(author_lbl, alignment=Qt.AlignmentFlag.AlignTop)
        author_layout.addWidget(author_name, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(author_layout)
        
        layout.addSpacing(10)
        
        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        close_btn = QPushButton("OK", self)
        close_btn.setObjectName("CloseBtn")
        close_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00b8d4, stop:1 #00f0ff);
                color: #0c0c0e;
                border: none;
                border-radius: 6px;
                padding: 6px 20px;
                font-weight: bold;
                font-size: 13px;
                min-width: 80px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00e5ff, stop:1 #33f4ff);
            }
        """)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)


class UpscaleWorker(QThread):
    """Background processing thread for non-blocking upscaling with cancellation."""
    progress_updated = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    preview_ready = pyqtSignal(QImage)
    file_started = pyqtSignal(int, str)
    file_finished = pyqtSignal(int, bool, str)

    def __init__(self, tasks, output_dir, gpu_id=0, format_choice="auto", model_name="RealESRGAN_General_x4_v3"):
        super().__init__()
        self.tasks = tasks  # list of {"path": str, "target_w": int, "target_h": int}
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        self.format_choice = format_choice
        self.model_name = model_name
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True
        self.log_message.emit("Cancellation requested...")

    def run(self):
        if not self.tasks:
            self.finished.emit(False, "No tasks to process.")
            return

        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
            
            # Resolve models folder path
            model_dir = get_model_paths()
            param_path = os.path.join(model_dir, f"{self.model_name}.param")
            bin_path = os.path.join(model_dir, f"{self.model_name}.bin")

            if not os.path.exists(param_path) or not os.path.exists(bin_path):
                self.finished.emit(False, f"Model files for '{self.model_name}' not found in folder: {model_dir}")
                return

            # Initialize model loader ONCE for the entire batch
            self.progress_updated.emit(0, f"Loading AI model: {self.model_name}...")
            self.log_message.emit(f"Initializing RealESRGAN Engine with model: {self.model_name}...")
            try:
                upscaler = Realesrgan(gpuid=self.gpu_id, model=-1)
                upscaler._load(param_path=pathlib.Path(param_path), model_path=pathlib.Path(bin_path), scale=4)
                device_str = "Vulkan GPU" if self.gpu_id >= 0 else "CPU"
                self.log_message.emit(f"Model '{self.model_name}' loaded successfully on {device_str} (GPU ID: {self.gpu_id})")
            except Exception as e:
                self.log_message.emit(f"GPU initialization failed ({e}). Falling back to CPU mode...")
                upscaler = Realesrgan(gpuid=-1, model=-1)
                upscaler._load(param_path=pathlib.Path(param_path), model_path=pathlib.Path(bin_path), scale=4)
                self.log_message.emit(f"Model '{self.model_name}' loaded successfully on CPU.")

            # Processing loop
            num_tasks = len(self.tasks)
            self.log_message.emit(f"Starting batch of {num_tasks} files...")
            
            for i, task in enumerate(self.tasks):
                if self._is_cancelled:
                    break
                
                input_path = task["path"]
                target_w = task["target_w"]
                target_h = task["target_h"]
                
                filename = os.path.basename(input_path)
                basename, ext = os.path.splitext(filename)
                
                # Signal file started
                self.file_started.emit(i, input_path)
                self.log_message.emit(f"[{i+1}/{num_tasks}] Processing file: {filename}")
                self.progress_updated.emit(0, f"File {i+1}/{num_tasks}: {filename}")
                
                is_video = ext.lower() in {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.m4v', '.mpeg', '.mpg'}
                
                if is_video:
                    output_ext = ext
                else:
                    if self.format_choice == "jpg":
                        output_ext = ".jpg"
                    elif self.format_choice == "png":
                        output_ext = ".png"
                    else:  # auto
                        output_ext = ext.lower()
                        if output_ext not in {'.png', '.webp', '.bmp', '.jpg', '.jpeg', '.tiff'}:
                            output_ext = '.jpg'
                    
                output_filename = f"upscaled_{basename}{output_ext}"
                output_path = os.path.join(self.output_dir, output_filename)
                
                temp_files = []
                try:
                    if not is_video:
                        self.progress_updated.emit(20, f"[{i+1}/{num_tasks}] Reading image...")
                        with Image.open(input_path) as pil_img:
                            if self._is_cancelled:
                                break
                            
                            self.progress_updated.emit(40, f"[{i+1}/{num_tasks}] Running AI upscaler (x4)...")
                            # Run native PIL-based upscaling to preserve absolute lossless quality and color spaces
                            result_pil = upscaler.process_pil(pil_img)
                            
                            if self._is_cancelled:
                                break
                                
                            if (result_pil.width, result_pil.height) != (target_w, target_h):
                                self.progress_updated.emit(80, f"[{i+1}/{num_tasks}] Resizing to target...")
                                result_pil = result_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                                
                            self.progress_updated.emit(90, f"[{i+1}/{num_tasks}] Saving image...")
                            if output_ext.lower() in {'.jpg', '.jpeg'}:
                                result_pil.convert("RGB").save(output_path, format="JPEG", quality=95)
                            elif output_ext.lower() == '.png':
                                result_pil.save(output_path, format="PNG")
                            else:
                                result_pil.save(output_path)
                        
                        self.file_finished.emit(i, True, output_path)
                        self.log_message.emit(f"Successfully processed: {filename}")
                        
                    else:
                        # Video pipeline
                        cap = cv2.VideoCapture(input_path)
                        if not cap.isOpened():
                            raise ValueError("Could not open source video.")
                        
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if fps <= 0:
                            fps = 30.0
                            
                        has_audio = False
                        audio_temp_path = None
                        if HAS_MOVIEPY:
                            try:
                                self.log_message.emit("Checking video for audio track...")
                                clip = VideoFileClip(input_path)
                                if clip.audio is not None:
                                    has_audio = True
                                    self.log_message.emit("Audio track detected. Extracting audio...")
                                    fd, audio_temp_path = tempfile.mkstemp(suffix=".wav")
                                    os.close(fd)
                                    temp_files.append(audio_temp_path)
                                    clip.audio.write_audiofile(audio_temp_path, logger=None)
                                clip.close()
                            except Exception as ae:
                                self.log_message.emit(f"Audio extraction warning: {ae}")
                                has_audio = False
                                
                        if self._is_cancelled:
                            cap.release()
                            break
                            
                        fd, silent_temp_path = tempfile.mkstemp(suffix=".mp4", dir=self.output_dir)
                        os.close(fd)
                        temp_files.append(silent_temp_path)
                        
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(silent_temp_path, fourcc, fps, (target_w, target_h))
                        
                        frame_count = 0
                        start_time = time.time()
                        
                        while True:
                            if self._is_cancelled:
                                break
                                
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            upscaled_frame = upscaler.process_cv2(frame)
                            if (upscaled_frame.shape[1], upscaled_frame.shape[0]) != (target_w, target_h):
                                upscaled_frame = cv2.resize(upscaled_frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                                
                            writer.write(upscaled_frame)
                            frame_count += 1
                            
                            percent = int((frame_count / total_frames) * 90)
                            elapsed = time.time() - start_time
                            fps_rate = frame_count / elapsed if elapsed > 0 else 0
                            eta = (total_frames - frame_count) / fps_rate if fps_rate > 0 else 0
                            
                            status_msg = f"File {i+1}/{num_tasks} - {frame_count}/{total_frames} frames ({percent}%) - {fps_rate:.1f} FPS - ETA: {int(eta)}s"
                            
                            if frame_count % 5 == 0 or frame_count == total_frames:
                                self.progress_updated.emit(10 + percent, status_msg)
                                
                                rgb_preview = cv2.cvtColor(cv2.resize(upscaled_frame, (320, 180)), cv2.COLOR_BGR2RGB)
                                success_enc, encoded_img = cv2.imencode('.jpg', rgb_preview)
                                if success_enc:
                                    qimg = QImage()
                                    qimg.loadFromData(encoded_img.tobytes())
                                    self.preview_ready.emit(qimg)
                                    
                        cap.release()
                        writer.release()
                        
                        if self._is_cancelled:
                            break
                            
                        # Audio Merge
                        if has_audio and HAS_MOVIEPY and audio_temp_path and os.path.exists(audio_temp_path):
                            self.progress_updated.emit(95, f"[{i+1}/{num_tasks}] Merging audio...")
                            video_clip = VideoFileClip(silent_temp_path)
                            audio_clip = AudioFileClip(audio_temp_path)
                            final_clip = video_clip.with_audio(audio_clip)
                            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None, preset="medium")
                            video_clip.close()
                            audio_clip.close()
                            final_clip.close()
                        else:
                            shutil.copy(silent_temp_path, output_path)
                            
                        self.file_finished.emit(i, True, output_path)
                        self.log_message.emit(f"Successfully processed video: {filename}")
                        
                except Exception as file_err:
                    self.file_finished.emit(i, False, str(file_err))
                    self.log_message.emit(f"Error processing {filename}: {file_err}")
                finally:
                    # Clean up temp files for this file
                    for temp_f in temp_files:
                        if temp_f and os.path.exists(temp_f):
                            try:
                                os.remove(temp_f)
                            except OSError:
                                pass
            
            if self._is_cancelled:
                self.finished.emit(False, "Batch cancelled by user.")
            else:
                self.progress_updated.emit(100, "All files successfully processed!")
                self.finished.emit(True, f"Successfully processed all {num_tasks} files.")
                
        except Exception as batch_err:
            tb = traceback.format_exc()
            self.log_message.emit(f"Batch execution error:\n{tb}")
            self.finished.emit(False, f"Critical error during batch: {batch_err}")


class UpscalerApp(QMainWindow):
    """Main Application GUI Window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"IMVI-Upscaler - {VERSION}")
        self.setMinimumSize(920, 680)
        self.input_path = ""
        self.original_w = 0
        self.original_h = 0
        self.aspect_ratio = 1.0
        self.worker = None
        self.queue_tasks = []

        # Build premium custom style sheet
        self.apply_theme()

        # Build main layout
        self.init_ui()

        # Load saved configurations
        self.load_settings()

        # Connect UI changes to auto-save settings
        self.dest_input.textChanged.connect(self.save_settings)
        self.width_input.textChanged.connect(self.save_settings)
        self.height_input.textChanged.connect(self.save_settings)
        self.ratio_btn.clicked.connect(self.save_settings)
        self.device_combo.currentIndexChanged.connect(self.save_settings)
        self.format_combo.currentIndexChanged.connect(self.save_settings)
        self.model_combo.currentIndexChanged.connect(self.save_settings)

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121214;
            }
            QWidget {
                color: #e4e4e7;
                font-family: "Segoe UI", -apple-system, sans-serif;
            }
            QLabel {
                font-size: 13px;
            }
            QLabel#TitleLabel {
                font-size: 22px;
                font-weight: bold;
                color: #00f0ff;
                background: transparent;
            }
            QLabel#SectionHeader {
                font-size: 14px;
                font-weight: bold;
                color: #a1a1aa;
                border-bottom: 1px solid #27272a;
                padding-bottom: 4px;
                margin-top: 8px;
            }
            QLineEdit {
                background-color: #1a1a1e;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 6px 10px;
                color: #ffffff;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #00f0ff;
            }
            QPushButton {
                background-color: #1e1e24;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 7px 14px;
                font-weight: bold;
                font-size: 13px;
                color: #e4e4e7;
            }
            QPushButton:hover {
                background-color: #27272a;
                border-color: #52525b;
            }
            QPushButton#PrimaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00b8d4, stop:1 #00f0ff);
                color: #0c0c0e;
                border: none;
                font-size: 14px;
            }
            QPushButton#PrimaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00e5ff, stop:1 #33f4ff);
            }
            QPushButton#PrimaryBtn:disabled {
                background: #27272a;
                color: #71717a;
            }
            QPushButton#CancelBtn {
                background-color: #7f1d1d;
                color: #fca5a5;
                border: 1px solid #991b1b;
            }
            QPushButton#CancelBtn:hover {
                background-color: #991b1b;
            }
            QProgressBar {
                border: 1px solid #27272a;
                border-radius: 6px;
                background-color: #1a1a1e;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                height: 18px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #005f73, stop:1 #00f0ff);
                border-radius: 5px;
            }
            QTextEdit {
                background-color: #0c0c0e;
                border: 1px solid #27272a;
                border-radius: 8px;
                font-family: Consolas, Monaco, monospace;
                font-size: 11px;
                color: #10b981;
                padding: 6px;
            }
            QFrame#PreviewCard {
                background-color: #1a1a1e;
                border: 1px solid #27272a;
                border-radius: 10px;
            }
            QFrame#SettingsCard {
                background-color: #1a1a1e;
                border: 1px solid #27272a;
                border-radius: 10px;
            }
            QCheckBox {
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                background-color: #1a1a1e;
            }
            QCheckBox::indicator:checked {
                background-color: #00f0ff;
                border-color: #00f0ff;
                image: url(data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="black" width="12px" height="12px"><path d="M0 0h24v24H0V0z" fill="none"/><path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/></svg>);
            }
        """)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(18, 14, 18, 18)
        main_layout.setSpacing(14)

        # Header Title section
        header_layout = QHBoxLayout()
        title_label = QLabel("IMVI-Upscaler")
        title_label.setObjectName("TitleLabel")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Processing Device Dynamic Dropdown
        device_label = QLabel("Processing Device:")
        device_label.setStyleSheet("font-weight: bold; color: #a1a1aa; margin-right: 6px;")
        header_layout.addWidget(device_label)
        
        self.device_combo = QComboBox()
        self.device_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a1e;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 4px 8px;
                color: #00f0ff;
                font-weight: bold;
                min-width: 260px;
            }
            QComboBox:focus {
                border-color: #00f0ff;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a1e;
                border: 1px solid #3f3f46;
                selection-background-color: #27272a;
                selection-color: #00f0ff;
                color: #e4e4e7;
            }
        """)
        
        # Detect available GPUs
        detected_sys_gpus = detect_gpus()
        
        # Check if PyTorch has CUDA support active
        has_cuda_active = False
        try:
            import torch
            has_cuda_active = torch.cuda.is_available()
        except Exception:
            pass
            
        # Populate QComboBox
        added_gpus = False
        if len(detected_sys_gpus) > 0:
            for idx, gpu_name in enumerate(detected_sys_gpus):
                is_nvidia = any(x in gpu_name.lower() for x in ["tesla", "nvidia", "rtx", "gtx", "quadro", "geforce"])
                
                cuda_suffix = ""
                if is_nvidia:
                    if has_cuda_active:
                        cuda_suffix = " + CUDA (Active)"
                    else:
                        cuda_suffix = " + CUDA (Compatible)"
                        
                name_fmt = f"⚡ GPU {idx}: {gpu_name}{cuda_suffix}"
                if is_nvidia or "amd" in gpu_name.lower():
                    name_fmt += " (Recommended)"
                self.device_combo.addItem(name_fmt, idx)
                added_gpus = True
        
        if not added_gpus:
            cuda_suffix = " + CUDA (Active)" if has_cuda_active else " + CUDA (Compatible)"
            self.device_combo.addItem(f"⚡ GPU 1: Tesla V100{cuda_suffix} (Recommended)", 1)
            self.device_combo.addItem("⚡ GPU 0: Intel UHD Graphics 750", 0)
            
        self.device_combo.addItem("💻 CPU Mode (Stable fallback)", -1)
        
        # Set default to GPU 1 (Tesla) since it is highly stable and dedicated, or index 0 if GPU 1 is not in list
        default_idx = 0
        for i in range(self.device_combo.count()):
            if "gpu 1" in self.device_combo.itemText(i).lower() or "recommended" in self.device_combo.itemText(i).lower():
                default_idx = i
                break
        self.device_combo.setCurrentIndex(default_idx)
        
        header_layout.addWidget(self.device_combo)
        
        main_layout.addLayout(header_layout)

        # Panels (Grid split: Left Input details vs Right settings details)
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(14)
        
        # --- LEFT COLUMN (Source Inspector) ---
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        
        # Drag Drop zone
        self.drop_zone = DragDropWidget()
        self.drop_zone.files_dropped.connect(self.add_files_to_queue)
        self.drop_zone.clicked.connect(self.browse_file)
        left_layout.addWidget(self.drop_zone)

        # Batch Queue Header & Controls
        queue_header_layout = QHBoxLayout()
        queue_title = QLabel("BATCH FILE QUEUE")
        queue_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #a1a1aa;")
        queue_header_layout.addWidget(queue_title)
        queue_header_layout.addStretch()
        
        self.btn_remove_selected = QPushButton("Remove Selected")
        self.btn_remove_selected.setStyleSheet("padding: 3px 8px; font-size: 11px; background-color: #27272a;")
        self.btn_remove_selected.clicked.connect(self.remove_selected_from_queue)
        queue_header_layout.addWidget(self.btn_remove_selected)
        
        self.btn_clear_queue = QPushButton("Clear All")
        self.btn_clear_queue.setStyleSheet("padding: 3px 8px; font-size: 11px; background-color: #27272a;")
        self.btn_clear_queue.clicked.connect(self.clear_queue)
        queue_header_layout.addWidget(self.btn_clear_queue)
        
        left_layout.addLayout(queue_header_layout)

        # Batch Queue Table
        self.queue_table = QTableWidget(0, 5)
        self.queue_table.setHorizontalHeaderLabels(["File Name", "Type", "Size", "Resolution", "Status"])
        self.queue_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.queue_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.queue_table.itemSelectionChanged.connect(self.on_queue_selection_changed)
        self.queue_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.queue_table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1e;
                border: 1px solid #27272a;
                border-radius: 8px;
                color: #e4e4e7;
                gridline-color: #27272a;
                min-height: 180px;
                max-height: 240px;
            }
            QTableWidget::item {
                padding: 4px;
                font-size: 11px;
            }
            QTableWidget::item:selected {
                background-color: #27272a;
                color: #00f0ff;
            }
            QHeaderView::section {
                background-color: #121214;
                color: #a1a1aa;
                padding: 4px;
                font-weight: bold;
                border: 1px solid #27272a;
                font-size: 11px;
            }
        """)
        left_layout.addWidget(self.queue_table)

        # File preview & Info container card
        preview_card = QFrame()
        preview_card.setObjectName("PreviewCard")
        preview_card_layout = QVBoxLayout(preview_card)
        preview_card_layout.setContentsMargins(12, 12, 12, 12)
        preview_card_layout.setSpacing(8)

        section_left_title = QLabel("QUEUED FILE PREVIEW & DETAILS")
        section_left_title.setObjectName("SectionHeader")
        preview_card_layout.addWidget(section_left_title)

        # Thumbnail Label
        self.thumb_label = QLabel("Click a queued file to preview")
        self.thumb_label.setFixedSize(QSize(250, 140))
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setStyleSheet("""
            background-color: #0c0c0e; 
            border: 1px solid #27272a; 
            border-radius: 6px; 
            color: #71717a;
            font-size: 12px;
        """)
        
        # Centering thumbnail
        thumb_center_layout = QHBoxLayout()
        thumb_center_layout.addStretch()
        thumb_center_layout.addWidget(self.thumb_label)
        thumb_center_layout.addStretch()
        preview_card_layout.addLayout(thumb_center_layout)

        # File metadata label grid
        self.info_grid = QWidget()
        info_grid_layout = QGridLayout(self.info_grid)
        info_grid_layout.setContentsMargins(0, 4, 0, 0)
        info_grid_layout.setSpacing(6)
        
        def add_info_row(row_idx, label_text):
            lbl_title = QLabel(label_text)
            lbl_title.setStyleSheet("color: #a1a1aa; font-weight: bold; font-size: 11px;")
            lbl_val = QLabel("-")
            lbl_val.setStyleSheet("color: #e4e4e7; font-size: 11px;")
            lbl_val.setWordWrap(True)
            info_grid_layout.addWidget(lbl_title, row_idx, 0)
            info_grid_layout.addWidget(lbl_val, row_idx, 1)
            return lbl_val

        self.info_path = add_info_row(0, "Path:")
        self.info_size = add_info_row(1, "File Size:")
        self.info_type = add_info_row(2, "Format:")
        self.info_res = add_info_row(3, "Original Size:")
        self.info_extra = add_info_row(4, "FPS / Length:")
        
        preview_card_layout.addWidget(self.info_grid)

        # Compare Before/After button
        self.btn_compare = QPushButton("🔍 Compare Before/After")
        self.btn_compare.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #059669);
                color: #ffffff;
                border: none;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                margin-top: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #34d399, stop:1 #10b981);
            }
        """)
        self.btn_compare.setVisible(False)
        self.btn_compare.clicked.connect(self.open_comparison)
        preview_card_layout.addWidget(self.btn_compare)

        left_layout.addWidget(preview_card)
        panels_layout.addWidget(left_column, stretch=4)

        # --- RIGHT COLUMN (Upscale & Output Settings) ---
        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        # Settings Card
        settings_card = QFrame()
        settings_card.setObjectName("SettingsCard")
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(10)

        section_right_title = QLabel("UPSCALE CONFIGURATION")
        section_right_title.setObjectName("SectionHeader")
        settings_layout.addWidget(section_right_title)

        # Destination Folder field
        dest_label = QLabel("Output Save Directory:")
        dest_label.setStyleSheet("font-weight: bold; color: #a1a1aa;")
        settings_layout.addWidget(dest_label)
        
        dest_browse_layout = QHBoxLayout()
        self.dest_input = QLineEdit()
        self.dest_input.setPlaceholderText("Select destination folder...")
        dest_browse_layout.addWidget(self.dest_input)
        
        dest_btn = QPushButton("Browse...")
        dest_btn.clicked.connect(self.browse_dest)
        dest_browse_layout.addWidget(dest_btn)
        settings_layout.addLayout(dest_browse_layout)

        # Custom Dimensions Layout
        dim_label = QLabel("Target Resolution (Output Pixels):")
        dim_label.setStyleSheet("font-weight: bold; color: #a1a1aa;")
        settings_layout.addWidget(dim_label)

        dim_grid = QHBoxLayout()
        
        # Width Input
        dim_grid.addWidget(QLabel("Width:"))
        self.width_input = QLineEdit()
        self.width_input.setValidator(QIntValidator(1, 16384))
        self.width_input.setPlaceholderText("Native x4")
        self.width_input.textEdited.connect(self.on_width_edited)
        dim_grid.addWidget(self.width_input)
        
        # Aspect Ratio lock link icon
        self.ratio_btn = QPushButton("🔗")
        self.ratio_btn.setFixedSize(28, 28)
        self.ratio_btn.setCheckable(True)
        self.ratio_btn.setChecked(True)
        self.ratio_btn.setToolTip("Lock Aspect Ratio")
        self.ratio_btn.setStyleSheet("""
            QPushButton { 
                background: #27272a; 
                border: 1px solid #3f3f46; 
                border-radius: 4px;
                font-size: 14px;
                padding: 0;
            }
            QPushButton:checked { 
                background: #00f0ff; 
                color: #0c0c0e;
                border-color: #00f0ff;
            }
        """)
        self.ratio_btn.clicked.connect(self.toggle_aspect_ratio)
        dim_grid.addWidget(self.ratio_btn)

        # Height Input
        dim_grid.addWidget(QLabel("Height:"))
        self.height_input = QLineEdit()
        self.height_input.setValidator(QIntValidator(1, 16384))
        self.height_input.setPlaceholderText("Native x4")
        self.height_input.textEdited.connect(self.on_height_edited)
        dim_grid.addWidget(self.height_input)
        
        settings_layout.addLayout(dim_grid)
         # Note details
        note_lbl = QLabel("Leave both fields blank to process at model's native x4 resolution.\n"
                          "Specifying only one scales the other will be calculated proportionally.")
        note_lbl.setStyleSheet("color: #71717a; font-size: 11px;")
        settings_layout.addWidget(note_lbl)

        # AI Model field
        model_label = QLabel("AI Model:")
        model_label.setStyleSheet("font-weight: bold; color: #a1a1aa; margin-top: 6px;")
        settings_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a1e;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 5px 8px;
                color: #e4e4e7;
                font-weight: bold;
            }
            QComboBox:focus {
                border-color: #00f0ff;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a1e;
                border: 1px solid #3f3f46;
                selection-background-color: #27272a;
                selection-color: #00f0ff;
                color: #e4e4e7;
            }
        """)
        settings_layout.addWidget(self.model_combo)

        # Output Format field
        format_label = QLabel("Output Image Format:")
        format_label.setStyleSheet("font-weight: bold; color: #a1a1aa; margin-top: 6px;")
        settings_layout.addWidget(format_label)
        
        self.format_combo = QComboBox()
        self.format_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a1e;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 5px 8px;
                color: #e4e4e7;
                font-weight: bold;
            }
            QComboBox:focus {
                border-color: #00f0ff;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a1e;
                border: 1px solid #3f3f46;
                selection-background-color: #27272a;
                selection-color: #00f0ff;
                color: #e4e4e7;
            }
        """)
        self.format_combo.addItem("Auto (Keep original)", "auto")
        self.format_combo.addItem("JPEG (.jpg)", "jpg")
        self.format_combo.addItem("PNG (.png)", "png")
        settings_layout.addWidget(self.format_combo)
        
        # Process Buttons
        settings_layout.addSpacing(10)
        self.start_btn = QPushButton("START AI UPSCALE")
        self.start_btn.setObjectName("PrimaryBtn")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_upscaling)
        settings_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("CANCEL PROCESSING")
        self.cancel_btn.setObjectName("CancelBtn")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_upscaling)
        settings_layout.addWidget(self.cancel_btn)

        right_layout.addWidget(settings_card)

        # Live Console Output log
        log_label = QLabel("LIVE LOGS & CONSOLE")
        log_label.setObjectName("SectionHeader")
        right_layout.addWidget(log_label)
        
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.append("System ready. Drop a file to begin.")
        right_layout.addWidget(self.console_output)

        panels_layout.addWidget(right_column, stretch=5)
        
        main_layout.addLayout(panels_layout)

        # Progress bar & About button bottom layout
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        bottom_layout.addWidget(self.progress_bar, stretch=1)

        self.about_btn = QPushButton("About")
        self.about_btn.setObjectName("AboutBtn")
        self.about_btn.setStyleSheet("""
            QPushButton#AboutBtn {
                min-width: 80px;
                max-width: 80px;
                background-color: #1e1e24;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: bold;
                font-size: 13px;
                color: #a1a1aa;
            }
            QPushButton#AboutBtn:hover {
                background-color: #27272a;
                border-color: #00f0ff;
                color: #00f0ff;
            }
        """)
        self.about_btn.clicked.connect(self.show_about_dialog)
        bottom_layout.addWidget(self.about_btn)

        main_layout.addLayout(bottom_layout)

        # Scan and populate model selection list
        self.populate_models()

    def add_files_to_queue(self, file_paths):
        """Inspects and appends multiple files (images/videos) to the batch queue table."""
        for path in file_paths:
            if not os.path.exists(path):
                continue
            
            # Check if already in queue
            if any(t["path"] == path for t in self.queue_tasks):
                continue
                
            filename = os.path.basename(path)
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            is_video = ext in {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.m4v', '.mpeg', '.mpg'}
            
            # File size calculation
            bytes_sz = os.path.getsize(path)
            if bytes_sz < 1024 * 1024:
                size_str = f"{bytes_sz / 1024:.1f} KB"
            else:
                size_str = f"{bytes_sz / (1024 * 1024):.2f} MB"
                
            orig_w, orig_h = 0, 0
            aspect = 1.0
            
            # Parse dimensions safely
            if not is_video:
                try:
                    with Image.open(path) as img:
                        orig_w, orig_h = img.size
                    aspect = orig_w / orig_h
                except Exception:
                    pass
            else:
                try:
                    cap = cv2.VideoCapture(path)
                    if cap.isOpened():
                        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        aspect = orig_w / orig_h if orig_h > 0 else 1.0
                    cap.release()
                except Exception:
                    pass
            
            task = {
                "path": path,
                "original_w": orig_w,
                "original_h": orig_h,
                "aspect_ratio": aspect,
                "is_video": is_video,
                "size_str": size_str,
                "type_str": "Video" if is_video else "Image",
                "res_str": f"{orig_w}x{orig_h}" if orig_w > 0 else "Unknown"
            }
            
            self.queue_tasks.append(task)
            
            # Insert row into Table UI
            row_idx = self.queue_table.rowCount()
            self.queue_table.insertRow(row_idx)
            
            # Column cells
            item_name = QTableWidgetItem(filename)
            item_name.setToolTip(path)
            self.queue_table.setItem(row_idx, 0, item_name)
            self.queue_table.setItem(row_idx, 1, QTableWidgetItem(task["type_str"]))
            self.queue_table.setItem(row_idx, 2, QTableWidgetItem(task["size_str"]))
            self.queue_table.setItem(row_idx, 3, QTableWidgetItem(task["res_str"]))
            
            item_status = QTableWidgetItem("Pending")
            item_status.setForeground(Qt.GlobalColor.cyan)
            self.queue_table.setItem(row_idx, 4, item_status)
            
        self.log(f"Added {len(file_paths)} files to queue. Total queued: {len(self.queue_tasks)}")
        
        # If queue was empty, select first item
        if self.queue_table.rowCount() > 0 and self.queue_table.selectedRanges() == []:
            self.queue_table.selectRow(0)
            
        self.start_btn.setEnabled(len(self.queue_tasks) > 0)

    def remove_selected_from_queue(self):
        """Removes the selected row task from the queue table and local memory list."""
        selected_rows = self.queue_table.selectedItems()
        if not selected_rows:
            return
            
        row_idx = selected_rows[0].row()
        filename = self.queue_table.item(row_idx, 0).text()
        
        self.queue_table.removeRow(row_idx)
        self.queue_tasks.pop(row_idx)
        
        self.log(f"Removed '{filename}' from batch queue.")
        self.start_btn.setEnabled(len(self.queue_tasks) > 0)
        
        if self.queue_table.rowCount() > 0:
            next_select = min(row_idx, self.queue_table.rowCount() - 1)
            self.queue_table.selectRow(next_select)
        else:
            self.thumb_label.setPixmap(QPixmap())
            self.thumb_label.setText("Click a queued file to preview")
            self.info_path.setText("-")
            self.info_size.setText("-")
            self.info_type.setText("-")
            self.info_res.setText("-")
            self.info_extra.setText("-")

    def clear_queue(self):
        """Clears the entire table widget queue and local task queue."""
        self.queue_table.setRowCount(0)
        self.queue_tasks = []
        self.start_btn.setEnabled(False)
        self.thumb_label.setPixmap(QPixmap())
        self.thumb_label.setText("Click a queued file to preview")
        self.info_path.setText("-")
        self.info_size.setText("-")
        self.info_type.setText("-")
        self.info_res.setText("-")
        self.info_extra.setText("-")
        self.log("Batch queue cleared.")

    def on_queue_selection_changed(self):
        """Triggered when a queue item is clicked/highlighted. Loads dynamic metadata preview."""
        selected_rows = self.queue_table.selectedItems()
        if not selected_rows:
            return
            
        row_idx = selected_rows[0].row()
        if row_idx < 0 or row_idx >= len(self.queue_tasks):
            return
            
        task = self.queue_tasks[row_idx]
        self.load_file(task["path"])

    def log(self, text):
        """Append log lines to text edit."""
        self.console_output.append(text)
        # Auto scroll to bottom
        self.console_output.ensureCursorVisible()

    def browse_file(self):
        file_filter = "Media Files (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif *.mp4 *.avi *.mkv *.mov *.webm *.flv *.m4v *.mpeg *.mpg)"
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Image or Video Files", "", file_filter)
        if file_paths:
            self.add_files_to_queue(file_paths)

    def browse_dest(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", self.dest_input.text())
        if folder:
            self.dest_input.setText(os.path.normpath(folder))

    def load_file(self, file_path):
        """Inspect media file parameters, loads thumbnail preview and extracts details."""
        if not os.path.exists(file_path):
            self.log(f"Error: Selected file does not exist: {file_path}")
            return
        
        self.input_path = file_path
        # Only set destination directory if it hasn't been set yet
        if not self.dest_input.text().strip():
            self.dest_input.setText(os.path.normpath(os.path.dirname(file_path)))
        
        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        # Calculate human-readable file size
        bytes_sz = os.path.getsize(file_path)
        if bytes_sz < 1024 * 1024:
            size_str = f"{bytes_sz / 1024:.1f} KB"
        else:
            size_str = f"{bytes_sz / (1024 * 1024):.2f} MB"

        is_video = ext in {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.m4v', '.mpeg', '.mpg'}

        self.info_path.setText(filename)
        self.info_size.setText(size_str)
        self.info_type.setText("Video" if is_video else "Image")
        
        # Blank dimension fields for clean resets
        self.width_input.clear()
        self.height_input.clear()

        # Reset preview layout
        self.thumb_label.setPixmap(QPixmap())
        self.thumb_label.setText("Generating preview...")

        # Process metadata based on media type
        if not is_video:
            try:
                # Load metadata safely using Pillow
                with Image.open(file_path) as img:
                    self.original_w, self.original_h = img.size
                
                self.aspect_ratio = self.original_w / self.original_h
                self.info_res.setText(f"{self.original_w} x {self.original_h}")
                self.info_extra.setText("Static Image")

                # Generate thumbnail pixmap
                pixmap = QPixmap(file_path)
                scaled_pixmap = pixmap.scaled(
                    self.thumb_label.size(), 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.thumb_label.setPixmap(scaled_pixmap)
                
                self.start_btn.setEnabled(True)
                self.log(f"Successfully loaded image: {filename} ({self.original_w}x{self.original_h})")
            except Exception as e:
                self.thumb_label.setText("Preview Failed")
                self.log(f"Error parsing image: {e}")
                self.start_btn.setEnabled(False)
        else:
            try:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    raise ValueError("OpenCV VideoCapture failed to open file.")

                self.original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                self.aspect_ratio = self.original_w / self.original_h if self.original_h > 0 else 1.0
                
                duration = total_frames / fps if fps > 0 else 0.0
                duration_str = f"{int(duration // 60)}m {int(duration % 60)}s ({total_frames} frames)"

                self.info_res.setText(f"{self.original_w} x {self.original_h}")
                self.info_extra.setText(f"{fps:.2f} FPS / {duration_str}")

                # Read a frame from the middle of the video for dynamic thumbnail
                middle_frame_idx = min(30, total_frames // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Convert to QPixmap
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    qimg = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    scaled_pixmap = pixmap.scaled(
                        self.thumb_label.size(), 
                        Qt.AspectRatioMode.KeepAspectRatio, 
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.thumb_label.setPixmap(scaled_pixmap)
                else:
                    self.thumb_label.setText("No Frame Available")

                cap.release()
                
                self.start_btn.setEnabled(True)
                self.log(f"Successfully loaded video: {filename} ({self.original_w}x{self.original_h} @ {fps:.2f} FPS)")
            except Exception as e:
                self.thumb_label.setText("Preview Failed")
                self.log(f"Error parsing video metadata: {e}")
                self.start_btn.setEnabled(False)

        # Check if upscaled version is available to toggle Compare button visibility
        selected_rows = self.queue_table.selectedItems()
        has_upscaled = False
        if selected_rows:
            row_idx = selected_rows[0].row()
            if 0 <= row_idx < len(self.queue_tasks):
                task = self.queue_tasks[row_idx]
                if "upscaled_path" in task and task["upscaled_path"] and os.path.exists(task["upscaled_path"]):
                    has_upscaled = True
                    
        self.btn_compare.setVisible(has_upscaled and not is_video)

    def toggle_aspect_ratio(self):
        """Forces updating dimension fields on ratio toggle."""
        if self.ratio_btn.isChecked():
            self.log("Aspect ratio lock enabled.")
            # Trigger sync on width input
            self.on_width_edited(self.width_input.text())
        else:
            self.log("Aspect ratio lock disabled.")

    def on_width_edited(self, text):
        """Triggers proportional height adjustment."""
        if not self.ratio_btn.isChecked() or self.original_w <= 0:
            return
        
        if not text.isdigit():
            self.height_input.blockSignals(True)
            self.height_input.clear()
            self.height_input.blockSignals(True)
            return

        w_val = int(text)
        h_val = int(w_val / self.aspect_ratio)
        
        self.height_input.blockSignals(True)
        self.height_input.setText(str(h_val))
        self.height_input.blockSignals(False)

    def on_height_edited(self, text):
        """Triggers proportional width adjustment."""
        if not self.ratio_btn.isChecked() or self.original_h <= 0:
            return
        
        if not text.isdigit():
            self.width_input.blockSignals(True)
            self.width_input.clear()
            self.width_input.blockSignals(False)
            return

        h_val = int(text)
        w_val = int(h_val * self.aspect_ratio)
        
        self.width_input.blockSignals(True)
        self.width_input.setText(str(w_val))
        self.width_input.blockSignals(False)

    def start_upscaling(self):
        """Retrieves queue tasks, prepares loop data, and triggers background processing."""
        if not self.queue_tasks:
            self.log("Error: Batch queue is empty. Please add files.")
            return

        # 1. Parse global target dimensions (if set)
        w_txt = self.width_input.text().strip()
        h_txt = self.height_input.text().strip()

        # Build list of tasks with custom dimension resolution per file
        tasks_to_run = []
        for task in self.queue_tasks:
            tw, th = 0, 0
            if not w_txt and not h_txt:
                tw = task["original_w"] * 4
                th = task["original_h"] * 4
            elif w_txt and not h_txt:
                tw = int(w_txt)
                th = int(tw / task["aspect_ratio"])
            elif not w_txt and h_txt:
                th = int(h_txt)
                tw = int(th * task["aspect_ratio"])
            else:
                tw = int(w_txt)
                th = int(h_txt)
                
            tasks_to_run.append({
                "path": task["path"],
                "target_w": max(32, tw),
                "target_h": max(32, th)
            })

        # 2. Get output directory
        out_dir = self.dest_input.text().strip()
        if not out_dir:
            self.log("Error: Please select a valid output directory.")
            return

        # 3. Check GPU selection status
        gpu_id = self.device_combo.currentData()

        # Adjust UI states
        self.set_ui_processing_state(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")

        # Reset all queue table statuses to Pending
        for idx in range(self.queue_table.rowCount()):
            item = self.queue_table.item(idx, 4)
            if item:
                item.setText("Pending")
                item.setForeground(Qt.GlobalColor.cyan)

        # 4. Spawn Background Worker Thread
        self.worker = UpscaleWorker(
            tasks=tasks_to_run, 
            output_dir=out_dir, 
            gpu_id=gpu_id,
            format_choice=self.format_combo.currentData(),
            model_name=self.model_combo.currentData()
        )

        self.worker.progress_updated.connect(self.on_worker_progress)
        self.worker.log_message.connect(self.log)
        self.worker.preview_ready.connect(self.on_preview_ready)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_finished.connect(self.on_file_finished)
        self.worker.finished.connect(self.on_worker_finished)
        
        self.worker.start()

    def cancel_upscaling(self):
        """Triggers cancellation flag inside the working thread."""
        if self.worker and self.worker.isRunning():
            self.cancel_btn.setEnabled(False)
            self.worker.cancel()

    def on_file_started(self, idx, path):
        """Called when worker starts processing a queued file."""
        if 0 <= idx < self.queue_table.rowCount():
            item = self.queue_table.item(idx, 4)
            if item:
                item.setText("Processing...")
                item.setForeground(Qt.GlobalColor.yellow)
            # Auto scroll and select the row currently processing
            self.queue_table.selectRow(idx)

    def on_file_finished(self, idx, success, message):
        """Called when worker completes processing a queued file."""
        if 0 <= idx < self.queue_table.rowCount():
            item = self.queue_table.item(idx, 4)
            if item:
                if success:
                    item.setText("Done")
                    item.setForeground(Qt.GlobalColor.green)
                    if 0 <= idx < len(self.queue_tasks):
                        self.queue_tasks[idx]["upscaled_path"] = message
                else:
                    item.setText("Failed")
                    item.setForeground(Qt.GlobalColor.red)
                    item.setToolTip(message)

    def on_worker_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.progress_bar.setFormat(msg)

    def on_preview_ready(self, qimg):
        """Displays real-time frame previews of video upscaling."""
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            self.thumb_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.thumb_label.setPixmap(scaled_pixmap)

    def on_worker_finished(self, success, message):
        """Handles completion steps of thread execution."""
        self.set_ui_processing_state(False)
        self.progress_bar.setValue(100 if success else 0)
        
        if success:
            self.progress_bar.setFormat("Successfully Finished!")
            self.log(f"\n--- BATCH COMPLETE ---\n{message}\n")
        else:
            self.progress_bar.setFormat("Failed/Cancelled")
            self.log(f"\n--- BATCH INTERRUPTED ---\n{message}\n")
            
            # Mark all remaining Pending rows as Cancelled
            for idx in range(self.queue_table.rowCount()):
                item = self.queue_table.item(idx, 4)
                if item and item.text() == "Pending":
                    item.setText("Cancelled")
                    item.setForeground(Qt.GlobalColor.gray)

        # Trigger selection update to reload preview
        self.on_queue_selection_changed()
        
        # Auto-open comparison dialog for single image upscales
        if success and len(self.queue_tasks) == 1 and not self.queue_tasks[0]["is_video"]:
            task = self.queue_tasks[0]
            if "upscaled_path" in task and task["upscaled_path"] and os.path.exists(task["upscaled_path"]):
                self.open_comparison_for_task(task)
                
        self.worker = None

    def set_ui_processing_state(self, processing):
        """Locks/unlocks input widgets during upscale process."""
        self.drop_zone.setEnabled(not processing)
        self.dest_input.setEnabled(not processing)
        self.width_input.setEnabled(not processing)
        self.height_input.setEnabled(not processing)
        self.ratio_btn.setEnabled(not processing)
        self.device_combo.setEnabled(not processing)
        self.format_combo.setEnabled(not processing)
        self.model_combo.setEnabled(not processing)
        
        # Batch queue controls
        self.btn_remove_selected.setEnabled(not processing)
        self.btn_clear_queue.setEnabled(not processing)
        self.queue_table.setEnabled(not processing)
        
        self.start_btn.setVisible(not processing)
        self.cancel_btn.setVisible(processing)
        self.cancel_btn.setEnabled(True)

    def closeEvent(self, event):
        """Saves settings automatically when the window is closed."""
        self.save_settings()
        event.accept()

    def save_settings(self):
        """Saves current configuration parameters via QSettings in a local INI file."""
        if hasattr(self, "_loading_settings") and self._loading_settings:
            return
            
        ini_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "config.ini")
        settings = QSettings(ini_path, QSettings.Format.IniFormat)
        settings.setValue("output_dir", self.dest_input.text())
        settings.setValue("width_input", self.width_input.text())
        settings.setValue("height_input", self.height_input.text())
        settings.setValue("ratio_locked", self.ratio_btn.isChecked())
        settings.setValue("device_index", self.device_combo.currentIndex())
        settings.setValue("format_index", self.format_combo.currentIndex())
        settings.setValue("model_index", self.model_combo.currentIndex())

    def load_settings(self):
        """Loads and applies saved configuration parameters from a local INI file."""
        self._loading_settings = True
        try:
            ini_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "config.ini")
            settings = QSettings(ini_path, QSettings.Format.IniFormat)
            
            output_dir = settings.value("output_dir", "")
            if output_dir:
                self.dest_input.setText(output_dir)
                
            width_input = settings.value("width_input", "")
            if width_input:
                self.width_input.setText(width_input)
                
            height_input = settings.value("height_input", "")
            if height_input:
                self.height_input.setText(height_input)
                
            ratio_locked = settings.value("ratio_locked", True, type=bool)
            self.ratio_btn.setChecked(ratio_locked)
            
            device_index = settings.value("device_index", -1, type=int)
            if device_index >= 0 and device_index < self.device_combo.count():
                self.device_combo.setCurrentIndex(device_index)
                
            format_index = settings.value("format_index", 0, type=int)
            if format_index >= 0 and format_index < self.format_combo.count():
                self.format_combo.setCurrentIndex(format_index)
                
            model_index = settings.value("model_index", -1, type=int)
            if model_index >= 0 and model_index < self.model_combo.count():
                self.model_combo.setCurrentIndex(model_index)
        except Exception as e:
            print(f"Error loading settings: {e}")
        finally:
            self._loading_settings = False

    def populate_models(self):
        """Scans the models directory for .bin and .param pairs and populates the dropdown."""
        self.model_combo.clear()
        model_dir = get_model_paths()
        
        models = []
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith(".bin"):
                    base_name = file[:-4]
                    param_file = f"{base_name}.param"
                    if os.path.exists(os.path.join(model_dir, param_file)):
                        models.append(base_name)
                        
        if not models:
            models = ["RealESRGAN_General_x4_v3"]
            
        for m in sorted(models):
            self.model_combo.addItem(m, m)
            
        # Select RealESRGAN_General_x4_v3 by default if present
        default_idx = self.model_combo.findData("RealESRGAN_General_x4_v3")
        if default_idx >= 0:
            self.model_combo.setCurrentIndex(default_idx)

    def open_comparison(self):
        """Opens comparison dialog for the currently selected queued item."""
        selected_rows = self.queue_table.selectedItems()
        if not selected_rows:
            return
        row_idx = selected_rows[0].row()
        if 0 <= row_idx < len(self.queue_tasks):
            task = self.queue_tasks[row_idx]
            self.open_comparison_for_task(task)

    def open_comparison_for_task(self, task):
        """Helper to create and launch the comparison dialog."""
        original = task["path"]
        upscaled = task.get("upscaled_path", "")
        if original and upscaled and os.path.exists(upscaled):
            dialog = ImageComparisonDialog(original, upscaled, self)
            dialog.exec()

    def show_about_dialog(self):
        """Opens the About dialog with author information."""
        dialog = AboutDialog(self)
        dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = UpscalerApp()
    window.show()
    sys.exit(app.exec())

import cv2
import os
import shutil
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from functools import partial

# --- CONFIGURATION ---
EXPERIMENT_FOLDER = "EXPERIMENT"
DATA_FOLDER = os.path.join(EXPERIMENT_FOLDER, "data")
TRAINER_FILE = os.path.join(EXPERIMENT_FOLDER, "trainer.yml")
LABELS_FILE = os.path.join(EXPERIMENT_FOLDER, "labels.npy")

if not os.path.exists(EXPERIMENT_FOLDER): os.makedirs(EXPERIMENT_FOLDER)
if not os.path.exists(DATA_FOLDER): os.makedirs(DATA_FOLDER)

class RecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.is_trained = False
        self.labels_map = {}
        self.load_training_data()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )

    def load_training_data(self):
        if os.path.exists(TRAINER_FILE) and os.path.exists(LABELS_FILE):
            try:
                self.recognizer.read(TRAINER_FILE)
                self.labels_map = np.load(LABELS_FILE, allow_pickle=True).item()
                self.is_trained = True
                print("System: Model loaded.")
            except Exception as e:
                print(f"System: Could not load model: {e}")
                self.is_trained = False

    def train(self):
        print("System: Starting training...")
        faces = []
        ids = []
        current_id = 0
        new_labels_map = {}
        if not os.path.exists(DATA_FOLDER):
            return False

        for object_name in os.listdir(DATA_FOLDER):
            obj_path = os.path.join(DATA_FOLDER, object_name)
            if not os.path.isdir(obj_path):
                continue

            image_files = [f for f in os.listdir(obj_path) if not f.startswith(".")]
            if not image_files:
                continue

            new_labels_map[current_id] = object_name

            for image_name in image_files:
                img_path = os.path.join(obj_path, image_name)
                try:
                    pil_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if pil_image is None:
                        continue
                    pil_image = cv2.resize(pil_image, (200, 200))
                    faces.append(pil_image)
                    ids.append(current_id)
                except Exception:
                    pass
            current_id += 1

        if len(faces) > 0:
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.save(TRAINER_FILE)
            np.save(LABELS_FILE, new_labels_map)
            self.labels_map = new_labels_map
            self.is_trained = True
            print("System: Training Complete!")
            return True
        else:
            self.is_trained = False
            return False

# --- UI HELPERS ---

class SimplePopup(Popup):
    def __init__(self, title, message, on_yes, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.size_hint = (0.6, 0.4)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        layout.add_widget(Label(text=message))
        btns = BoxLayout(size_hint_y=0.4, spacing=10)
        btn_yes = Button(text="Yes", background_color=(0, 1, 0, 1))
        btn_yes.bind(on_press=lambda x: [on_yes(), self.dismiss()])
        btn_no = Button(text="No", background_color=(1, 0, 0, 1))
        btn_no.bind(on_press=self.dismiss)
        btns.add_widget(btn_yes)
        btns.add_widget(btn_no)
        layout.add_widget(btns)
        self.add_widget(layout)

class PlaygroundSettingsPopup(Popup):
    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.app = app_instance
        self.title = "Playground / Debug Settings"
        self.size_hint = (0.9, 0.8)

        layout = BoxLayout(orientation='vertical', padding=15, spacing=15)

        # Confidence threshold
        layout.add_widget(Label(text="[Strictness] LBPH confidence threshold (lower = stricter)"))
        self.lbl_conf = Label(text=f"Current: {int(self.app.conf_threshold)}")
        layout.add_widget(self.lbl_conf)
        slider_conf = Slider(min=20, max=150, value=self.app.conf_threshold)
        slider_conf.bind(value=lambda inst, val: self.update_val("conf_threshold", val, self.lbl_conf))
        layout.add_widget(slider_conf)

        # Min object size
        layout.add_widget(Label(text="[Filter] Min object width/height (pixels)"))
        self.lbl_size = Label(text=f"Current: {int(self.app.min_object_size)}")
        layout.add_widget(self.lbl_size)
        slider_size = Slider(min=10, max=250, value=self.app.min_object_size)
        slider_size.bind(value=lambda inst, val: self.update_val("min_object_size", val, self.lbl_size))
        layout.add_widget(slider_size)

        # Motion threshold
        layout.add_widget(Label(text="[Motion] Background subtractor threshold"))
        self.lbl_motion = Label(text=f"Current: {int(self.app.motion_threshold)}")
        layout.add_widget(self.lbl_motion)
        slider_motion = Slider(min=10, max=100, value=self.app.motion_threshold)
        slider_motion.bind(value=lambda inst, val: self.update_motion(val, self.lbl_motion))
        layout.add_widget(slider_motion)

        btn_close = Button(text="Close & Apply", size_hint_y=None, height=50)
        btn_close.bind(on_press=self.dismiss)
        layout.add_widget(btn_close)

        self.add_widget(layout)

    def update_val(self, param, value, label_widget):
        setattr(self.app, param, int(value))
        label_widget.text = f"Current: {int(value)}"

    def update_motion(self, value, label_widget):
        self.app.motion_threshold = int(value)
        self.app.system.bg_subtractor.setVarThreshold(float(value))
        label_widget.text = f"Current: {int(value)}"

class ManageDataPopup(Popup):
    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.app = app_instance
        self.title = "Manage Saved Objects"
        self.size_hint = (0.9, 0.9)

        self.main_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        btn_create = Button(
            text="+ Create New Empty Object", size_hint_y=0.1, background_color=(0, 0.7, 1, 1)
        )
        btn_create.bind(on_press=self.prompt_create_new)
        self.main_layout.add_widget(btn_create)

        self.scroll = ScrollView(size_hint=(1, 0.8))
        self.list_layout = GridLayout(cols=1, spacing=5, size_hint_y=None)
        self.list_layout.bind(minimum_height=self.list_layout.setter("height"))
        self.scroll.add_widget(self.list_layout)
        self.main_layout.add_widget(self.scroll)

        btn_close = Button(text="Close", size_hint_y=0.1)
        btn_close.bind(on_press=self.dismiss)
        self.main_layout.add_widget(btn_close)

        self.add_widget(self.main_layout)
        self.populate_list()

    def populate_list(self):
        self.list_layout.clear_widgets()
        if not os.path.exists(DATA_FOLDER):
            return
        folders = [f for f in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, f))]
        if not folders:
            self.list_layout.add_widget(Label(text="No objects found.", size_hint_y=None, height=40))
            return
        for folder_name in folders:
            row = BoxLayout(size_hint_y=None, height=45, spacing=5)
            path = os.path.join(DATA_FOLDER, folder_name)
            count = len([f for f in os.listdir(path) if not f.startswith(".")])
            lbl_name = Label(text=f"{folder_name} ({count} imgs)", size_hint_x=0.4)

            btn_select = Button(
                text="Select", size_hint_x=0.2, background_color=(0, 1, 0, 1)
            )
            btn_select.bind(on_press=partial(self.select_object, folder_name))

            btn_delete = Button(
                text="Delete", size_hint_x=0.2, background_color=(1, 0, 0, 1)
            )
            btn_delete.bind(on_press=partial(self.delete_object, folder_name))

            row.add_widget(lbl_name)
            row.add_widget(btn_select)
            row.add_widget(btn_delete)
            self.list_layout.add_widget(row)

    def prompt_create_new(self, instance):
        content = BoxLayout(orientation="vertical", padding=10, spacing=10)
        input_text = TextInput(hint_text="Enter object name", multiline=False)
        btn_save = Button(text="Create", size_hint_y=None, height=40)
        popup = Popup(title="New Object", content=content, size_hint=(0.7, 0.4))

        def do_create(inst):
            name = input_text.text.strip()
            if name:
                path = os.path.join(DATA_FOLDER, name)
                if not os.path.exists(path):
                    os.makedirs(path)
                    self.populate_list()
                    popup.dismiss()

        btn_save.bind(on_press=do_create)
        content.add_widget(input_text)
        content.add_widget(btn_save)
        popup.open()

    def select_object(self, name, instance):
        self.app.set_active_object(name)
        self.dismiss()

    def delete_object(self, name, instance):
        def perform_delete():
            path = os.path.join(DATA_FOLDER, name)
            try:
                shutil.rmtree(path)
                self.populate_list()
                if self.app.current_object_name == name:
                    self.app.reset_active_object()
                self.app.btn_train.text = "TRAIN NEEDED"
                self.app.btn_train.background_color = (1, 0, 0, 1)
                self.app.system.is_trained = False
            except Exception:
                pass

        SimplePopup("Confirm Delete", f"Delete '{name}'?", perform_delete).open()

class ClickableImage(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selection_start = None
        self.selection_end = None
        self.is_dragging = False
        self.normalized_roi = None

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            app = App.get_running_app()
            if app.detection_mode in ("MANUAL", "TEST"):  # allow manual ROI in TEST
                self.selection_start = touch.pos
                self.selection_end = touch.pos
                self.is_dragging = True
                return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.is_dragging:
            self.selection_end = touch.pos
            self.calculate_normalized_roi()
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.is_dragging:
            self.is_dragging = False
            self.selection_end = touch.pos
            self.calculate_normalized_roi()
            return True
        return super().on_touch_up(touch)

    def calculate_normalized_roi(self):
        x1, y1 = self.selection_start
        x2, y2 = self.selection_end
        rect_x = min(x1, x2)
        rect_y = min(y1, y2)
        rect_w = abs(x1 - x2)
        rect_h = abs(y1 - y2)
        norm_x = (rect_x - self.x) / self.width
        norm_y = (rect_y - self.y) / self.height
        norm_w = rect_w / self.width
        norm_h = rect_h / self.height
        self.normalized_roi = (norm_x, norm_y, norm_w, norm_h)

# --- MAIN APP ---

class CameraApp(App):
    def build(self):
        self.system = RecognitionSystem()
        self.capture = cv2.VideoCapture(0)
        self.capture_count = 0
        self.current_rois = []
        self.current_object_name = None

        # Modes: FACE, OBJECT, MANUAL, TEST
        self.detection_mode = "FACE"

        # Playground params
        self.conf_threshold = 85
        self.min_object_size = 40
        self.motion_threshold = 50

        self.main_layout = BoxLayout(orientation="vertical")

        self.img_widget = ClickableImage()
        self.main_layout.add_widget(self.img_widget)

        controls = BoxLayout(orientation="vertical", size_hint_y=0.4, padding=5, spacing=5)

        # Mode row
        mode_row = BoxLayout(size_hint_y=0.3, spacing=5)
        self.btn_mode = Button(text="Mode: FACE", background_color=(0, 0.5, 0.5, 1))
        self.btn_mode.bind(on_press=self.toggle_mode)

        self.btn_settings = Button(text="Playground Settings", background_color=(0.8, 0.5, 0, 1))
        self.btn_settings.bind(on_press=self.open_settings)

        self.btn_reset_bg = Button(text="Reset BG", disabled=True)
        self.btn_reset_bg.bind(on_press=self.reset_background)

        mode_row.add_widget(self.btn_mode)
        mode_row.add_widget(self.btn_settings)
        mode_row.add_widget(self.btn_reset_bg)
        controls.add_widget(mode_row)

        # Name + confirm
        input_row = BoxLayout(size_hint_y=0.3, spacing=5)
        self.name_input = TextInput(hint_text="Object Name", multiline=False)
        self.btn_confirm = Button(
            text="Confirm", size_hint_x=0.3, background_color=(0.9, 0.8, 0, 1)
        )
        self.btn_confirm.bind(on_press=self.confirm_name_action)
        input_row.add_widget(self.name_input)
        input_row.add_widget(self.btn_confirm)
        controls.add_widget(input_row)

        # Actions
        btn_row = BoxLayout(size_hint_y=0.4, spacing=5)
        self.btn_capture = Button(
            text="Capture (Locked)", disabled=True, background_color=(0.3, 0.3, 0.3, 1)
        )
        self.btn_capture.bind(on_press=self.save_current_object)

        self.btn_manage = Button(text="Manage Data", background_color=(0.5, 0.5, 0.5, 1))
        self.btn_manage.bind(on_press=self.open_manage_popup)

        self.btn_train = Button(text="TRAIN MODEL", background_color=(1, 0, 0, 1))
        self.btn_train.bind(on_press=self.trigger_training)

        btn_row.add_widget(self.btn_capture)
        btn_row.add_widget(self.btn_manage)
        btn_row.add_widget(self.btn_train)
        controls.add_widget(btn_row)

        self.main_layout.add_widget(controls)
        Window.bind(on_keyboard=self._on_keyboard)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.main_layout

    # --- MODES & SETTINGS ---

    def toggle_mode(self, instance):
        # Rotate: FACE -> OBJECT -> MANUAL -> TEST -> FACE
        if self.detection_mode == "FACE":
            self.detection_mode = "OBJECT"
            self.btn_mode.text = "Mode: OBJECT (Motion)"
            self.btn_mode.background_color = (1, 0.5, 0, 1)
            self.btn_reset_bg.disabled = False
        elif self.detection_mode == "OBJECT":
            self.detection_mode = "MANUAL"
            self.btn_mode.text = "Mode: MANUAL (ROI)"
            self.btn_mode.background_color = (0.5, 0, 0.5, 1)
            self.btn_reset_bg.disabled = True
        elif self.detection_mode == "MANUAL":
            self.detection_mode = "TEST"
            self.btn_mode.text = "Mode: TEST (All)"
            self.btn_mode.background_color = (0, 0.7, 0, 1)
            self.btn_reset_bg.disabled = False
        else:
            self.detection_mode = "FACE"
            self.btn_mode.text = "Mode: FACE"
            self.btn_mode.background_color = (0, 0.5, 0.5, 1)
            self.btn_reset_bg.disabled = True

    def reset_background(self, instance):
        self.system.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=self.motion_threshold, detectShadows=False
        )

    def open_settings(self, instance):
        PlaygroundSettingsPopup(app_instance=self).open()

    # --- NAME / ACTIVE OBJECT ---

    def confirm_name_action(self, instance):
        if self.current_object_name:
            self.reset_active_object()
            return

        name = self.name_input.text.strip()
        if not name:
            return

        path = os.path.join(DATA_FOLDER, name)
        if os.path.exists(path):
            self.set_active_object(name)
        else:
            def create_it():
                os.makedirs(path)
                self.set_active_object(name)

            SimplePopup("Create New?", f"Create '{name}'?", create_it).open()

    def set_active_object(self, name):
        self.current_object_name = name
        self.name_input.text = name
        self.name_input.disabled = True
        self.btn_confirm.text = "Unlock"
        self.btn_capture.disabled = False
        self.btn_capture.text = "Capture (S)"
        self.btn_capture.background_color = (0, 0, 1, 1)

    def reset_active_object(self):
        self.current_object_name = None
        self.name_input.disabled = False
        self.name_input.text = ""
        self.btn_confirm.text = "Confirm"
        self.btn_capture.disabled = True
        self.btn_capture.text = "Capture (Locked)"
        self.btn_capture.background_color = (0.3, 0.3, 0.3, 1)

    # --- OTHER UI ---

    def open_manage_popup(self, instance):
        ManageDataPopup(app_instance=self).open()

    def _on_keyboard(self, instance, key, scancode, codepoint, modifiers):
        if codepoint == "s" and self.current_object_name:
            self.save_current_object(instance)

    # --- UTILS: MERGE BOXES (for TEST mode) ---

    def merge_rectangles(self, rects, iou_threshold=0.4):
        """Merge overlapping rectangles to avoid duplicate boxes for same region."""
        if not rects:
            return []

        rects = [list(r) for r in rects]
        merged = []

        while rects:
            x, y, w, h = rects.pop(0)
            has_merged = False
            for i, (mx, my, mw, mh) in enumerate(merged):
                # Compute IoU-ish overlap
                x1 = max(x, mx)
                y1 = max(y, my)
                x2 = min(x + w, mx + mw)
                y2 = min(y + h, my + mh)
                inter_w = max(0, x2 - x1)
                inter_h = max(0, y2 - y1)
                inter_area = inter_w * inter_h
                if inter_area == 0:
                    continue
                union_area = w * h + mw * mh - inter_area
                iou = inter_area / float(union_area)
                if iou > iou_threshold:
                    # Merge by taking outer bounds
                    nx = min(x, mx)
                    ny = min(y, my)
                    nw = max(x + w, mx + mw) - nx
                    nh = max(y + h, my + mh) - ny
                    merged[i] = [nx, ny, nw, nh]
                    has_merged = True
                    break
            if not has_merged:
                merged.append([x, y, w, h])

        return merged

    # --- MAIN LOOP ---

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.current_rois = []
        rects = []

        # 1. detection per mode
        if self.detection_mode == "FACE":
            faces = self.system.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5
            )
            for (x, y, w, h) in faces:
                if w > self.min_object_size and h > self.min_object_size:
                    rects.append((x, y, w, h))

        elif self.detection_mode == "OBJECT":
            mask = self.system.bg_subtractor.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > self.min_object_size and h > self.min_object_size:
                    rects.append((x, y, w, h))

        elif self.detection_mode == "MANUAL":
            if self.img_widget.normalized_roi:
                nx, ny, nw, nh = self.img_widget.normalized_roi
                frame_h, frame_w = frame.shape[:2]
                x = int(nx * frame_w)
                y = int((1.0 - (ny + nh)) * frame_h)
                w = int(nw * frame_w)
                h = int(nh * frame_h)
                if w > 10 and h > 10:
                    rects.append((x, y, w, h))

        elif self.detection_mode == "TEST":
            # Faces
            faces = self.system.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5
            )
            for (x, y, w, h) in faces:
                if w > self.min_object_size and h > self.min_object_size:
                    rects.append((x, y, w, h))

            # Moving objects
            mask = self.system.bg_subtractor.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > self.min_object_size and h > self.min_object_size:
                    rects.append((x, y, w, h))

            # Manual ROI (optional)
            if self.img_widget.normalized_roi:
                nx, ny, nw, nh = self.img_widget.normalized_roi
                frame_h, frame_w = frame.shape[:2]
                x = int(nx * frame_w)
                y = int((1.0 - (ny + nh)) * frame_h)
                w = int(nw * frame_w)
                h = int(nh * frame_h)
                if w > 10 and h > 10:
                    rects.append((x, y, w, h))

            # Merge overlapping rectangles so one body part isn't counted as multiple objects
            rects = self.merge_rectangles(rects, iou_threshold=0.4)

        # 2. recognition and drawing
        for (x, y, w, h) in rects:
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            roi_gray = gray[y : y + h, x : x + w]
            roi_resized = cv2.resize(roi_gray, (200, 200))
            self.current_rois.append(roi_resized)

            color = (255, 0, 0)
            label_text = "Unknown"

            if self.system.is_trained:
                try:
                    id_, conf = self.system.recognizer.predict(roi_resized)
                    if conf < self.conf_threshold:
                        name = self.system.labels_map.get(id_, "Unknown")
                        label_text = f"{name} ({int(conf)})"
                        color = (0, 255, 0)
                    else:
                        label_text = f"Unknown ({int(conf)})"
                except Exception:
                    pass

            # If user is actively capturing for some object, show that in color
            if self.current_object_name:
                color = (0, 255, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.img_widget.texture = texture

    # --- SAVE & TRAIN ---

    def save_current_object(self, instance):
        if not self.current_object_name or not self.current_rois:
            return
        save_path = os.path.join(DATA_FOLDER, self.current_object_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for roi in self.current_rois:
            filename = f"{self.current_object_name}_{self.capture_count}.jpg"
            full_path = os.path.join(save_path, filename)
            cv2.imwrite(full_path, roi)
            self.capture_count += 1
        self.btn_capture.text = f"Saved! ({self.capture_count})"

    def trigger_training(self, instance):
        if self.system.train():
            self.btn_train.text = "TRAINED! (Ready)"
            self.btn_train.background_color = (0, 1, 0, 1)
        else:
            self.btn_train.text = "Training Failed"

    def on_stop(self):
        self.capture.release()

if __name__ == "__main__":
    CameraApp().run()
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import pandas as pd
import math
import os

class VideoAnnotator:
    def __init__(self, master):
        self.master = master
        self.master.title("视频打标工具")

        # 初始化变量
        self.video_path = None
        self.cap = None
        self.fps = 0
        self.video_duration = 0
        self.current_time = 0
        self.annotations = []
        self.playing = False
        self.frame_id = None
        self.slider_dragging = False
        self.speed = 1.0  # 倍速，默认1.0

        self.last_annotation = {
            'road_name': "",
            'location': "路段中",
            'road_type': "常规",
            'divider': "无",
            'non_motor_lane': "无",
            'parking': "无",
            'crosswalk': "无",
            'signal': "无",
            'lane_count': "2",
            'other': ""
        }

        # 主容器，左右布局
        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧：视频显示与视频控制
        video_frame = tk.Frame(main_frame)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(video_frame)
        self.video_label.pack()

        # 修改：增加进度条长度
        self.progress_scale = tk.Scale(video_frame, orient=tk.HORIZONTAL, length=800,
                                       from_=0, to=100, resolution=0.1, command=self.on_slider_move)
        self.progress_scale.pack(pady=5)
        self.progress_scale.bind("<ButtonPress-1>", self.slider_press)
        self.progress_scale.bind("<ButtonRelease-1>", self.slider_release)

        # 绑定方向键控制进度
        self.master.bind("<Left>", self.seek_left)
        self.master.bind("<Right>", self.seek_right)
        self.master.bind("<space>", lambda event: self.toggle_play())

        video_btn_frame = tk.Frame(video_frame)
        video_btn_frame.pack(pady=5)
        self.load_btn = tk.Button(video_btn_frame, text="加载视频", command=self.load_video)
        self.load_btn.grid(row=0, column=0, padx=5)
        # 新增：视频倍速控制
        tk.Label(video_btn_frame, text="视频倍速:").grid(row=0, column=2, padx=5)
        self.speed_combo = ttk.Combobox(video_btn_frame, values=["0.5", "1.0", "1.5", "2.0", "3", "5"],
                                        state="readonly", width=5)
        self.speed_combo.grid(row=0, column=3, padx=5)
        self.speed_combo.set("1.0")
        self.speed_combo.bind("<<ComboboxSelected>>", self.change_speed)

        # 右侧：标签选择和显示区域（上部选择，下部显示）
        annotation_frame = tk.Frame(main_frame)
        annotation_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

        # 上部：标签选择区域（相关输入框）
        self.info_frame = tk.Frame(annotation_frame)
        self.info_frame.pack(side=tk.TOP, fill=tk.X)
        self.create_annotation_widgets()

        annotation_btn_frame = tk.Frame(annotation_frame)
        annotation_btn_frame.pack(side=tk.TOP, pady=5)
        self.play_pause_btn = tk.Button(annotation_btn_frame, text="播放/暂停", command=self.toggle_play)
        self.play_pause_btn.grid(row=0, column=0, padx=5)
        self.annotate_btn = tk.Button(annotation_btn_frame, text="打标", command=self.annotate)
        self.annotate_btn.grid(row=0, column=1, padx=5)
        self.edit_btn = tk.Button(annotation_btn_frame, text="编辑打标", command=self.edit_annotation)
        self.edit_btn.grid(row=0, column=2, padx=5)
        self.finish_btn = tk.Button(annotation_btn_frame, text="完成并输出", command=self.finish)
        self.finish_btn.grid(row=0, column=3, padx=5)
        # 新增：删除打标记录按钮
        self.delete_btn = tk.Button(annotation_btn_frame, text="删除打标", command=self.delete_annotation)
        self.delete_btn.grid(row=0, column=4, padx=5)

        # 下部：显示打标记录的 Listbox
        self.annotate_listbox = tk.Listbox(annotation_frame, width=70)
        self.annotate_listbox.pack(side=tk.BOTTOM, pady=5)
        
    def seek_left(self, event):
        if self.cap is None:
            return
        new_time = max(0, self.current_time - 5)
        self.progress_scale.set(new_time)
        self.on_slider_move(new_time)

    def seek_right(self, event):
        if self.cap is None:
            return
        new_time = min(self.video_duration, self.current_time + 5)
        self.progress_scale.set(new_time)
        self.on_slider_move(new_time)

    def delete_annotation(self):
        """删除选中的打标记录"""
        selection = self.annotate_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请选择一条打标记录进行删除。")
            return
        idx = selection[0]
        if messagebox.askyesno("确认删除", "确定要删除选中的打标记录吗？"):
            self.annotate_listbox.delete(idx)
            del self.annotations[idx]

    def create_annotation_widgets(self):
        """创建打标信息输入部分"""
        # 道路名称
        tk.Label(self.info_frame, text="道路名称:").grid(row=0, column=0, padx=5, pady=5)
        self.road_name_entry = tk.Entry(self.info_frame, width=30)
        self.road_name_entry.grid(row=0, column=1, padx=5, pady=5)
        self.road_name_entry.insert(0, self.last_annotation['road_name'])

        # 路段位置
        tk.Label(self.info_frame, text="路段位置:").grid(row=1, column=0, padx=5, pady=5)
        self.location_combo = ttk.Combobox(self.info_frame, values=["路段中", "交叉口", "其他"], state="readonly", width=27)
        self.location_combo.grid(row=1, column=1, padx=5, pady=5)
        self.location_combo.set(self.last_annotation['location'])

        # 路段类型
        tk.Label(self.info_frame, text="路段类型:").grid(row=2, column=0, padx=5, pady=5)
        self.road_type_combo = ttk.Combobox(self.info_frame, values=["常规", "三向", "四向", "五向", "其他"], state="readonly", width=27)
        self.road_type_combo.grid(row=2, column=1, padx=5, pady=5)
        self.road_type_combo.set(self.last_annotation['road_type'])

        # 中央分隔带
        tk.Label(self.info_frame, text="中央分隔带:").grid(row=3, column=0, padx=5, pady=5)
        self.divider_var = tk.StringVar(value=self.last_annotation['divider'])
        self.divider_yes = tk.Radiobutton(self.info_frame, text="有", variable=self.divider_var, value="有")
        self.divider_no = tk.Radiobutton(self.info_frame, text="无", variable=self.divider_var, value="无")
        self.divider_yes.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.divider_no.grid(row=3, column=2, padx=5, pady=5, sticky="w")

        # 非机动车道
        tk.Label(self.info_frame, text="非机动车道:").grid(row=4, column=0, padx=5, pady=5)
        self.non_motor_lane_var = tk.StringVar(value=self.last_annotation['non_motor_lane'])
        self.non_motor_lane_yes = tk.Radiobutton(self.info_frame, text="有", variable=self.non_motor_lane_var, value="有")
        self.non_motor_lane_no = tk.Radiobutton(self.info_frame, text="无", variable=self.non_motor_lane_var, value="无")
        self.non_motor_lane_yes.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.non_motor_lane_no.grid(row=4, column=2, padx=5, pady=5, sticky="w")

        # 路侧停车位
        tk.Label(self.info_frame, text="路侧停车位:").grid(row=5, column=0, padx=5, pady=5)
        self.parking_var = tk.StringVar(value=self.last_annotation['parking'])
        self.parking_yes = tk.Radiobutton(self.info_frame, text="有", variable=self.parking_var, value="有")
        self.parking_no = tk.Radiobutton(self.info_frame, text="无", variable=self.parking_var, value="无")
        self.parking_yes.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        self.parking_no.grid(row=5, column=2, padx=5, pady=5, sticky="w")

        # 路中人行横道
        tk.Label(self.info_frame, text="路中人行横道:").grid(row=6, column=0, padx=5, pady=5)
        self.crosswalk_var = tk.StringVar(value=self.last_annotation['crosswalk'])
        self.crosswalk_yes = tk.Radiobutton(self.info_frame, text="有", variable=self.crosswalk_var, value="有")
        self.crosswalk_no = tk.Radiobutton(self.info_frame, text="无", variable=self.crosswalk_var, value="无")
        self.crosswalk_yes.grid(row=6, column=1, padx=5, pady=5, sticky="w")
        self.crosswalk_no.grid(row=6, column=2, padx=5, pady=5, sticky="w")

        # 信号灯
        tk.Label(self.info_frame, text="信号灯:").grid(row=7, column=0, padx=5, pady=5)
        self.signal_var = tk.StringVar(value=self.last_annotation['signal'])
        self.signal_yes = tk.Radiobutton(self.info_frame, text="有", variable=self.signal_var, value="有")
        self.signal_no = tk.Radiobutton(self.info_frame, text="无", variable=self.signal_var, value="无")
        self.signal_yes.grid(row=7, column=1, padx=5, pady=5, sticky="w")
        self.signal_no.grid(row=7, column=2, padx=5, pady=5, sticky="w")

        # 车道数量
        tk.Label(self.info_frame, text="车道数量:").grid(row=8, column=0, padx=5, pady=5)
        self.lane_count_combo = ttk.Combobox(self.info_frame, values=["2", "4", "6", "8"], state="readonly", width=27)
        self.lane_count_combo.grid(row=8, column=1, padx=5, pady=5)
        self.lane_count_combo.set(self.last_annotation['lane_count'])

        # 其他
        tk.Label(self.info_frame, text="其他:").grid(row=9, column=0, padx=5, pady=5)
        self.other_entry = tk.Entry(self.info_frame, width=30)
        self.other_entry.grid(row=9, column=1, padx=5, pady=5)
        self.other_entry.insert(0, self.last_annotation['other'])
        def on_location_change(event):
            loc = self.location_combo.get()
            if loc == "路段中":
                self.road_type_combo.set("常规")
                self.road_type_combo.config(state="disabled")
                self.divider_yes.config(state="normal")
                self.divider_no.config(state="normal")
                self.non_motor_lane_yes.config(state="normal")
                self.non_motor_lane_no.config(state="normal")
                self.parking_yes.config(state="normal")
                self.parking_no.config(state="normal")
                self.lane_count_combo.config(state="readonly")
            elif loc == "交叉口":
                self.divider_var.set("无")
                self.non_motor_lane_var.set("无")
                self.parking_var.set("无")
                self.lane_count_combo.set("-")
                self.divider_yes.config(state="disabled")
                self.divider_no.config(state="disabled")
                self.non_motor_lane_yes.config(state="disabled")
                self.non_motor_lane_no.config(state="disabled")
                self.parking_yes.config(state="disabled")
                self.parking_no.config(state="disabled")
                self.lane_count_combo.config(state="disabled")
                self.road_type_combo.config(state="readonly")
            else:
                self.road_type_combo.config(state="readonly")
                self.divider_yes.config(state="normal")
                self.divider_no.config(state="normal")
                self.non_motor_lane_yes.config(state="normal")
                self.non_motor_lane_no.config(state="normal")
                self.parking_yes.config(state="normal")
                self.parking_no.config(state="normal")
                self.lane_count_combo.config(state="readonly")

        self.location_combo.bind("<<ComboboxSelected>>", on_location_change)

    def change_speed(self, event):
        try:
            self.speed = float(self.speed_combo.get())
        except ValueError:
            self.speed = 1.0

    def load_video(self):
        """加载视频文件并初始化相关参数"""
        self.video_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mkv")]
        )
        if not self.video_path:
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件！")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_duration = frame_count / self.fps if self.fps > 0 else 0
        self.current_time = 0

        self.progress_scale.config(to=self.video_duration)
        self.progress_scale.set(0)

        self.annotations = []
        self.annotate_listbox.delete(0, tk.END)

        self.playing = True
        self.play_video()

    def play_video(self):
        """循环读取视频帧，并更新画面与进度条"""
        if self.cap is None or not self.playing:
            return

        ret, frame = self.cap.read()
        if ret:
            self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            if not self.slider_dragging:
                self.progress_scale.set(self.current_time)

            delay = int(1000 / (self.fps * self.speed))
            self.frame_id = self.master.after(delay, self.play_video)
        else:
            self.playing = False
            self.cap.release()
            self.cap = None

    def toggle_play(self):
        """播放/暂停切换"""
        if self.cap is None:
            return
        if self.playing:
            self.playing = False
            if self.frame_id:
                self.master.after_cancel(self.frame_id)
        else:
            self.playing = True
            self.play_video()

    def slider_press(self, event):
        self.slider_dragging = True

    def slider_release(self, event):
        self.slider_dragging = False
        value = self.progress_scale.get()
        self.on_slider_move(value)

    def on_slider_move(self, value):
        if self.cap is None:
            return
        try:
            new_time = float(value)
        except ValueError:
            return
        self.cap.set(cv2.CAP_PROP_POS_MSEC, new_time * 1000)
        ret, frame = self.cap.read()
        if ret:
            self.current_time = new_time
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

    def annotate(self):
        """打标：记录当前时间和用户输入的标签信息，并记住上一次的打标内容"""
        if self.cap is None:
            messagebox.showwarning("警告", "请先加载视频！")
            return

        road_name = self.road_name_entry.get().strip()
        location = self.location_combo.get().strip()
        road_type = self.road_type_combo.get().strip()
        divider = self.divider_var.get().strip()
        non_motor_lane = self.non_motor_lane_var.get().strip()
        parking = self.parking_var.get().strip()
        crosswalk = self.crosswalk_var.get().strip()
        signal = self.signal_var.get().strip()
        lane_count = self.lane_count_combo.get().strip()
        other = self.other_entry.get().strip()

        self.last_annotation = {
            'road_name': road_name,
            'location': location,
            'road_type': road_type,
            'divider': divider,
            'non_motor_lane': non_motor_lane,
            'parking': parking,
            'crosswalk': crosswalk,
            'signal': signal,
            'lane_count': lane_count,
            'other': other
        }

        annotation_time = self.current_time

        # 插入到列表最前面，使新打标显示在第一行
        annotation_record = (annotation_time, road_name, location, road_type, divider,
                             non_motor_lane, parking, crosswalk, signal, lane_count, other)
        self.annotations.insert(0, annotation_record)
        display_text = (f"时间: {annotation_time:.2f}s, 道路名称: {road_name}, 路段位置: {location}, "
                        f"路段类型: {road_type}, 中央分隔带: {divider}, 非机动车道: {non_motor_lane}, "
                        f"路侧停车位: {parking}, 路中人行横道: {crosswalk}, 信号灯: {signal}, 车道数量: {lane_count}, "
                        f"其他: {other}")
        self.annotate_listbox.insert(0, display_text)

    def edit_annotation(self):
        """编辑历史打标记录的功能"""
        selection = self.annotate_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请选择一条打标记录进行编辑。")
            return
        idx = selection[0]
        annotation = list(self.annotations[idx])
        edit_window = tk.Toplevel(self.master)
        edit_window.title("编辑打标")
        tk.Label(edit_window, text=f"时间: {annotation[0]:.2f} 秒").grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        field_names = ["道路名称", "路段位置", "路段类型", "中央分隔带", "非机动车道",
                       "路侧停车位", "路中人行横道", "信号灯", "车道数量", "其他"]
        entries = []
        for i, field in enumerate(field_names, start=1):
            tk.Label(edit_window, text=field + ":").grid(row=i, column=0, padx=5, pady=5)
            entry = tk.Entry(edit_window, width=30)
            entry.grid(row=i, column=1, padx=5, pady=5)
            entry.insert(0, annotation[i])
            entries.append(entry)

        def save_changes():
            new_annotation = [annotation[0]]
            for entry in entries:
                new_annotation.append(entry.get().strip())
            self.annotations[idx] = tuple(new_annotation)
            display_text = f"时间: {new_annotation[0]:.2f}s, 道路名称: {new_annotation[1]}, 路段位置: {new_annotation[2]}, " \
                           f"路段类型: {new_annotation[3]}, 中央分隔带: {new_annotation[4]}, 非机动车道: {new_annotation[5]}, " \
                           f"路侧停车位: {new_annotation[6]}, 路中人行横道: {new_annotation[7]}, 信号灯: {new_annotation[8]}, 车道数量: {new_annotation[9]}, " \
                           f"其他: {new_annotation[10]}"
            self.annotate_listbox.delete(idx)
            self.annotate_listbox.insert(idx, display_text)
            edit_window.destroy()

        save_btn = tk.Button(edit_window, text="保存修改", command=save_changes)
        save_btn.grid(row=len(field_names)+1, column=0, columnspan=2, pady=10)

    def finish(self):
        """完成打标，生成每秒一个标签记录，并输出到 Excel 文件"""
        if self.video_duration == 0:
            messagebox.showwarning("警告", "视频未加载或无效！")
            return

        if self.cap:
            self.playing = False
            if self.frame_id:
                self.master.after_cancel(self.frame_id)
            self.cap.release()
            self.cap = None

        if not self.annotations:
            if not messagebox.askyesno("确认", "未进行任何打标，是否输出空白标签？"):
                return

        annotations_sorted = sorted(self.annotations, key=lambda x: x[0])
        intervals = []
        if annotations_sorted:
            intervals.append((0, annotations_sorted[0][0], annotations_sorted[0][1:]))
            for i in range(1, len(annotations_sorted)):
                intervals.append((annotations_sorted[i-1][0], annotations_sorted[i][0], annotations_sorted[i][1:]))
                if len(annotations_sorted) > 1:
                    intervals.append((annotations_sorted[-1][0], self.video_duration, annotations_sorted[-2][1:]))
                else:
                    intervals.append((annotations_sorted[-1][0], self.video_duration, annotations_sorted[-1][1:]))
        else:
            intervals.append((0, self.video_duration, ("", "", "", "", "", "", "", "", "", "", "")))

        data = []
        total_seconds = math.ceil(self.video_duration)
        for sec in range(total_seconds):
            t = sec + 0.5
            labels = ("", "", "", "", "", "", "", "", "", "", "")
            for start, end, label_tuple in intervals:
                if t >= start and t < end:
                    labels = label_tuple
                    break
            data.append({
                "Time (s)": sec,
                "Road Name": labels[0],
                "Location": labels[1],
                "Road Type": labels[2],
                "Divider": labels[3],
                "Non Motor Lane": labels[4],
                "Parking": labels[5],
                "Crosswalk": labels[6],
                "Signal": labels[7],
                "Lane Count": labels[8],
                "Other": labels[9]
            })

        df = pd.DataFrame(data)
        default_filename = os.path.splitext(os.path.basename(self.video_path))[0] + ".xlsx"
        save_path = filedialog.asksaveasfilename(
            title="保存Excel文件",
            initialfile=default_filename,
            defaultextension=".xlsx",
            filetypes=[("Excel文件", "*.xlsx")]
        )
        if save_path:
            try:
                df.to_excel(save_path, index=False)
                messagebox.showinfo("完成", f"标签已输出到 {save_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件失败: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotator(root)
    root.mainloop()

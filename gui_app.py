import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from cow_detector import CowDetectorClassifier

class CowDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cow Detection & Breed Classification")
        self.root.geometry("1200x800")
        
        self.detector = None
        self.cap = None
        self.is_running = False
        self.current_frame = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(control_frame, text="Load Model", command=self.load_model).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Start Webcam", command=self.start_webcam).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Load Video", command=self.load_video).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Stop", command=self.stop_detection).grid(row=0, column=3, padx=5)
        
        # Video display
        self.video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        self.video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.video_label = ttk.Label(self.video_frame, text="No video loaded")
        self.video_label.pack()
        
        # Results panel
        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results tree
        self.results_tree = ttk.Treeview(results_frame, columns=("Breed", "Confidence"), show="tree headings")
        self.results_tree.heading("#0", text="Cow #")
        self.results_tree.heading("Breed", text="Breed")
        self.results_tree.heading("Confidence", text="Confidence")
        self.results_tree.column("#0", width=80)
        self.results_tree.column("Breed", width=120)
        self.results_tree.column("Confidence", width=100)
        
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken")
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Breed Classification Model",
            filetypes=[("H5 files", "*.h5"), ("All files", "*.*")],
            initialdir="../New"
        )
        
        if model_path:
            try:
                self.detector = CowDetectorClassifier(model_path)
                self.status_var.set(f"Model loaded: {model_path}")
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                
    def start_webcam(self):
        if not self.detector:
            messagebox.showwarning("Warning", "Please load model first!")
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam!")
            return
            
        self.is_running = True
        self.status_var.set("Webcam started")
        threading.Thread(target=self.video_loop, daemon=True).start()
        
    def load_video(self):
        if not self.detector:
            messagebox.showwarning("Warning", "Please load model first!")
            return
            
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video file!")
                return
                
            self.is_running = True
            self.status_var.set(f"Video loaded: {video_path}")
            threading.Thread(target=self.video_loop, daemon=True).start()
            
    def stop_detection(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.status_var.set("Stopped")
        
    def video_loop(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process frame
            results = self.detector.process_frame(frame)
            
            # Draw results
            display_frame = self.detector.draw_results(frame.copy(), results)
            
            # Update UI
            self.update_video_display(display_frame)
            self.update_results_table(results)
            
        self.is_running = False
        
    def update_video_display(self, frame):
        # Resize frame for display
        height, width = frame.shape[:2]
        max_width, max_height = 640, 480
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo
        
    def update_results_table(self, results):
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        # Add new results
        for i, result in enumerate(results, 1):
            breed = result['breed']
            confidence = f"{result['breed_conf']:.2f}"
            self.results_tree.insert("", "end", text=f"Cow {i}", values=(breed, confidence))
            
        # Update status
        self.status_var.set(f"Detected {len(results)} cows")

def main():
    root = tk.Tk()
    app = CowDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
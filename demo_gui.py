#!/usr/bin/env python3
"""
Graphical wrapper around demo.py for showcasing the ASR pipeline end-to-end.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from demo import (
    load_artifacts,
    prepare_features,
    predict_phrase,
    play_audio,
    speak_text,
    discover_demo_samples,
)
from real_phrases import phrases
from train_combined_model import SAMPLE_RATE, extract_label


class DemoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Speech Recognition Demo for Cerebral Palsy")
        self.root.configure(bg="#f5f5f5")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 750)

        try:
            self.model, self.scaler = load_artifacts()
        except SystemExit as exc:
            messagebox.showerror("Artifacts Missing", str(exc))
            root.destroy()
            raise

        try:
            self.test_files = discover_demo_samples()
        except RuntimeError as exc:
            messagebox.showerror("No Demo Audio", str(exc))
            root.destroy()
            raise SystemExit(1)
        self.sample_lookup = {path.name: path for path in self.test_files}

        self.total_runs = 0
        self.correct_runs = 0
        self.is_running = threading.Event()

        self.selected_file = tk.StringVar(value=self.test_files[0].name)
        self.confidence_var = tk.DoubleVar(value=0.0)
        self.confidence_label_var = tk.StringVar(value="Confidence: --")
        self.accuracy_var = tk.StringVar(value="Accuracy: 0/0 (--)")

        self._build_layout()

    def _build_layout(self) -> None:
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=20, pady=(20, 10))

        ttk.Label(control_frame, text="Select Test Audio:", font=("Helvetica", 14)).pack(
            side="left"
        )
        self.file_combo = ttk.Combobox(
            control_frame,
            values=[p.name for p in self.test_files],
            state="readonly",
            textvariable=self.selected_file,
            width=40,
            font=("Helvetica", 14),
        )
        self.file_combo.pack(side="left", padx=10)

        run_button = tk.Button(
            control_frame,
            text="RUN DEMO",
            command=self.start_single_run,
            bg="#2e7d32",
            fg="white",
            activebackground="#1b5e20",
            font=("Helvetica", 16, "bold"),
            padx=25,
            pady=10,
        )
        run_button.pack(side="left", padx=20)

        run_all_button = tk.Button(
            control_frame,
            text="Run All Tests",
            command=self.start_run_all,
            bg="#1565c0",
            fg="white",
            activebackground="#0d47a1",
            font=("Helvetica", 14, "bold"),
            padx=20,
            pady=10,
        )
        run_all_button.pack(side="left", padx=10)

        accuracy_label = ttk.Label(
            control_frame, textvariable=self.accuracy_var, font=("Helvetica", 14, "bold")
        )
        accuracy_label.pack(side="right")

        waveform_frame = ttk.Frame(self.root)
        waveform_frame.pack(fill="both", expand=True, padx=20, pady=10)

        figure = Figure(figsize=(8, 3), dpi=100)
        self.wave_ax = figure.add_subplot(111)
        self.wave_ax.set_title("Waveform")
        self.wave_ax.set_xlabel("Time (s)")
        self.wave_ax.set_ylabel("Amplitude")
        self.wave_ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(figure, master=waveform_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        self.status_text = tk.Text(
            status_frame,
            height=12,
            wrap="word",
            font=("Helvetica", 16),
            bg="#ffffff",
            state="disabled",
        )
        self.status_text.pack(fill="both", expand=True)

        self.status_text.tag_configure("info", foreground="#1976d2")
        self.status_text.tag_configure("process", foreground="#6a1b9a")
        self.status_text.tag_configure("prediction", font=("Helvetica", 24, "bold"))
        self.status_text.tag_configure(
            "correct",
            foreground="#1b5e20",
            background="#c8e6c9",
        )
        self.status_text.tag_configure(
            "incorrect",
            foreground="#b71c1c",
            background="#ffcdd2",
        )
        self.status_text.tag_configure("default", foreground="#212121")
        self.status_text.tag_configure("error", foreground="#d32f2f")

        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill="x", padx=20, pady=(0, 20))

        self.confidence_progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.confidence_var,
        )
        self.confidence_progress.pack(fill="x", pady=5)

        ttk.Label(
            progress_frame,
            textvariable=self.confidence_label_var,
            font=("Helvetica", 14, "bold"),
        ).pack()

    def append_status(self, message: str, tag: str = "default") -> None:
        def writer() -> None:
            self.status_text.configure(state="normal")
            self.status_text.insert("end", message + "\n", tag)
            self.status_text.see("end")
            self.status_text.configure(state="disabled")

        self.status_text.after(0, writer)

    def clear_status(self) -> None:
        def clearer() -> None:
            self.status_text.configure(state="normal")
            self.status_text.delete("1.0", "end")
            self.status_text.configure(state="disabled")

        self.status_text.after(0, clearer)

    def update_confidence(self, value: float) -> None:
        def updater() -> None:
            self.confidence_var.set(value)
            self.confidence_label_var.set(f"Confidence: {value:.2f}%")

        self.root.after(0, updater)

    def update_accuracy(self) -> None:
        def updater() -> None:
            if self.total_runs:
                percent = (self.correct_runs / self.total_runs) * 100
                text = f"Accuracy: {self.correct_runs}/{self.total_runs} ({percent:.1f}%)"
            else:
                text = "Accuracy: 0/0 (--)"
            self.accuracy_var.set(text)

        self.root.after(0, updater)

    def update_waveform(self, audio: np.ndarray, sr: int) -> None:
        def drawer() -> None:
            self.wave_ax.clear()
            duration = len(audio) / sr
            times = np.linspace(0, duration, num=len(audio))
            self.wave_ax.plot(times, audio, color="#1976d2")
            self.wave_ax.set_title("Waveform")
            self.wave_ax.set_xlabel("Time (s)")
            self.wave_ax.set_ylabel("Amplitude")
            self.wave_ax.grid(True, alpha=0.3)
            self.canvas.draw_idle()

        self.root.after(0, drawer)

    def start_single_run(self) -> None:
        if self.is_running.is_set():
            messagebox.showinfo("In Progress", "A demo is already running.")
            return

        file_name = self.selected_file.get()
        audio_path = self.sample_lookup.get(file_name)
        if audio_path is None or not audio_path.is_file():
            messagebox.showerror("Missing File", f"{file_name} was not found.")
            return

        self.is_running.set()
        thread = threading.Thread(
            target=self._run_demo_sequence, args=(audio_path, True), daemon=True
        )
        thread.start()

    def start_run_all(self) -> None:
        if self.is_running.is_set():
            messagebox.showinfo("In Progress", "A demo is already running.")
            return

        self.is_running.set()

        def runner() -> None:
            self.clear_status()
            self.update_confidence(0.0)
            local_correct = 0
            local_total = 0
            self.append_status("Running all test clips...\n", "process")
            for path in self.test_files:
                self.root.after(0, lambda p=path: self.selected_file.set(p.name))
                correct = self._run_demo_sequence(path, False)
                if correct is not None:
                    local_total += 1
                    local_correct += int(correct)
                self.append_status("-" * 60, "default")
            if local_total:
                self.correct_runs += local_correct
                self.total_runs += local_total
                self.update_accuracy()
            self.append_status("Finished running all tests.", "process")
            self.is_running.clear()

        threading.Thread(target=runner, daemon=True).start()

    def _run_demo_sequence(self, audio_path: Path, update_global_accuracy: bool) -> Optional[bool]:
        try:
            self.clear_status()
            self.update_confidence(0.0)
            self.append_status(f"Selected file: {audio_path.name}", "default")

            try:
                audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
                self.update_waveform(audio, sr)
            except Exception as exc:
                self.append_status(f"Waveform load error: {exc}", "error")

            self.append_status("Playing original audio...", "info")
            play_audio(audio_path)

            self.append_status("\nProcessing through model...", "process")
            feature_vector, duration, scaled = prepare_features(audio_path, self.scaler)
            label, confidence = predict_phrase(self.model, scaled)
            predicted_phrase = phrases.get(label, f"[Unknown label {label}]")

            try:
                true_label = extract_label(audio_path.name)
                expected_phrase = phrases.get(true_label, f"[Unknown label {true_label}]")
            except Exception:
                true_label = None
                expected_phrase = "Unknown"

            correct = true_label == label if true_label is not None else None

            self.append_status(f"Duration: {duration:.2f}s", "default")
            if expected_phrase is not None:
                self.append_status(f"Expected: {expected_phrase}", "default")

            prediction_tag = "prediction"
            if correct is True:
                prediction_tag = ("prediction", "correct")
            elif correct is False:
                prediction_tag = ("prediction", "incorrect")

            self.append_status(f"Prediction: {predicted_phrase}", prediction_tag)
            self.append_status(f"Confidence: {confidence:.2f}%", "default")
            self.update_confidence(confidence)

            self.append_status("\nSpeaking prediction...", "process")
            speak_text(predicted_phrase, rate=175, volume=1.0)

            if update_global_accuracy and correct is not None:
                self.total_runs += 1
                self.correct_runs += int(correct)
                self.update_accuracy()

            self.append_status("\nDemo complete!", "info")
            return correct

        except Exception as exc:
            self.append_status(f"Demo error: {exc}", "error")
            return None
        finally:
            if update_global_accuracy:
                self.is_running.clear()


def main() -> None:
    root = tk.Tk()
    try:
        DemoApp(root)
        root.mainloop()
    except SystemExit:
        pass


if __name__ == "__main__":
    main()

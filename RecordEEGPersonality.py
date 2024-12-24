import tkinter as tk
from tkinter import messagebox
from pylsl import resolve_stream, StreamInlet
import csv
import time
import threading


class EEGRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Muse 2 EEG Recorder")
        self.stream = None
        self.inlet = None
        self.recording = False
        self.start_time = None
        self.data = []
        self.participant_id = None
        self.durations = {}
        self.stopwatch_running = False

        # GUI Layout
        self.status_label = tk.Label(root, text="Status: Not connected to Muse 2")
        self.status_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        self.text_field_frame = None  # Placeholder for the text field frame
        self.buttons = {}

        # Stopwatch Label
        self.stopwatch_label = tk.Label(root, text="Stopwatch: 00:00:00", font=("Arial", 14))
        self.stopwatch_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        for i, (label, key) in enumerate([
            ("Record Eyes Closed", "EC"),
            ("Record Eyes Opened", "EO"),
            ("Record Personality 1", "Personality1"),
            ("Record Personality 2", "Personality2"),
            ("Record Personality 3", "Personality3")
        ]):
            button = tk.Button(root, text=label, state=tk.DISABLED, command=lambda k=key: self.toggle_recording(k))
            button.grid(row=i + 3, column=0, columnspan=2, pady=5)
            self.buttons[key] = button

        # Start monitoring Muse connection
        threading.Thread(target=self.monitor_muse_connection, daemon=True).start()

    def monitor_muse_connection(self):
        """Continuously monitor the connection to the Muse 2 headband."""
        connected = False
        while True:
            try:
                streams = resolve_stream('type', 'EEG')
                if streams:
                    if not connected:
                        self.stream = streams[0]
                        self.inlet = StreamInlet(self.stream)
                        self.on_connection()
                        connected = True
                else:
                    if connected:
                        self.on_disconnection()
                        connected = False
            except Exception as e:
                if connected:
                    self.on_disconnection()
                    connected = False
            time.sleep(1)

    def on_connection(self):
        """Handle actions to perform when Muse 2 connects."""
        self.status_label.config(text="Status: Connected to Muse 2")
        self.show_text_field()

    def on_disconnection(self):
        """Handle actions to perform when Muse 2 disconnects."""
        self.status_label.config(text="Status: Not connected to Muse 2")
        self.hide_text_field()
        self.disable_buttons()

    def show_text_field(self):
        """Show the text field for entering participant ID."""
        if not self.text_field_frame:
            self.text_field_frame = tk.Frame(self.root)
            self.text_field_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

            tk.Label(self.text_field_frame, text="Participant ID:").grid(row=0, column=0, padx=5, pady=5)
            self.participant_id_entry = tk.Entry(self.text_field_frame)
            self.participant_id_entry.grid(row=0, column=1, padx=5, pady=5)
            self.participant_id_entry.bind("<KeyRelease>", self.enable_buttons)
        self.text_field_frame.grid()

    def hide_text_field(self):
        """Hide the text field for participant ID."""
        if self.text_field_frame:
            self.text_field_frame.grid_remove()

    def disable_buttons(self):
        """Disable all recording buttons."""
        for button in self.buttons.values():
            button.config(state=tk.DISABLED)

    def enable_buttons(self, event=None):
        """Enable the first button if participant ID is entered."""
        self.participant_id = self.participant_id_entry.get().strip()
        if self.participant_id:
            self.buttons["EC"].config(state=tk.NORMAL)

    def toggle_recording(self, key):
        """Start or stop recording EEG data."""
        if self.recording:
            # Stop recording
            self.recording = False
            self.stopwatch_running = False
            self.save_data(key)
            self.reset_stopwatch()
            self.buttons[key].config(text=f"Record {key.replace('_', ' ')}", state=tk.DISABLED)
            next_key = self.get_next_key(key)
            if next_key:
                self.buttons[next_key].config(state=tk.NORMAL)
        else:
            # Start recording
            self.recording = True
            self.start_time = time.time()
            self.data = []
            self.start_stopwatch()
            threading.Thread(target=self.record_data, daemon=True).start()
            self.buttons[key].config(text="Stop Recording")

    def record_data(self):
        """Continuously pull EEG data while recording."""
        while self.recording:
            sample, timestamp = self.inlet.pull_sample()
            self.data.append([timestamp] + sample)

    def save_data(self, key):
        """Save recorded EEG data to a CSV file."""
        end_time = time.time()
        duration = end_time - self.start_time
        self.durations[key] = duration

        filename = f"{self.participant_id}_{key}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write headers
            writer.writerow(["timestamps", "TP9", "AF7", "AF8", "TP10", "Right AUX"])
            # Write data
            writer.writerows(self.data)

        # Update duration info
        self.update_duration_file()

        messagebox.showinfo("Recording Saved", f"{key} data saved to {filename}")

    def update_duration_file(self):
        """Update the duration file with recording details."""
        filename = f"{self.participant_id}_duration.txt"
        with open(filename, 'w') as f:
            f.write(f"Participant ID: {self.participant_id}\n")
            for key, duration in self.durations.items():
                f.write(f"{key}: {duration:.2f} seconds\n")

    def start_stopwatch(self):
        """Start the stopwatch."""
        self.stopwatch_running = True
        threading.Thread(target=self.update_stopwatch, daemon=True).start()

    def update_stopwatch(self):
        """Update the stopwatch display."""
        start_time = time.time()
        while self.stopwatch_running:
            elapsed_time = int(time.time() - start_time)
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.stopwatch_label.config(text=f"Stopwatch: {hours:02}:{minutes:02}:{seconds:02}")
            time.sleep(1)

    def reset_stopwatch(self):
        """Reset the stopwatch display."""
        self.stopwatch_label.config(text="Stopwatch: 00:00:00")

    def get_next_key(self, current_key):
        """Get the next recording key in sequence."""
        keys = list(self.buttons.keys())
        current_index = keys.index(current_key)
        return keys[current_index + 1] if current_index + 1 < len(keys) else None


if __name__ == "__main__":
    root = tk.Tk()
    app = EEGRecorderApp(root)
    root.mainloop()

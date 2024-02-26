import cv2
from ultralytics import YOLO
from collections import defaultdict
import csv
from datetime import datetime

# Load the YOLOv8 model
model = YOLO('track.pt')

# Load video
video_path = 'test2.mp4'
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

entry_times = {}  # Dictionary to store entry times
durations = {}   # Dictionary to store detected durations
tracked_list = []

# Create a CSV file to save object durations
csv_file = open("customer_waiting_duration.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Track ID", "Entry Time", "Exit Time", "Duration (seconds)"])

# Create a CSV file to save alerts
alert_file = open("alerts.csv", mode="w", newline="")
alert_writer = csv.writer(alert_file)
alert_writer.writerow(["Alert", "At Time"])

# Get frames per second of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Timer variables for alert
alert_timer = defaultdict(lambda: 0)
alert_interval = 2  # Alert every 2 seconds
alert_threshold = 10 # Alert if waiting time exceeds 5 seconds

# Variables for highest waited customer
max_wait_time = 0
max_wait_id = None

# Variables for total waiting time and total number of customers
total_waiting_time = 0
total_customers = 0

customer_threshold = 2  # Alert if customers exceed this threshold

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        frame = cv2.resize(frame, (1300, 800))

        results = model.track(frame, persist=True)

        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = current_frame / fps

        boxes = results[0].boxes.xywh.cpu()
        labels = results[0].boxes.data.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        person_count = len(labels)
        # print(len(labels))
        for track_id, box in zip(track_ids, boxes):
            # print(person_count)
            x, y, w, h = box

            if track_id not in entry_times:
                entry_times[track_id] = current_time
                total_customers += 1

            else:
                exit_time = current_time
                duration = exit_time - entry_times[track_id]
                durations[track_id] = duration
                total_waiting_time += duration

                if duration > alert_threshold:
                    if current_time - alert_timer[track_id] > alert_interval:
                        print("\033[91m" + f"Alert: ID {track_id} has been waiting for {duration} seconds." + "\033[0m")
                        alert_timer[track_id] = current_time
                        current = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        alert_writer.writerow([f"{track_id} has been waiting for more then 10 minutes", current
                        ])

                if duration > max_wait_time:
                    max_wait_time = duration
                    max_wait_id = track_id

            cv2.putText(frame, f"ID: {track_id}, time: {durations.get(track_id, 0):.2f} sec", (int(x), int(y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        for track_id in tracked_list:
            if track_id not in track_ids:
                entry_times.pop(track_id, None)

        tracked_list = track_ids

        cv2.imshow("Customer Wait Time Tracking", frame)

        if max_wait_id is not None:
            print(f"Highest waited customer (ID {max_wait_id}) waited for {max_wait_time:.2f} seconds.")

        # print(f"Total number of customers: {total_customers}")

        if person_count > customer_threshold:
            print("\033[91m" + f"Alert: Number of customers exceeds {person_count} detected!" + "\033[0m")
            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            alert_writer.writerow([
            f"Number of customers exceeds{person_count}", time_now
            ])

        if total_customers > 0:
            avg_waiting_time = total_waiting_time / total_customers
            average_waiting_time = avg_waiting_time / 60
            print(f"Average waiting time: {average_waiting_time:.2f} seconds")
        else:
            print("No customers detected.")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

for track_id, duration in durations.items():
    entry_time = entry_times.get(track_id, 0)
    exit_time = entry_time + duration

    if duration > 50:  # If waiting time exceeds 10 minutes (600 seconds)
        # current = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([
            track_id, entry_time, exit_time, duration
        ])
    else:
        csv_writer.writerow([
            track_id, entry_time, exit_time, duration, "", ""
        ])

csv_file.close()
cap.release()
cv2.destroyAllWindows()

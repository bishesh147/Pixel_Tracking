import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import easygui

# Function to process video and track points
def process_video(cap, p0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(500, 700), maxLevel=0, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    if not ret:
        print("Failed to read the video file.")
        return None, None, None, None, None
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    if len(p0) == 0:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=10, qualityLevel=0.2, minDistance=10, blockSize=7)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    df = pd.DataFrame()
    x = None
    y = None
    ref_frame = None
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, stt, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[stt == 1]
            good_old = p0[stt == 1]

        # draw the tracks
        x = [[] for _ in good_new] if x is None else x
        y = [[] for _ in good_new] if y is None else y
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            x[i].append(a)
            y[i].append(b)
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            cv2.putText(frame, str(i), (int(a - 25), int(b - 25)), font, 1.25, (255, 255, 255), 3)

        img = cv2.add(frame, mask)

        if ref_frame is None:
            ref_frame = img

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        frame_number += 1

    time = [frame_number / fps for frame_number in range(len(x[0]))]

    df["time/s"] = time
    for i in range(len(x)):
        df[f"x{i}"] = x[i]
        df[f"y{i}"] = y[i]

    return df, ref_frame, x, y, fps

# Get video file path
video_path = easygui.fileopenbox("Input the full path of the video you want to track!")
if video_path is None:
    print("No video file selected.")
    exit()

cap = cv2.VideoCapture(str(video_path))

# Extract the video title from the file path
video_title = os.path.splitext(os.path.basename(video_path))[0]

# Specify points manually
cv2.namedWindow('frame test', cv2.WINDOW_AUTOSIZE)
ref, firstframe = cap.read()
height, width, channels = firstframe.shape
old_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
all_points = list()

def mouse_events(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        all_points.append([np.float32(x), np.float32(y)])
        cv2.circle(firstframe, (x, y), 5, (255, 255, 255), -1)
        cv2.putText(firstframe, f"({x}, {y})", (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3)

cv2.setMouseCallback('frame test', mouse_events)

while True:
    cv2.imshow('frame test', firstframe)
    key = cv2.waitKey(20) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
all_points = np.array(all_points, dtype=np.float32).reshape(-1, 1, 2)

# Process video and track points
df, ref_frame, x, y, fps = process_video(cap, all_points)
if df is not None:
    csv_filename = f"{video_title}_all_points.csv"
    df.to_csv(csv_filename, index=False)

    # Plot and save figures
    fig_dir = "Plots"
    os.makedirs(fig_dir, exist_ok=True)
    
    for i in range(len(x)):
        frameX = [j / fps for j in range(len(x[i]))]
        frameY = [j / fps for j in range(len(x[i]))]

        point_coordinates = f"({int(all_points[i][0][0])},{int(all_points[i][0][1])})"

        figureX = plt.figure()
        plt.title(f"Graph for point {point_coordinates} in the X axis")
        plt.xlabel("time/sec")
        plt.ylabel("position/pixel")
        plt.plot(frameX, x[i])
        figureX.savefig(f"{fig_dir}/{video_title}_{point_coordinates}_X.png")
        plt.close(figureX)

        figureY = plt.figure()
        plt.title(f"Graph for point {point_coordinates} in the Y axis")
        plt.xlabel("time/sec")
        plt.ylabel("position/pixel")
        plt.plot(frameY, y[i])
        figureY.savefig(f"{fig_dir}/{video_title}_{point_coordinates}_Y.png")
        plt.close(figureY)

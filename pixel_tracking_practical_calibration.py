import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import easygui
import math

def select_points(frame):
    """Allows the user to manually select points on the first frame."""
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([np.float32(x), np.float32(y)])
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
            cv2.putText(frame, f"({x}, {y})", (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3)

    cv2.namedWindow('Select Points to track', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Select Points to track', mouse_callback)

    while True:
        cv2.imshow('Select Points to track', frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)

# Function to process video and track points
def calculate_optical_flow(cap, p0):
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

def save_tracking_data(video_title, df, x, y, fps, initial_points):
    """Saves the tracking data to a CSV file and plots the trajectories."""
    if df is not None:
        csv_filename = f"{video_title}_all_points.csv"
        df.to_csv(csv_filename, index=False)

    # Plot and save figures
    fig_dir = "Plots"
    os.makedirs(fig_dir, exist_ok=True)
    
    for i in range(len(x)):
        frameX = [j / fps for j in range(len(x[i]))]
        frameY = [j / fps for j in range(len(x[i]))]

        point_coordinates = f"({int(initial_points[i][0][0])},{int(initial_points[i][0][1])})"

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

def get_scaling_factor(video_path):
    def collect_points(event, x, y, flags, param):
        points, image = param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Calibration", image)
        
    def calculate_distance(point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    def convert_to_millimeters(distance, unit):
        conversion_factors = {
            "mm": 1,
            "cm": 10,
            "meter": 1000,
            "feet": 304.8
        }
        return distance * conversion_factors[unit]
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video file.")
        return None

    points = []
    cv2.imshow("Select points with known distance", frame)
    cv2.setMouseCallback("Select points with known distance", collect_points, [points, frame])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 2:
        print("You need to select exactly two points.")
        return None

    # Input real-life distance and select the unit
    real_distance_str = easygui.enterbox("Enter the real-life distance between the selected points:")
    if not real_distance_str:
        print("No distance entered.")
        return None

    try:
        real_distance = float(real_distance_str)
    except ValueError:
        print("Invalid distance entered.")
        return None

    unit = easygui.choicebox("Select the unit of measurement:", choices=["mm", "cm", "meter", "feet"])
    if not unit:
        print("No unit selected.")
        return None

    real_distance_mm = convert_to_millimeters(real_distance, unit)

    pixel_distance = calculate_distance(points[0], points[1])
    scaling_factor = real_distance_mm / pixel_distance

    return scaling_factor


def convert_to_real_life(video_title, scaling_factor):
    # Load the CSV file into a DataFrame
    csv_filename = f"{video_title}_all_points.csv"
    df = pd.read_csv(csv_filename)

    # Assign the first values of x0 and y0 as references
    ref_x0 = df['x0'].iloc[0]
    ref_y0 = df['y0'].iloc[0]

    # Subtract the reference values from the rest of the data
    df['x0'] = df['x0'] - ref_x0
    df['y0'] = df['y0'] - ref_y0

    df['x0'] = df['x0'] * scaling_factor
    df['y0'] = df['y0'] * scaling_factor


    # Save the new DataFrame to a CSV file
    df.to_csv(f'real_{video_title}_all_points.csv', index=False)

    # Save the new DataFrame to an Excel file
    df.to_excel(f'real_{video_title}_all_points.xlsx', index=False)


def main():
    video_path = easygui.fileopenbox("Input the full path of the video you want to track!")
    if video_path is None:
        print("No video file selected.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open the video file.")
        return

    video_title = os.path.splitext(os.path.basename(video_path))[0]
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video file.")
        return
    
    scaling_factor = get_scaling_factor(video_path)

    initial_points = select_points(first_frame)
    df, ref_frame, x, y, fps = calculate_optical_flow(cap, initial_points)
    save_tracking_data(video_title, df, x, y, fps, initial_points)
    convert_to_real_life(video_title, scaling_factor)

if __name__ == "__main__":
    main()

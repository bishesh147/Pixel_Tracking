import cv2
import easygui

# Replace 'path_to_video.mp4' with the actual path to your video file
file_path = easygui.fileopenbox("Input the full path of the video you want to track!")
cap = cv2.VideoCapture(file_path)

num = 0

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
interval = int(fps)  # Calculate the interval in frames for saving an image every second

frame_count = 0

while cap.isOpened():
    success, img = cap.read()

    if not success:
        break

    # Save the frame every 'interval' frames
    if frame_count % interval == 0:
        cv2.imwrite('images/img' + str(num) + '.png', img)
        print(f"Image {num} saved!")
        num += 1

    frame_count += 1

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()

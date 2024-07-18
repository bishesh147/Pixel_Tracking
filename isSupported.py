import cv2
import easygui

def is_supported_format(file_path):
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():
        cap.release()
        return True
    else:
        return False

# Example usage:
file_path = easygui.fileopenbox("Input the full path of the video you want to track!")

if is_supported_format(file_path):
    print(f"The file '{file_path}' is a supported format.")
else:
    print(f"The file '{file_path}' is not a supported format.")

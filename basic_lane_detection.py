import cv2
import numpy as np

cap = cv2.VideoCapture(r'D:\computer vision\drive-download-20251028T233500Z-1-001\Basic-Lane-Detection\solidYellowLeft.mp4')

# Read first frame to get width and height
ret, frame = cap.read()
if not ret:
    print("Error: Couldn't read video.")
    cap.release()
    exit()

height, width = frame.shape[:2]

# Define VideoWriter outside the loop
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset video to start

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = np.zeros((height, width), dtype=np.uint8)
    trapezoid = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.9), height),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.4), int(height * 0.6))
    ]], dtype=np.int32)
    cv2.fillPoly(mask, trapezoid, 255)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (13, 13), 0)
    canny = cv2.Canny(blur, 50, 150)

    masked = cv2.bitwise_and(canny, mask)
    contours, _ = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # write the processed frame
    out.write(frame)

    cv2.imshow('Lane Detection', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
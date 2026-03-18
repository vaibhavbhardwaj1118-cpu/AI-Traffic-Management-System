import cv2

# Load video
cap = cv2.VideoCapture("traffic.mp4")

vehicle_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Simple threshold (basic detection)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # filter noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            vehicle_count += 1

    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Traffic Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

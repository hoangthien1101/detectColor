# import cv2
# import numpy as np
#
# # Hàm ánh xạ tọa độ từ khoảng pixel sang khoảng [-100, 100]
# def map_to_range(x, min_input, max_input):
#     # Ánh xạ tọa độ trong khoảng [0, max_input] vào khoảng [-100, 100]
#     return int((x - min_input) * (200 / (max_input - min_input)) - 100)
#
# # Phát hiện màu, trả về tọa độ tâm và vẽ khung bao quanh
# def detect_color(frame, lower_bound, upper_bound, min_area=500):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, lower_bound, upper_bound)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     max_contour = None
#     max_area = 0
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > min_area and area > max_area:  # Chỉ lấy contour lớn nhất
#             max_contour = contour
#             max_area = area
#
#     if max_contour is not None:
#         x, y, w, h = cv2.boundingRect(max_contour)
#         # Tính tâm của khung hình
#         center_x = x + w // 2
#         center_y = y + h // 2
#
#         # Ánh xạ tọa độ X và Y từ pixel sang khoảng [-100, 100]
#         frame_width = frame.shape[1]
#         frame_height = frame.shape[0]
#         mapped_x = map_to_range(center_x, 0, frame_width)
#         mapped_y = map_to_range(center_y, 0, frame_height)
#
#         # Vẽ khung bao quanh đối tượng và tâm
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ hình chữ nhật bao quanh đối tượng
#         cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), 2)  # Vẽ điểm tâm của đối tượng
#         cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)  # Đường ngang qua tâm
#         cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)  # Đường dọc qua tâm
#
#         # Trả về tọa độ đã ánh xạ
#         return mapped_x, mapped_y
#     return None
#
#
# # Tính giá trị trung bình có trọng số để làm mượt tọa độ
# def weighted_average(current_coords, previous_coords, weight=0.8):
#     if previous_coords is None:
#         return current_coords
#     avg_x = int(weight * previous_coords[0] + (1 - weight) * current_coords[0])
#     avg_y = int(weight * previous_coords[1] + (1 - weight) * current_coords[1])
#     return avg_x, avg_y
#
#
# def main():
#     cap = cv2.VideoCapture(0)  # Thay 0 bằng ID camera nếu cần
#     if not cap.isOpened():
#         print("Không thể truy cập camera")
#         return
#
#     # Biến lưu tọa độ trung bình cuối cùng
#     last_red_coords = None
#     last_blue_coords = None
#
#     # Phạm vi màu HSV
#     lower_red = np.array([0, 120, 70])
#     upper_red = np.array([10, 255, 255])
#     lower_red2 = np.array([170, 120, 70])
#     upper_red2 = np.array([180, 255, 255])
#     lower_blue = np.array([100, 150, 0])
#     upper_blue = np.array([140, 255, 255])
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Không đọc được khung hình từ camera")
#             break
#
#         # Phát hiện màu đỏ
#         red_coords1 = detect_color(frame, lower_red, upper_red)
#         red_coords2 = detect_color(frame, lower_red2, upper_red2)
#         red_coords = red_coords1 or red_coords2
#
#         # Phát hiện màu xanh
#         blue_coords = detect_color(frame, lower_blue, upper_blue)
#
#         # Ổn định tọa độ màu đỏ
#         if red_coords is not None:
#             last_red_coords = weighted_average(red_coords, last_red_coords)
#             print(f"Tọa độ X màu đỏ: {last_red_coords[0]}")  # In tọa độ X màu đỏ ra terminal
#
#         # Ổn định tọa độ màu xanh
#         if blue_coords is not None:
#             last_blue_coords = weighted_average(blue_coords, last_blue_coords)
#             print(f"Tọa độ X màu xanh: {last_blue_coords[0]}")  # In tọa độ X màu xanh ra terminal
#
#         # Vẽ dấu cộng tại tọa độ ổn định
#         if last_red_coords is not None:
#             print(f"Tọa độ đỏ: X = {last_red_coords[0]}, Y = {last_red_coords[1]}")
#
#         if last_blue_coords is not None:
#             print(f"Tọa độ xanh: X = {last_blue_coords[0]}, Y = {last_blue_coords[1]}")
#
#         # Hiển thị khung hình
#         cv2.imshow('Color Detection', frame)
#
#         # Nhấn 'q' để thoát
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()
import cv2
import numpy as np

# Hàm ánh xạ tọa độ từ khoảng pixel sang khoảng [-100, 100]
def map_to_range(x, min_input, max_input):
    return int((x - min_input) * (200 / (max_input - min_input)) - 100)

# Phát hiện màu, trả về tọa độ tâm và vẽ khung bao quanh
def detect_color(frame, lower_bound, upper_bound, min_area=500):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area > max_area:  # Chỉ lấy contour lớn nhất
            max_contour = contour
            max_area = area

    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        center_x = x + w // 2
        center_y = y + h // 2

        # Ánh xạ tọa độ X và Y từ pixel sang khoảng [-100, 100]
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        mapped_x = map_to_range(center_x, 0, frame_width)
        mapped_y = map_to_range(center_y, 0, frame_height)

        # Vẽ khung bao quanh đối tượng và tâm
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ hình chữ nhật bao quanh đối tượng
        cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), 2)  # Vẽ điểm tâm của đối tượng
        cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)  # Đường ngang qua tâm
        cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)  # Đường dọc qua tâm

        # Trả về tọa độ đã ánh xạ
        return mapped_x, mapped_y
    return None

# Tính giá trị trung bình có trọng số để làm mượt tọa độ
def weighted_average(current_coords, previous_coords, weight=0.8):
    if previous_coords is None:
        return current_coords
    avg_x = int(weight * previous_coords[0] + (1 - weight) * current_coords[0])
    avg_y = int(weight * previous_coords[1] + (1 - weight) * current_coords[1])
    return avg_x, avg_y

# Hàm tính trung bình tọa độ trong một số frame
def average_over_frames(coords_list):
    if len(coords_list) == 0:
        return None
    avg_x = int(np.mean([coords[0] for coords in coords_list]))
    avg_y = int(np.mean([coords[1] for coords in coords_list]))
    return avg_x, avg_y

def main():
    cap = cv2.VideoCapture(0)  # Thay 0 bằng ID camera nếu cần
    if not cap.isOpened():
        print("Không thể truy cập camera")
        return

    # Biến lưu tọa độ trung bình cuối cùng
    last_red_coords = None
    last_blue_coords = None

    # Duy trì danh sách các tọa độ trong 10 frame
    red_coords_list = []
    blue_coords_list = []

    # Phạm vi màu HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được khung hình từ camera")
            break

        # Phát hiện màu đỏ
        red_coords1 = detect_color(frame, lower_red, upper_red)
        red_coords2 = detect_color(frame, lower_red2, upper_red2)
        red_coords = red_coords1 or red_coords2

        # Phát hiện màu xanh
        blue_coords = detect_color(frame, lower_blue, upper_blue)

        # Lưu các tọa độ vào danh sách, tối đa 10 frame
        if red_coords is not None:
            red_coords_list.append(red_coords)
            if len(red_coords_list) > 10:
                red_coords_list.pop(0)  # Xóa tọa độ của frame đầu tiên nếu danh sách đã đủ 10 frame

        if blue_coords is not None:
            blue_coords_list.append(blue_coords)
            if len(blue_coords_list) > 10:
                blue_coords_list.pop(0)  # Xóa tọa độ của frame đầu tiên nếu danh sách đã đủ 10 frame

        # Tính trung bình tọa độ từ danh sách các frame
        if len(red_coords_list) > 0:
            last_red_coords = average_over_frames(red_coords_list)

        if len(blue_coords_list) > 0:
            last_blue_coords = average_over_frames(blue_coords_list)

        # In tọa độ ổn định
        if last_red_coords is not None:
            print(f"Tọa độ đỏ: X = {last_red_coords[0]}, Y = {last_red_coords[1]}")

        if last_blue_coords is not None:
            print(f"Tọa độ xanh: X = {last_blue_coords[0]}, Y = {last_blue_coords[1]}")

        # Hiển thị khung hình
        cv2.imshow('Color Detection', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

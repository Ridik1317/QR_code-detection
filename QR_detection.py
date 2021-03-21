import cv2
import numpy as np
import matplotlib.pyplot as plt


def view_image(display, image):
    cv2.namedWindow(display, cv2.WINDOW_NORMAL)
    cv2.imshow(display, image)
    cv2.waitKey(0)


img_original = cv2.imread('photo/1.jpg')
img_gray = cv2.cvtColor(img_original.copy(), cv2.COLOR_BGR2GRAY)
(_, thresh) = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)

#ищем основыные 3 квадрата qr кода с помощью детектора контуров
# find contours
markers = []
contours,  _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
if contours:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # find three square markers #можо еще ввести сравнение плоящадей обхватываемого квадрата и самого контура,тем самым если они почти совпадают то контур прямоугольник
    for i in range(0, len(contours)):
        markers = [i]
        (x1, y1, w1, h1) = cv2.boundingRect(contours[i])
        if 1.2 > w1/h1 > 0.8:
            for j in range(1+i, len(contours)):
                (x2, y2, w2, h2) = cv2.boundingRect(contours[j])
                if 1.15 > cv2.contourArea(contours[j])/cv2.contourArea(contours[i]) > \
                        0.85 and 1.15 > h2/h1 > 0.85 and 1.15 > w2/w1 > 0.85:
                    markers.append(j)
            if len(markers) == 3:
                break
#обьединяем найденные 3 контура
# unit three markers in one
a = np.concatenate((contours[markers[0]], contours[markers[1]]), axis=0)
a = np.concatenate((a, contours[markers[2]]), axis=0)
contours[0] = a
img_a_contour = img_original.copy()
cv2.drawContours(img_a_contour, contours, 0, (0, 0, 255), 3)

#охватываем 3 контура прямоугольником,тем самым должны получить в нем наш qr код
# bound contour 'a' with rectangle
rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)
box_points = np.int0(box)
img_end = img_original.copy()
cv2.drawContours(img_end, [box_points], 0, (255, 0, 0), 6)
for i in markers:
    cv2.drawContours(img_end, contours, i, (100, 255, 0), 4)

#вывод изображения
# Display
fig = plt.figure(figsize=(15, 10))

# 1
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 1)
plt.imshow(img_original)
plt.title('img_original')
plt.axis('off')

# 2
thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 2)
plt.imshow(thresh)
plt.title('thresh')
plt.axis('off')

# 3
img_a_contour = cv2.cvtColor(img_a_contour, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 3)
plt.imshow(img_a_contour)
plt.title('img_a_contour')
plt.axis('off')

# 4
img_end = cv2.cvtColor(img_end, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 4)
plt.imshow(img_end)
plt.title('img_end')
plt.axis('off')

plt.show()

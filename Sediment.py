import cv2
import glob
import re
import matplotlib.pyplot as plt

img_paths = glob.glob('img_path')
sorted_img_paths = sorted(img_paths, key=lambda x: int(re.search(r'\((\d+)\).png', x).group(1)))
num_pic = len(sorted_img_paths)
print(num_pic)

Sediment_list = [0]
for img_path in sorted_img_paths[0:]:
    print(img_path)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enhan_image = cv2.imread(img_path)
    enhan_image = cv2.cvtColor(enhan_image, cv2.COLOR_BGR2RGB)
    for row in range(enhan_image.shape[0]):
        for col in range(enhan_image.shape[1]):
            if enhan_image[row, col, 0] == 0:
                enhan_image[row, col] = [255, 255, 255]

    R_valv = 111
    G_valv = 140
    B_valv = 73
    Over = 110
    Sediment = 0

    for row in range(enhan_image.shape[0]):
        for col in range(enhan_image.shape[1]):
            if enhan_image[row, col, 1] <= G_valv and enhan_image[row, col, 0] >= R_valv:
                enhan_image[row, col] = [0, 0, 0]
                Sediment += 1
            if enhan_image[row, col, 0] <= Over and enhan_image[row, col, 1] <= Over:
                enhan_image[row, col] = [0, 0, 0]
                Sediment += 1

    print('Sediment Block:', Sediment)
    Sediment_list.append(Sediment)

    cv2.imshow('Enhanced Image', cv2.cvtColor(enhan_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

plt.plot(Sediment_list)
plt.xticks(range(num_pic+1))
plt.tight_layout()
plt.show()
plt.savefig('Result.jpg')

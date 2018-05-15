from skin_detector import skindectector
import numpy as np
import cv2

img = cv2.imread('./images/original/test5.jpeg')
height, width = img.shape[0], img.shape[1]
nu_img = cv2.resize(img, (width, height))

column_image = np.reshape(nu_img, (int(nu_img.size/3), 3))

skindetector = skindectector.SkinDetector("./data/Skin_NonSkin.txt")
l, mean, cov = skindetector.train()
prob = skindetector.test(l, mean, cov, x_test=column_image)

for i in range(len(prob)):
    if prob[i][0] > 0.5:
        column_image[i] = [255, 255, 255]
    else:
        column_image[i] = [0, 0, 0]

mask = np.reshape(column_image, nu_img.shape)
# mask_height, mask_width = mask.shape[0], mask.shape[1]
# mask_img = cv2.resize(mask, (int(round(mask_height/10, 0)), (int(round(width/10, 0)))))
cv2.imwrite("./images/mask/mask5.jpg", mask)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# counter = 0
# for i in range(len(prob)):
#     if prob[i][0] == 1 and labels[i][0] == 0:
#         counter += 1
#     elif prob[i][0] == 0 and labels[i][0] == 1:
#         counter += 1
#
# accuracy = (1 - (counter/len(prob))) * 100
# print(round(accuracy, 3))

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import sobel

# Import image
raw_image = np.asarray(Image.open('beach_hut.jpg'))

# Convert to greyscale
greyscale = np.zeros(raw_image.shape[:2])
for i, column in enumerate(raw_image):
    for j, pixel in enumerate(column):
        greyscale[i][j] = sum(pixel)/3
        
#apply sobel operator
image = sobel(greyscale)

#display
fig = plt.figure(figsize=(16, 9))

ax = fig.add_subplot(1, 2, 1)
plt.imshow(raw_image)
ax.set_xticks([])
ax.set_yticks([])

image = np.where(abs(image), abs(image), 0)
ax = fig.add_subplot(1, 2, 2)
plt.imshow(image, cmap='Greys_r')
ax.set_xticks([])
ax.set_yticks([])

plt.show()
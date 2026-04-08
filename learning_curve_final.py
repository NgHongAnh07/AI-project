from PIL import Image
import matplotlib.pyplot as plt

image_path = "sessions/learning_curve_final.png"

img = Image.open(image_path)

plt.imshow(img)
plt.axis('off')
plt.title("Learning Curve")
plt.show()
import torch

# list all available models
torch.hub.list('B-cos/B-cos-v2')

# load a pretrained model
model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)

from PIL import Image
import matplotlib.pyplot as plt

# load image
img = model.transform(Image.open('cat.jpg'))
img = img[None].requires_grad_()

# predict and explain
model.eval()
expl_out = model.explain(img)
print("Prediction:", expl_out["prediction"])  # predicted class idx
plt.imshow(expl_out["explanation"])
plt.savefig("explanation.png")
plt.show()

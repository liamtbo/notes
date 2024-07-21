import torch
import torchvision.models as models

# saving and loading model weights
model = models.vgg16(weights='IMAGENET1K_V1') # pretrained model
torch.save(model.state_dict(), 'model_weights.pth')

# To load model weights, you need to create an instance of the same model 
# first, and then load the parameters using load_state_dict() method.
model = models.vgg16() # we do not specify wieghts, i.e. create an untrained model
model.load_state_dict(torch.load('model_weights.pth'))
# be sure to call model.eval() method before inferencing to set the dropout and batch 
# normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.
model.eval()

# saving and loading models with shapes
torch.save(model, 'model.pth')
model = torch.load('model.pth')

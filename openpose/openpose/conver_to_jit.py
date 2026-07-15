import torch
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from action_detect.net import NetV2

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 5, 1)
#         self.conv2 = nn.Conv2d(32, 64, 5, 1)
#         self.fc1 = nn.Linear(4 * 4 * 64, 512)
#         self.fc2 = nn.Linear(512, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 64)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
#
#
# def myPth_to_pt():
#     model = torch.load(r'.\weights\checkpoint_iter_370000.pth')
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     #summary(model, input_size=(1,3, 256, 456))
#     model = model.to(device)
#     traced_script_module = torch.jit.trace(model, torch.ones(1, 1, 28, 28).to(device))
#     traced_script_module.save("testConver.pt")

def openpose_to_jit():

    x = torch.randn(1,3,256,456)

    net = PoseEstimationWithMobileNet().cpu()
    checkpoint = torch.load(r'.\weights\checkpoint_iter_370000.pth', map_location='cpu')
    load_state(net, checkpoint)
    net.eval()
    net(x)
    script_model = torch.jit.trace(net, x)
    script_model.save('weights/openpose.jit')

def test_openpose_jit():
    x = torch.randn(1, 3, 256, 456)
    model = torch.jit.load(r".\action_detect\checkPoint\action.pt")

    print(model(x))

def action_to_jit():
    x = torch.randn(1, 16384)

    net = NetV2().cpu()
    checkpoint = torch.load(r'.\action_detect\checkPoint\myFirstAction.pt', map_location='cpu')
    net.load_state_dict(checkpoint)
    net.eval()
    net(x)
    script_model = torch.jit.trace(net, x)
    script_model.save('myFirstAction.jit')

    print(x.shape)

def test_action_jit():
    x = torch.randn(1, 16384)
    #model = torch.jit.load(r".\action_detect\checkPoint\action.pt")#原来的
    model = torch.jit.load(r".\action_detect\checkPoint\action.pt")
    print(model(x))


if __name__ == '__main__':
    #test_action_jit()
    action_to_jit()
    #test_openpose_jit()
    #openpose_to_jit()
    #myPth_to_pt()
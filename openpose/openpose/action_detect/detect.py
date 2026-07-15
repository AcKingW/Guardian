
import numpy as np
from torch import from_numpy, argmax

DEVICE = "cpu"


def action_detect(net,pose,crown_proportion):
    # img = cv2.cvtColor(pose.img_pose,cv2.IMREAD_GRAYSCALE)
    #print(crown_proportion) #宽高比显示
    img = pose.img_pose.reshape(-1)
    img = img / 255  # 把数据转成[0,1]之间的数据

    img = np.float32(img)

    img = from_numpy(img[None,:]).cpu()

    predect = net(img)
    #print(predect)

    action_id = int(argmax(predect,dim=1).cpu().detach().item())
    #print(action_id)
    #print(predect[:,action_id])
    possible_rate = 0.6 * predect[:, action_id] + 0.4 * (crown_proportion - 1)
    #print(possible_rate)
    possible_rate = possible_rate.detach().numpy()[0]
    #print(possible_rate)
    if possible_rate > 0.55:
        pose.pose_action = 'fallPeople'
        if possible_rate > 1:
            possible_rate = 1
        pose.action_fall = possible_rate
        pose.action_normal = 1-possible_rate
    else:
        pose.pose_action = 'normalPeople'
        if possible_rate >= 0.5:
            pose.action_fall = 1-possible_rate
            pose.action_normal = possible_rate
        else:
            pose.action_fall = possible_rate
            pose.action_normal = 1 - possible_rate

    return pose
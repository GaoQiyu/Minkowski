import torch
import os
import MinkowskiEngine as ME
import model.minkunet as Minkowski


if __name__ == '__main__':

    model_dict = torch.load('/home/gaoqiyu/PycharmProjects/PointCloudSeg_Minkowski/resume/weights.pth')
    model = Minkowski.MinkUNet34C(3, 20)
    model.load_state_dict(model_dict)
    tmp = ME.MinkowskiConvolution(96, 14, kernel_size=1, has_bias=True, dimension=3)
    ME.utils.kaiming_normal_(tmp.kernel, mode='fan_out', nonlinearity='relu')
    model.final = tmp
    torch.save(model.state_dict(), '/home/gaoqiyu/PycharmProjects/PointCloudSeg_Minkowski/resume/weights_14.pth', )


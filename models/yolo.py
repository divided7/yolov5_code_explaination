# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    print("export_mode? :",export)

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes  # 80
        self.no = nc + 5  # number of outputs per anchor  # 85
        self.nl = len(anchors)  # number of detection layers # 3,检测的层数（即模型有多少个输出）
        self.na = len(anchors[0]) // 2  # number of anchors # 3,每个输出的每个patch上的anchor个数
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid # 初始化grid，生成nl=3个空的tensor放入grid这个list中
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid # 初始化anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        print("Detect_ch:", ch)  # ch = [128, 256, 512]
        # 即：self.m = ModuleList(nn.Conv2d(128,255), nn.Conv2d(256,255), nn.Conv2d(512,255))
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            # nl = 3 ,number of layer 输出的个数
            print("x[", i, "].shape =", x[i].shape)
            # 直接运行本yolo.py得到的x的三个尺寸似乎有问题，在export.py中调用本forward函数则是正常的20；40；80尺寸
            x[i] = self.m[i](x[i])  # 对三个输出分别做卷积，只改变通道为255
            # 将输出的格式转换一下，由[batch size=1, C=255, H, W] -> [batch size=1, number of anchor=3, y, x, 85]
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # print("ny,nx:",ny,nx)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # 关于contiguous的参考：https://zhuanlan.zhihu.com/p/64551412
            print("i=", i, "x[i].shape", x[i].shape)
            # print("self.grid:",self.grid)
            # 推理模式下：只进行推理，不更新权重

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # 获取grid和anchor_grid，grid是patch的坐标-0.5，anchor_grid是预设的anchor值；形状都为[bs=1,na=3,w,h,2]
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                if isinstance(self, Segment):  # (boxes + masks)
                    # 语义分割
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    # 目标检测
                    print("检测**")
                    print("len(x) = ",len(x)) # len(x)=3
                    print("x[",i,"].shape = ", x[i].shape) # x[i].shape = 1,3,h,w,85
                    print("x[i].max and min",x[i].max(),x[i].min()) # x张量有正有负
                    print("**")
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    # x[i].shape = 1,3,20,20,85, 先sigmoid后再split拆分为xy wh和conf三个张量
                    # print("xy.shape",xy.shape) # 1,3,20,20,2
                    # print(wh.shape) # 1,3,20,20,2
                    # print(conf.shape) # 1,3,20,20,81
                    # split((2,2,81),4)是按第4个维度（也就是数量为85的那个维度）拆分，拆分形状为2,2,81，即拆成了两个xy，两个wh，81个conf（80分类+是否背景）
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    # x = (x * 2 + "grid" - 0.5) * stride
                    # 注：yolov5的xy回归公式：偏移量*2-0.5+Cx（偏移量是我们要预测的，也就是模型的输出，也就是x[i]里的东西，也就是这里的xy）
                    # Cx - 0.5 看成一个整体，其实就是self.grid，形状都为[bs=1,na=3,w,h,2],最后一个维度里两个元素代表所在patch的坐标-0.5
                    # stride是步长，也就是压缩比例，和初始输入（640）的缩放比，这样xy就还原成原图对应的尺寸了
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    # wh也是预测的偏移量，公式为原本w值*2*（偏移量^2），这里原本w值便是预先提供的回归得到的anchor值的w
                    y = torch.cat((xy, wh, conf), 4)
                    # 又把修正完的xywh和conf给拼回去了,这里的y就是预测的结果了
                z.append(y.view(bs, self.na * nx * ny, self.no))
                # 把y的形状改为1,3*w*h,85,存在z中
        # 如果是训练模式，只返回当前的x就行，因为我们要计算损失去更新权重
        # 如果不是训练模式，是export模式(即运行export.py时会触发),则返回(torch.cat(z, 1),) # 返回torch.cat(z, 1)和None
        # 如果不是训练，也不是export，则为推理模式，返回torch.cat(z, 1) 和 x
        # cat(z,1)是将z这个list里的张量按dim=1进行拼接，注意z里面的内容形状为（1,3*w*h,85）,也就是拼接完后结果形状是（1，3*20*20+3*40*40+3*80*80,85）=(1,25200,85)
        print("self.export = ",self.export) # 当运行export.py时self.export = True 否则为False

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        # 这里的nx和ny是输入特征图的x和y（或者说w和h，就是前面的20*20和40*40和80*80） ???好像有问题
        # print的结果nx和ny是
        # print("self.anchors[i]",self.anchors[i])
        # 注意这里的anchors为原本anchors除过strides的结果，所以要乘上strides

        d = self.anchors[i].device
        t = self.anchors[i].dtype
        print("d,t = ",d,t)
        # print("anchors[i] = ",self.anchors[i])
        # 这里有个疑问 self.anchors是从哪来的
        shape = 1, self.na, ny, nx, 2  # grid shape = [1, 3, 20, 20, 2] / [1, 3, 40, 40, 2] / [1, 3, 80, 80, 2]
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        # print("ny, nx, y, x:", ny, nx, y, x) # ny nx = 80, 40 ,20 不要相信print的结果
        # x，y为：tensor([ 0.,  1.,  2., ... , nx or ny], device='cuda:0')
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        # meshgrid画网格矩阵，即一个80*80的矩阵
        # print("yv,xv:",yv,xv)
        # yv: 80*80，第一行全为0，第二行全为1 。。。
        # xv: 80*80，第一列全为0，第二列全为1。。。
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        # print("grid.shape",grid.shape) # grid.shape = [1,3,20,20,2]
        # print("grid", grid)
        # grid值为： [[[[0,0],[0,1],...,[0,80],[1,0],...,[80,80] ] ]]] - 0.5
        # print("self.stride[i]", self.stride[i])  # stride = 8,16,32
        # print("self.anchors[i]", self.anchors[i])
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        # 注意这里的anchors为原本anchors除过strides的结果，所以要乘上strides
        # print("anchor_grid:",anchor_grid,"\n","anchor_grid.shape",anchor_grid.shape)

        # grid为网格的坐标（patch的坐标），anchor_grid为预设的anchor
        # anchor_grid.shape =  torch.Size([1, 3, 20, 20, 2])
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            # 判断这里的cfg是不是一个字典，如果是字典形式直接赋给yaml
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            # 十有八九是不是了，因为我们用的是yaml文件，所以作者这里读取yaml了
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                # 这里的yaml本质还是读了yolov5s.yaml里的内容并存为字典
                self.yaml = yaml.safe_load(f)  # model dict # save_load是读取yaml的一种函数，相对于load，能避免读到一些危害到程序的内容

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels 模型输入通道，也就是RGB啦，所以是3，灰度图要改成1
        if nc and nc != self.yaml['nc']:
            # 这里的nc是有在train.py里指定的，同时在yaml文件里也指定，总之就是好几个地方都指定了目标检测的类别数，这里做个判断确保nc是一样的
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            # 最终保留的是nc，也就是train.py里的nc，所以说train.py里的nc优先级最高
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            # 同理，如果anchor有多个遇到重复加载，以train.py为主，anchor四舍五入为整型
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # print("yaml:", self.yaml)
        """
        yaml: 
            {
            'nc': 80, 'depth_multiple': 0.33, 'width_multiple': 0.5, 
            'anchors': [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], 
            'backbone': [[-1, 1, 'Conv', [64, 6, 2, 2]], [-1, 1, 'Conv', [128, 3, 2]], [-1, 3, 'C3', [128]], [-1, 1, 'Conv', [256, 3, 2]], [-1, 6, 'C3', [256]], [-1, 1, 'Conv', [512, 3, 2]], [-1, 9, 'C3', [512]], [-1, 1, 'Conv', [1024, 3, 2]], [-1, 3, 'C3', [1024]], [-1, 1, 'SPPF', [1024, 5]]], 
            'head': [[-1, 1, 'Conv', [512, 1, 1]], [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], [[-1, 6], 1, 'Concat', [1]], [-1, 3, 'C3', [512, False]], [-1, 1, 'Conv', [256, 1, 1]], [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], [[-1, 4], 1, 'Concat', [1]], [-1, 3, 'C3', [256, False]], [-1, 1, 'Conv', [256, 3, 2]], [[-1, 14], 1, 'Concat', [1]], [-1, 3, 'C3', [512, False]], [-1, 1, 'Conv', [512, 3, 2]], [[-1, 10], 1, 'Concat', [1]], [-1, 3, 'C3', [1024, False]], [[17, 20, 23], 1, 'Detect', ['nc', 'anchors']]], 
            'ch': 3
            }
        """
        # 到模型了！
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # 得到模型：
        # print("model:", self.model)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # print("inplace = ", self.inplace)
        # Build strides, anchors
        m = self.model[-1]  # Detect() # model里最后一层是Detect：[[17, 20, 23], 1, Detect, [nc, anchors]]
        # print("list(m.children())[-1:] ", list(m.children())[-1:])
        # xxx = torch.randn(1,3,640,640)
        # yyy = m(xxx)
        # print("yyy.shape",yyy.shape)
        """
        m:
        (24): Detect(
            (m): ModuleList(
            (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
        )
        """
        if isinstance(m, (Detect, Segment)):
            # m是model[-1]，也就是模型的最后一层
            # 这里判断模型的最后一层是不是Detect头或者分割头
            s = 256  # 2x min stride
            m.inplace = self.inplace  # m.inplace = True
            # 如果m是Segment，是分割的话，则forward = forward(x)[0]，否则forward(x)
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # print("m.stride = ",m.stride) # m.stride = tensor([ 8., 16., 32.]) # 三个输出的下采样比例
            check_anchor_order(m) #  Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
            # 在此处，anchor是 [10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]
            m.anchors /= m.stride.view(-1, 1, 1)
            # 在此处，anchor是 [10,13, 16,30, 33,23]/8, [30,61, 62,45, 59,119]/16, [116,90, 156,198, 373,326]/32
            # print("m.anchors = ", m.anchors)
            self.stride = m.stride # 把Detect模块的stride送到全局
            self._initialize_biases()  # only run once # https://zhuanlan.zhihu.com/p/63626711 涉及到focal loss初始化偏置的问题

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://zhuanlan.zhihu.com/p/63626711
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


# 核心模型部分，train.py中调用的就是models.yolo.Model，其原代码为：
# model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary 解析yolov5_.yaml字典
    # 记录日志
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 读取 anchors, nc, gd, gw, act ，在yaml文件里是没有act这个参数的，说明我们可以在.yaml文件里修改激活函数
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    # print(act)  # act默认是none
    if act:
        # act里是字符串，用eval转为函数。act替换成ReLU时报错，但是替换逻辑应该是对的，在yaml文件里添加activation: nn.SiLU()
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # print("anchors = ", anchors)  # [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    # print("na = ", na)  # na = 3 = 6//2
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    # print("no = ", no)  # no = 255 = 3 * 85
    # print("ch = ", ch)
    # 创建layers和save空list容器，c2=3
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out #  ch = [3]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # 以backbone第一层为例： [-1, 1, 'Conv', [64, 6, 2, 2]]
        # 此时 i=0, f=-1, n=1, m = Conv, args=[64, 6, 2, 2]
        m = eval(m) if isinstance(m, str) else m  # module是字符串，转化成模型
        # print("args", args)
        for j, a in enumerate(args):
            # 对args里的数据做一个审核，确保是数字而不是str
            # 注意在最后一层：[[17, 20, 23], 1, Detect, [nc, anchors]]
            with contextlib.suppress(NameError):
                # contextlib.suppress是异常抑制功能，如果报错NameError则抑制，不执行这部分内容，不影响程序运行
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings 这行代码会被执行，但如果报错NameError则不执行

        # 配置bottleneck个数
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain, gd=depth_multiple
        # n是C3层中的bottleneck个数，通过gd来动态调整bottleneck的个数
        # yolov5n,s,m,l,x的gd(depth_mutiple)分别为：0.33, 0.33, 0.67, 1.00, 1.33
        # yolo的C3中bottleneck个数：backbone中：3,6,9,3 head中全是3
        # print("n = ", n)  # n = 1
        if m in {
            Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
            BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]  # c2就是c_out c1 = ch[f=-1] ,f是from,c1是c_in
            # ch是一个list,不断添加每层的输出通道数，即可获取下层的输入通道数
            # print("no:", no, "c2:", c2)
            if c2 != no:  # if not output, no = 255 = 3 * 85
                # yolov5n,s,m,l,x的gw(width_mutiple)分别为：0.25, 0.50, 0.75, 1.0, 1.25
                c2 = make_divisible(c2 * gw, 8)  # divisible:divided able 可除的
                # make_divisible(a,8),让a变成8的整数倍，即C2 = C2 *gw，且C2为8的整数倍
                # 总结一下，根据width_multiple，每层输出的通道数按width比例缩放
                # print("c2 = ", c2)
            args = [c1, c2, *args[1:]]
            # print("args:", args)  # args = [c1,c2]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats # args = [c1,c2,n=bottleneck个数]
                n = 1
            # print("args:", args)  # 在3个bottleneck的C3模块中args = [64,64,3]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
            # print("BN_args:", args)
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # print("m_", m_)  # m_是每个模块
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # print("t", t)  # t为models.common.Concat、models.common.Conv、Detect等
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5l.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)
a
    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()

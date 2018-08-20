import torch
import torch.nn as nn
import torchvision
from . import resnet, resnext
from lib.nn import SynchronizedBatchNorm2d


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    @staticmethod
    def pixel_acc(pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    @staticmethod
    def part_loss(pred, part_label, object_label):


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, loss_scale=[None, None, None, None]):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit_list = nn.ModuleList()
        self.loss_scale = loss_scale

        # scene crit
        self.crit_list.append(nn.NLLLoss(ignore_index=-1))
        # object crit
        self.crit_list.append(nn.NLLLoss(ignore_index=-1))
        # part crit


    def forward(self, feed_dict, output_switch, *, segSize=None):
        if segSize is None: # training
            pred_list = self.decoder(
                self.encoder(feed_dict['img_data'], return_feature_maps=True),
                output_switch
            )


            if pred_list[i] is not None:
                loss += sub_loss
            loss = 0
            loss += part_loss

            loss = self.crit(pred, feed_dict['seg_label'])


            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        else: # inference
            pred_list = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            return pred_list


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class ModelBuilder:
    def __init__(self):
        pass

    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='resnet50_dilated8', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        if arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34_dilated8':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    def build_decoder(self, arch='ppm_bilinear_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        if arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


# upernet
class UPerNet(nn.Module):
    def __init__(self, nr_classes=(0, 0, 0, 0), fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256,512,1024,2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_fusion = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)

        self.nr_scene_class, self.nr_object_class, self.nr_part_class, self.nr_material_class = nr_classes

        # input: PPM out, input_dim: fpn_dim
        self.scene_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fpn_dim, self.nr_scene_class, kernel_size=1, bias=True)
        )

        # input: Fusion out, input_dim: fpn_dim
        self.object_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 3),
            nn.Conv2d(fpn_dim, self.nr_object_class, kernel_size=1, bias=True)
        )

        # input: Fusion out, input_dim: fpn_dim
        self.part_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 3),
            nn.Conv2d(fpn_dim, self.nr_part_class, kernel_size=1, bias=True)
        )

        # input: FPN_2 (P2), input_dim: fpn_dim
        self.material_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 3),
            nn.Conv2d(fpn_dim, self.nr_material_class, kernel_size=1, bias=True)
        )

    def forward(self, conv_out, output_switch=(True, True, True, True), segSize=None):
        out_scene, out_object, out_part, out_material = output_switch # output according task if True
        output_list = [None, None, None, None]

        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        if out_scene: # scene
            output_list[0] = self.scene_head(f)

        if out_object or out_part or out_material:
            fpn_feature_list = [f]
            for i in reversed(range(len(conv_out) - 1)):
                conv_x = conv_out[i]
                conv_x = self.fpn_in[i](conv_x) # lateral branch

                f = nn.functional.upsample(
                    f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
                f = conv_x + f

                fpn_feature_list.append(self.fpn_out[i](f))
            fpn_feature_list.reverse() # [P2 - P5]

            # material
            if out_material:
                output_list[-1] = self.material_head(fpn_feature_list[0])

            if out_object or out_part:
                output_size = fpn_feature_list[0].size()[2:]
                fusion_list = [fpn_feature_list[0]]
                for i in range(1, len(fpn_feature_list)):
                    fusion_list.append(nn.functional.upsample(
                        fpn_feature_list[i],
                        output_size,
                        mode='bilinear', align_corners=False))
                fusion_out = torch.cat(fusion_list, 1)
                x = self.conv_fusion(fusion_out)

                if out_object: # object
                    output_list[1] = self.object_head(x)
                if out_part:
                    output_list[2] = self.part_head(x)

        if self.use_softmax:  # is True during inference
            # inference scene prob
            x = output_list[0]
            x = x.squeeze([2, 3])
            x = nn.functional.softmax(x, dim=1)
            output_list[0] = x

            for i in range(1, 4): # inference object, part, material
                x = output_list[i]
                x = nn.functional.upsample(
                    x, size=segSize, mode='bilinear', align_corners=False)
                x = nn.functional.softmax(x, dim=1)
                output_list[i] = x
        else:   # Training
            for i in range(4):
                if output_list[i] is None:
                    continue
                x = output_list[i]
                x = nn.functional.log_softmax(x, dim=1)
                if i == 0: # for scene
                    x = x.squeeze([2, 3])
                output_list[i] = x

        return output_list

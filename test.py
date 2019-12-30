#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test2.py
# @Author: Jehovah
# @Date  : 18-8-8
# @Desc  :


from torchvision.utils import save_image
import option
from data_loader3 import *
from models.sys_trans_sp_multi import *
from pix2pix_model import *

opt = option.init()
# opt.datalist = 'files/list_test.txt'
net_G = Sys_Generator(opt.input_nc, opt.output_nc)

net_G.load_state_dict(torch.load(opt.pre_netG))

dataset = MyDataset(opt, isTrain=1)
data_iter = data.DataLoader(
    dataset, batch_size=opt.batchSize,
    num_workers=16)

if not os.path.exists(opt.output):
    os.mkdir(opt.output)

# net_G.eval()
net_G.cuda()


for i, image in enumerate(data_iter):
    imgA = image[0]
    # imgB = image[1]
    # imgB = image['A']

    real_A = imgA.cuda()
    # real_B = imgB.cuda()

    fake_B = net_G(real_A)
    # output = output.cpu()
    output_name = '{:s}/{:s}{:s}'.format(
        opt.output, str(i+1), '.jpg')
    save_image(fake_B[:,:,3:253,28:228], output_name, normalize=True, scale_each=True)

    print output_name + "  succeed"


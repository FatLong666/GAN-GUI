import shutil
import threading

from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
import argparse
import sys
import os
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from models import Generator
from datasets import ImageDataset


class UI:
    def __init__(self):
        self.ui = uic.loadUi('./GuiDesign/ui.ui')
        self.fake_APath = None
        self.fake_BPath = None
        self.real_APath = None
        self.real_BPath = None
        self.stop_event = threading.Event()     # 创建一个事件管理标志

        self.ui.pushButton.clicked.connect(self.generate)
        self.ui.pushButton_2.clicked.connect(self.changeSignal)
        self.ui.pushButton_3.clicked.connect(self.saveFakeAImg)
        self.ui.pushButton_4.clicked.connect(self.saveFakeBImg)

    def generate(self):
        def receive():
            parser = argparse.ArgumentParser()
            parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
            parser.add_argument('--dataroot', type=str, default='chip2defect/', help='root directory of the dataset')
            parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
            parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
            parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
            parser.add_argument('--cuda', default='0', action='store_true', help='use GPU computation')
            parser.add_argument('--n_cpu', type=int, default=8,
                                help='number of cpu threads to use during batch generation')
            parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth',
                                help='A2B generator checkpoint file')
            parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth',
                                help='B2A generator checkpoint file')
            opt = parser.parse_args()
            print(opt)

            if torch.cuda.is_available() and not opt.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")

            ###### Definition of variables ######
            # Networks
            netG_A2B = Generator(opt.input_nc, opt.output_nc)
            netG_B2A = Generator(opt.output_nc, opt.input_nc)

            if opt.cuda:
                netG_A2B.cuda()
                netG_B2A.cuda()

            # Load state dicts
            netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
            netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

            # Set model's test mode
            netG_A2B.eval()
            netG_B2A.eval()

            # Inputs & targets memory allocation
            Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
            input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
            input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

            # Dataset loader
            transforms_ = [transforms.Resize((256, 256)),
                           transforms.ToTensor(),
                           ]
            dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='train'),
                                    batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
            ###################################

            ###### Testing######

            # Create output dirs if they don't exist
            if not os.path.exists('output/A'):
                os.makedirs('output/A')
            if not os.path.exists('output/B'):
                os.makedirs('output/B')

            for i, batch in enumerate(dataloader):

                self.stop_event.clear()

                real_A = Variable(input_A.copy_(batch['A']))
                real_B = Variable(input_B.copy_(batch['B']))

                    # Generate output
                fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
                fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

                # Save image files

                save_image(fake_A, 'output/A/%04d.png' % (i + 1))
                save_image(fake_B, 'output/B/%04d.png' % (i + 1))
                save_image(real_A, 'input/A/%04d.png' % (i + 1))
                save_image(real_B, 'input/B/%04d.png' % (i + 1))

                self.fake_APath = 'output/A/%04d.png' % (i + 1)
                self.fake_BPath = 'output/B/%04d.png' % (i + 1)
                self.real_APath = 'input/A/%04d.png' % (i + 1)
                self.real_BPath = 'input/B/%04d.png' % (i + 1)

                # 显示在qlabel上

                self.ui.label.setPixmap(QPixmap(self.real_APath))
                self.ui.label_2.setPixmap(QPixmap(self.real_BPath))
                self.ui.label_3.setPixmap(QPixmap(self.fake_BPath))
                self.ui.label_4.setPixmap(QPixmap(self.fake_APath))

                sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

                # self.ui.progressBar.setValue(round(i/len(dataloader), 2)*100)   # 控制进度条

                sys.stdout.write('\n')

                if self.stop_event.isSet():
                    continue
                else:
                    self.stop_event.wait()



        t = threading.Thread(target=receive)
        t.start()

    def changeSignal(self):

        self.stop_event.set()

    def saveFakeAImg(self):
        def receive():
            src_path = self.fake_APath
            img_name= src_path.split('/')[2]
            dst_path = './GuiSaveImg/fake_A/'+img_name
            shutil.copyfile(src_path, dst_path)
            print('Successful save image ' + self.fake_APath)

        t = threading.Thread(target=receive)
        t.start()

    def saveFakeBImg(self):
        def receive():
            src_path = self.fake_BPath
            img_name= src_path.split('/')[2]
            dst_path = './GuiSaveImg/fake_B/'+img_name
            shutil.copyfile(src_path, dst_path)
            print('Successful save image ' + self.fake_BPath)

        t = threading.Thread(target=receive)
        t.start()


if __name__ == '__main__':
    app = QApplication([])
    stats = UI()
    stats.ui.show()
    app.exec_()

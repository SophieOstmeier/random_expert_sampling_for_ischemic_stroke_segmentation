#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.dice_loss import DC_focal_loss
import torch

from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn


class nnUNetTrainerV2_Dice_Focal(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        print("Setting up self.loss = DC_focal_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False},"
              "{'apply_nonlin': softmax_helper, 'alpha':0.5, 'gamma':2, 'smooth':1e-5}")
        self.loss = DC_focal_loss({'batch_dice': self.batch_dice, 'smooth': 1000, 'do_bg': False},
                                      {'apply_nonlin': nn.Softmax(dim=1), 'alpha':0.5, 'gamma':2,'smooth':1e-5})

        self.max_num_epochs = 280 # changed from 1000
        self.initial_lr = 1e-2 # changed from 3e-5
        self.weight_decay = 3e-5 # changed from 3e-5
        self.gpu_id_to_use = 1
        self.threshold = 0
        self.dropout = 0.05  # changed from 0
        self.momentum = 0.9 # changed from 0.99
    
    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something h
        ere

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': self.dropout, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            print(f'current device for network and optimizer: {torch.cuda.current_device()}')
            self.network.cuda()
            print(f'current device for network and optimizer after sending: {self.network.get_device()}')
        self.network.inference_apply_nonlin = softmax_helper
    
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=self.momentum, nesterov=True) # changed from momentum=0.99, tried 0.80
        self.lr_scheduler = None
import math
import os
from functools import partial

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
from chainer import Sequential, Variable
from chainer.links.caffe import CaffeFunction
from chainerrl.agents import a3c


class RNet_trained(chainer.Chain, a3c.A3CModel):
    def __init__(self, input_chn=2, n_actions=2, C=16):
        dil = [1, 2, 3, 4, 5]  # dilate rates
        init_w = chainer.initializers.HeNormal()

        super(RNet_trained, self).__init__(
            block1=Sequential(
                L.Convolution3D(input_chn, C, ksize=3, stride=1, pad=dil[0], dilate=dil[0], nobias=True,
                                initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[0], dil[0], 0), dilate=dil[0], nobias=True,
                                initialW=init_w),
                F.relu),
            comp1=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block2=Sequential(
                L.Convolution3D(C, C, ksize=3, stride=1, pad=dil[1], dilate=dil[1], nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[1], dil[1], 0), dilate=dil[1], nobias=True,
                                initialW=init_w),
                F.relu),
            comp2=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block3=Sequential(
                L.Convolution3D(C, C, ksize=3, stride=1, pad=dil[2], dilate=dil[2], nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[2], dil[2], 0), dilate=dil[2], nobias=True,
                                initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[2], dil[2], 0), dilate=dil[2], nobias=True,
                                initialW=init_w),
                F.relu),
            comp3=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block4_pi=Sequential(
                L.Convolution3D(C, C, ksize=3, stride=1, pad=dil[3], dilate=dil[3], nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[3], dil[3], 0), dilate=dil[3], nobias=True,
                                initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[3], dil[3], 0), dilate=dil[3], nobias=True,
                                initialW=init_w),
                F.relu),
            comp4_pi=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block4_v=Sequential(
                L.Convolution3D(C, C, ksize=3, stride=1, pad=dil[3], dilate=dil[3], nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[3], dil[3], 0), dilate=dil[3], nobias=True,
                                initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[3], dil[3], 0), dilate=dil[3], nobias=True,
                                initialW=init_w),
                F.relu),
            comp4_v=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block5_pi=Sequential(
                L.Convolution3D(C, C, ksize=3, stride=1, pad=dil[4], dilate=dil[4], nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[4], dil[4], 0), dilate=dil[4], nobias=True,
                                initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[4], dil[4], 0), dilate=dil[4], nobias=True,
                                initialW=init_w),
                F.relu),
            comp5_pi=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block5_v=Sequential(
                L.Convolution3D(C, C, ksize=3, stride=1, pad=dil[4], dilate=dil[4], nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[4], dil[4], 0), dilate=dil[4], nobias=True,
                                initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[4], dil[4], 0), dilate=dil[4], nobias=True,
                                initialW=init_w),
                F.relu),
            comp5_v=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block6_pi=Sequential(
                L.Convolution3D(5 * C // 4, C, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),
                F.relu,
                chainerrl.policies.SoftmaxPolicy(
                    L.Convolution3D(C, n_actions, ksize=3, stride=1, pad=1, dilate=1, nobias=True, initialW=init_w))),

            block6_v=Sequential(
                L.Convolution3D(5 * C // 4, C, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, 1, ksize=3, stride=1, pad=1, dilate=1, nobias=True, initialW=init_w),
                F.relu),
        )

        self.train = True


# class DilatedConv3dBlock(chainer.Chain):

# 	def __init__(self, in_chn, out_chn, ksize, stride, pad, initial_w, dilate):
# 		super(DilatedConv3dBlock, self).__init__(
# 			diconv = L.Convolution3D(in_channels=in_chn, out_channels=out_chn, ksize=ksize, stride=stride, pad=pad, \
# 				nobias=True, initialW=initial_w, dilate=dilate)
# 		)

# 		self.train = True

# 	def __call__(self, x):
# 		h = F.relu(self.diconv(x))
# 		return h

class RNet(chainer.Chain, a3c.A3CModel):
    def __init__(self, input_chn=4, n_actions=6, C=16, pretrained=True):
        dil = [1, 2, 3, 4, 5]  # dilate rates
        init_w = chainer.initializers.HeNormal()

        # net = RNet_trained(input_chn=2, n_actions=n_actions)
        # rnet_trained_model_path = os.path.join(result_path, init_model_name,'%s_%d.pth' % (init_model_name, init_model_epoch))
        # rnet_trained_model_path = os.path.join(result_path, init_model_name,
        #                                        '%s_%d' % (init_model_name, init_model_epoch), 'model.npz')
        # chainer.serializers.load_npz(rnet_trained_model_path, net)

        # d1 = list(net.block1.children())[0].W.data
        # s = d1.shape
        # d2 = np.random.randn(*s) / np.sqrt(2.)
        # d2 = d1[:, 1:2, ...]
        # d = np.concatenate((d1, d2, d2), axis=1)
        super(RNet, self).__init__(
            block11=Sequential(
                L.Convolution3D(input_chn, C, ksize=3, stride=1, pad=dil[0], dilate=dil[0], nobias=True,
                                initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[0], dil[0], 0), dilate=dil[0],
                                nobias=True, initialW=init_w),
                F.relu),
            comp1=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block2=Sequential(
                L.Convolution3D(C, C, ksize=3, stride=1, pad=dil[1], dilate=dil[1], nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[1], dil[1], 0), dilate=dil[1],
                                nobias=True, initialW=init_w),
                F.relu),
            comp2=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block3=Sequential(
                L.Convolution3D(C, C, ksize=3, stride=1, pad=dil[2], dilate=dil[2], nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[2], dil[2], 0), dilate=dil[2],
                                nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[2], dil[2], 0), dilate=dil[2],
                                nobias=True, initialW=init_w),
                F.relu),
            comp3=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block4_pi=Sequential(
                L.Convolution3D(C, C, ksize=3, stride=1, pad=dil[3], dilate=dil[3], nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[3], dil[3], 0), dilate=dil[3],
                                nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[3], dil[3], 0), dilate=dil[3],
                                nobias=True, initialW=init_w),
                F.relu),
            comp4_pi=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block4_v=Sequential(
                L.Convolution3D(C, C, ksize=3, stride=1, pad=dil[3], dilate=dil[3], nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[3], dil[3], 0), dilate=dil[3],
                                nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[3], dil[3], 0), dilate=dil[3],
                                nobias=True, initialW=init_w),
                F.relu),
            comp4_v=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block5_pi=Sequential(
                L.Convolution3D(C, C, ksize=3, stride=1, pad=dil[4], dilate=dil[4], nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[4], dil[4], 0), dilate=dil[4],
                                nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[4], dil[4], 0), dilate=dil[4],
                                nobias=True, initialW=init_w),
                F.relu),
            comp5_pi=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block5_v=Sequential(
                L.Convolution3D(C, C, ksize=3, stride=1, pad=dil[4], dilate=dil[4], nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[4], dil[4], 0), dilate=dil[4],
                                nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, C, ksize=(3, 3, 1), stride=1, pad=(dil[4], dil[4], 0), dilate=dil[4],
                                nobias=True, initialW=init_w),
                F.relu),
            comp5_v=L.Convolution3D(C, C // 4, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),

            block6_pi=Sequential(
                L.Convolution3D(5 * C // 4, C, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),
                F.relu,
                chainerrl.policies.SoftmaxPolicy(
                    L.Convolution3D(C, n_actions, ksize=3, stride=1, pad=1, dilate=1, nobias=True,
                                    initialW=init_w))),

            block6_v=Sequential(
                L.Convolution3D(5 * C // 4, C, ksize=1, stride=1, pad=0, dilate=1, nobias=True, initialW=init_w),
                F.relu,
                L.Convolution3D(C, 1, ksize=3, stride=1, pad=1, dilate=1, nobias=True, initialW=init_w),
                F.relu),
        )


        self.train = True

    def pi_and_v(self, x):

        # batch x c x a x b x h

        x1 = self.block11(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        x4_pi = self.block4_pi(x3)
        x5_pi = self.block5_pi(x4_pi)

        x4_v = self.block4_v(x3)
        x5_v = self.block5_v(x4_v)

        x1 = F.relu(self.comp1(x1))
        x2 = F.relu(self.comp2(x2))
        x3 = F.relu(self.comp3(x3))

        x4_pi = F.relu(self.comp4_pi(x4_pi))
        x5_pi = F.relu(self.comp5_pi(x5_pi))
        comp_concat_pi = F.concat((x1, x2, x3, x4_pi, x5_pi), axis=1)
        pout = self.block6_pi(comp_concat_pi)

        x4_v = F.relu(self.comp4_v(x4_v))
        x5_v = F.relu(self.comp5_v(x5_v))
        comp_concat_v = F.concat((x1, x2, x3, x4_v, x5_v), axis=1)
        vout = self.block6_v(comp_concat_v)

        return pout, vout

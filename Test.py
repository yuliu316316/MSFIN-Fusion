from PIL import Image
import numpy as np
import os
import torch
import cv2
import time
import imageio
import pydensecrf.densecrf as dcrf
import torchvision.transforms as transforms
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from Networks.MSFIN import MODEL as net

# print(torch.cuda.current_device())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# ids = torch.cuda.device_count()
device = torch.device('cuda:0')       # CUDA:0


model = net(inplanes=6)
model_path = "models/MSFIN_WBCE_1e-4_16/model_100.pth"  # MSNet_cascade_1e-3_8_80_BCE
use_gpu = torch.cuda.is_available()
# use_gpu = False

if use_gpu:
    print('GPU Mode Acitavted')
    model = model.cuda()
    model.cuda()
    # device_ids = range(torch.cuda.device_count())
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load(model_path))
    # print(model)
else:
    print('CPU Mode Acitavted')
    state_dict = torch.load(model_path, map_location='cpu')
    # load params
    model.load_state_dict(state_dict)


# normalize the predicted probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def fusion_color(file_path, type, save_path, couples):
    # num = 20
    for num in range(1, couples+1):
        tic = time.time()
        path1 = file_path + '/c_{}{}_1.'.format(num//10, num % 10) + type           # for the "Lytro" dataset
        path2 = file_path + '/c_{}{}_2.'.format(num//10, num % 10) + type           # for the "Lytro" dataset
        # path1 = file_path + '/mffw_{}{}_A.'.format(num//10, num % 10) + type      # for the "MFFW"dataset
        # path2 = file_path + '/mffw_{}{}_B.'.format(num//10, num % 10) + type      # for the "MFFW" dataset
        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')
        img1_read = np.array(img1)
        img2_read = np.array(img2)      # R G B
        h = img1_read.shape[0]
        w = img1_read.shape[1]
        img1_org = img1
        img2_org = img2
        tran = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))])
        img1_org = tran(img1_org)
        img2_org = tran(img2_org)
        input_img = torch.cat((img1_org, img2_org), 0).unsqueeze(0)
        if use_gpu:
            input_img = input_img.cuda()
        else:
            input_img = input_img
        model.eval()
        out32, out16, out8, out4 = model(input_img)
        out32 = normPRED(out32)
        out16 = normPRED(out16)
        out8 = normPRED(out8)
        out4 = normPRED(out4)
        d_map_1_4 = np.squeeze(out4.detach().cpu().numpy())
        decision_1_4 = (d_map_1_4 * 255).astype(np.uint8)
        # imageio.imwrite(save_path+'/d4_{}{}_1.tif'.format(num//10, num % 10), decision_1_4)

        d_map_1_8 = np.squeeze(out8.detach().cpu().numpy())
        decision_1_8 = (d_map_1_8 * 255).astype(np.uint8)
        # imageio.imwrite(save_path+'/d8_{}{}_1.tif'.format(num//10, num % 10), decision_1_8)

        d_map_1_16 = np.squeeze(out16.detach().cpu().numpy())
        decision_1_16 = (d_map_1_16 * 255).astype(np.uint8)
        # imageio.imwrite(save_path+'/d16_{}{}_1.tif'.format(num//10, num % 10), decision_1_16)

        d_map_1_32 = np.squeeze(out32.detach().cpu().numpy())
        decision_1_32 = (d_map_1_32 * 255).astype(np.uint8)
        # imageio.imwrite(save_path+'/d32_{}{}_1.tif'.format(num//10, num % 10), decision_1_32)

        d_map_sig = np.expand_dims(d_map_1_32, axis=2)
        d_map_sig = np.concatenate((d_map_sig, d_map_sig, d_map_sig), axis=-1)
        fused_image_1 = img1_read * d_map_sig + img2_read * (1 - d_map_sig)
        fused_image_1 = Image.fromarray(np.uint8(fused_image_1))          # img: float32->uint8
        # fused_image_1.save(save_path+'/c_{}{}_sig.tif'.format(num//10, num % 10))

        # --------------binary-------------- #
        d_map_1_32[d_map_1_32 > 0.5] = 1
        d_map_1_32[d_map_1_32 <= 0.5] = 0
        decision_2 = (d_map_1_32 * 255).astype(np.uint8)
        # imageio.imwrite(save_path+'/d_{}{}_2.tif'.format(num//10, num % 10), decision_2)
        d_map_binary = np.expand_dims(d_map_1_32, axis=2)
        d_map_binary = np.concatenate((d_map_binary, d_map_binary, d_map_binary), axis=-1)
        fused_image_2 = img1_read * d_map_binary + img2_read * (1 - d_map_binary)
        fused_image_2 = Image.fromarray(np.uint8(fused_image_2))          # img: float32->uint8
        # fused_image_2.save(save_path+'/c_{}{}_binary.tif'.format(num//10, num % 10))

        # ----------------CRF----------------#
        img = np.array(fused_image_2)
        d_map_rgb = d_map_binary.astype(np.uint32)
        d_map_lbl = d_map_rgb[:, :, 0] + (d_map_rgb[:, :, 1] << 8) + (d_map_rgb[:, :, 2] << 16)
        colors, labels = np.unique(d_map_lbl, return_inverse=True)
        HAS_UNK = False
        colorize = np.empty((len(colors), 3), np.uint8)
        colorize[:, 0] = (colors & 0x0000FF)
        colorize[:, 1] = (colors & 0x00FF00) >> 8
        colorize[:, 2] = (colors & 0xFF0000) >> 16
        n_labels = len(set(labels.flat)) - int(HAS_UNK)
        use_2d = True
        if use_2d:
            d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
            # get unary potentials (neg log probability)
            U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
            d.setUnaryEnergy(U)
            d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL,
                                  normalization=dcrf.NORMALIZE_SYMMETRIC)
            d.addPairwiseBilateral(sxy=7, srgb=7, rgbim=img,
                                   compat=1,
                                   kernel=dcrf.DIAG_KERNEL,
                                   normalization=dcrf.NORMALIZE_SYMMETRIC)
        else:
            d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)
            U = unary_from_labels(labels, n_labels, gt_prob=0.7,
                                  zero_unsure=HAS_UNK)
            d.setUnaryEnergy(U)
            feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
            d.addPairwiseEnergy(feats, compat=3,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)
            # feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
            #                                   img=img1_read, chdim=2)
            # d.addPairwiseEnergy(feats, compat=10,
            #                     kernel=dcrf.DIAG_KERNEL,
            #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(15)
        MAP = np.argmax(Q, axis=0)
        MAP = colorize[MAP, :]
        MAP = MAP.reshape(img1_read.shape)
        decision_4 = (MAP[:, :, 0] * 255).astype(np.uint8)
        # imageio.imwrite(save_path+'/d_{}{}.tif'.format(num // 10, num % 10), decision_4)
        fused_image_4 = img1_read * MAP + img2_read * (1 - MAP)
        fused_image_4 = Image.fromarray(np.uint8(fused_image_4))
        fused_image_4.save(save_path+'/c_{}{}.tif'.format(num//10, num % 10))

        # -----------------------------------------------------------------------------------------------
        # decision = post_process(out, h, w)
        # min2 = torch.max(img_fused)
        # print(min2)
        # plt.imshow(FUSE, cmap=cmap)
        # plt.axis('off')
        # plt.show()
        toc = time.time()
        print('end Lytro {}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))


def fusion_gray(file_path, type, save_path, couples):
    # num = 1
    for num in range(1, couples+1):
        tic = time.time()
        path1 = file_path + '/g_{}{}_1.'.format(num//10, num % 10) + type
        path2 = file_path + '/g_{}{}_2.'.format(num//10, num % 10) + type
        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')
        img1_read = np.array(img1)
        img2_read = np.array(img2)      # R G B
        h = img1_read.shape[0]
        w = img1_read.shape[1]
        img1_org = img1
        img2_org = img2
        tran = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))])
        img1_org = tran(img1_org)
        img2_org = tran(img2_org)
        input_img = torch.cat((img1_org, img2_org), 0).unsqueeze(0)
        if use_gpu:
            input_img = input_img.cuda()
        else:
            input_img = input_img
        model.eval()
        out32, out16, out8, out4 = model(input_img)
        out32 = normPRED(out32)
        out16 = normPRED(out16)
        out8 = normPRED(out8)
        out4 = normPRED(out4)
        d_map_1_4 = np.squeeze(out4.detach().cpu().numpy())
        decision_1_4 = (d_map_1_4 * 255).astype(np.uint8)
        # imageio.imwrite(save_path+'/d4_{}{}_1.tif'.format(num//10, num % 10), decision_1_4)
        #
        d_map_1_8 = np.squeeze(out8.detach().cpu().numpy())
        decision_1_8 = (d_map_1_8 * 255).astype(np.uint8)
        # imageio.imwrite(save_path+'/d8_{}{}_1.tif'.format(num//10, num % 10), decision_1_8)
        #
        d_map_1_16 = np.squeeze(out16.detach().cpu().numpy())
        decision_1_16 = (d_map_1_16 * 255).astype(np.uint8)
        # imageio.imwrite(save_path+'/d16_{}{}_1.tif'.format(num//10, num % 10), decision_1_16)

        d_map_1_32 = np.squeeze(out32.detach().cpu().numpy())
        decision_1_32 = (d_map_1_32 * 255).astype(np.uint8)
        # imageio.imwrite(save_path+'/d32_{}{}_1.tif'.format(num//10, num % 10), decision_1_32)

        d_map_sig = np.expand_dims(d_map_1_32, axis=2)
        d_map_sig = np.concatenate((d_map_sig, d_map_sig, d_map_sig), axis=-1)
        fused_image_1 = img1_read * d_map_sig + img2_read * (1 - d_map_sig)
        fused_image_1 = Image.fromarray(np.uint8(fused_image_1)).convert('L')    # img: float32->uint8
        # fused_image_1.save(save_path+'/g_{}{}_sig.tif'.format(num//10, num % 10))

        # --------------binary-------------- #
        d_map_1_32[d_map_1_32 > 0.5] = 1
        d_map_1_32[d_map_1_32 <= 0.5] = 0
        decision_2 = (d_map_1_32 * 255).astype(np.uint8)
        # imageio.imwrite(save_path+'/d_{}{}_2.tif'.format(num//10, num % 10), decision_2)
        d_map_binary = np.expand_dims(d_map_1_32, axis=2)
        d_map_binary = np.concatenate((d_map_binary, d_map_binary, d_map_binary), axis=-1)
        fused_image_2 = img1_read * d_map_binary + img2_read * (1 - d_map_binary)
        fused_image_2 = Image.fromarray(np.uint8(fused_image_2)).convert('L')    # img: float32->uint8
        # fused_image_2.save(save_path+'/g_{}{}_binary.tif'.format(num//10, num % 10))

        # ----------------CRF----------------#
        img = np.array(fused_image_2.convert('RGB'))
        d_map_rgb = d_map_binary.astype(np.uint32)
        d_map_lbl = d_map_rgb[:, :, 0] + (d_map_rgb[:, :, 1] << 8) + (d_map_rgb[:, :, 2] << 16)
        colors, labels = np.unique(d_map_lbl, return_inverse=True)
        HAS_UNK = False
        colorize = np.empty((len(colors), 3), np.uint8)
        colorize[:, 0] = (colors & 0x0000FF)
        colorize[:, 1] = (colors & 0x00FF00) >> 8
        colorize[:, 2] = (colors & 0xFF0000) >> 16
        n_labels = len(set(labels.flat)) - int(HAS_UNK)
        use_2d = True
        if use_2d:
            d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
            # get unary potentials (neg log probability)
            U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
            d.setUnaryEnergy(U)
            d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL,
                                  normalization=dcrf.NORMALIZE_SYMMETRIC)

            d.addPairwiseBilateral(sxy=7, srgb=7, rgbim=img,
                                   compat=1,
                                   kernel=dcrf.DIAG_KERNEL,
                                   normalization=dcrf.NORMALIZE_SYMMETRIC)
        else:
            d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)
            U = unary_from_labels(labels, n_labels, gt_prob=0.7,
                                  zero_unsure=HAS_UNK)
            d.setUnaryEnergy(U)
            feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
            d.addPairwiseEnergy(feats, compat=3,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)
            # feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
            #                                   img=img1_read, chdim=2)
            # d.addPairwiseEnergy(feats, compat=10,
            #                     kernel=dcrf.DIAG_KERNEL,
            #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(15)
        MAP = np.argmax(Q, axis=0)
        MAP = colorize[MAP, :]
        MAP = MAP.reshape(img1_read.shape)
        decision_4 = (MAP[:, :, 0] * 255).astype(np.uint8)
        # imageio.imwrite(save_path+'/d_{}{}.tif'.format(num // 10, num % 10), decision_4)
        fused_image_4 = img1_read * MAP + img2_read * (1 - MAP)
        fused_image_4 = Image.fromarray(np.uint8(fused_image_4)).convert('L')
        fused_image_4.save(save_path+'/g_{}{}.tif'.format(num // 10, num % 10))

        # -----------------------------------------------------------------------------------------------
        # decision = post_process(out, h, w)
        # min2 = torch.max(img_fused)
        # print(min2)
        # plt.imshow(FUSE, cmap=cmap)
        # plt.axis('off')
        # plt.show()
        toc = time.time()
        print('end gray {}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))


if __name__ == '__main__':
    fusion_color('sourceimages/Lytro', 'tif', 'results_Lytro', 20)     # fuse the "Lytro" dataset

    # fusion_color('sourceimages/Natural', 'tif', 'results_Natural', 5)  # fuse the color images from the "Natural" dataset
    # fusion_gray('sourceimages/Natural', 'tif', 'results_Natural', 5)  # fuse the gray images from the "Natural" dataset

    # fusion_color('sourceimages/MFFW', 'jpg', 'results_MFFW', 13)     # fuse the "MFFW" dataset (the pathes in the fusion_color function should be adjusted)



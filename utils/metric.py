# -*- coding: utf-8 -*-
# @Time    : 2020/7/7
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : metric.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, convolve, distance_transform_edt as bwdist


class CalFM(object):
    # Fmeasure(maxFm, meanFm)---Frequency-tuned salient region detection(CVPR 2009)
    def __init__(self, num, thds=255):
        self.precision = np.zeros((num, thds))
        self.recall = np.zeros((num, thds))
        self.meanF = np.zeros(num)
        self.idx = 0
        self.num = num

    def update(self, pred, gt):
        if gt.max() != 0:
            prediction, recall, mfmeasure = self.cal(pred, gt)
            self.precision[self.idx, :] = prediction
            self.recall[self.idx, :] = recall
            self.meanF[self.idx] = mfmeasure
        self.idx += 1

    def cal(self, pred, gt):
        ########################meanF##############################
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        binary = np.zeros_like(pred)
        binary[pred >= th] = 1
        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 0.5] = 1
        tp = (binary * hard_gt).sum()
        if tp == 0:
            mfmeasure = 0
        else:
            pre = tp / binary.sum()
            rec = tp / hard_gt.sum()
            mfmeasure = 1.3 * pre * rec / (0.3 * pre + rec)

        ########################maxF##############################
        pred = np.uint8(pred * 255)
        target = pred[gt > 0.5]
        nontarget = pred[gt <= 0.5]
        targetHist, _ = np.histogram(target, bins=range(256))
        nontargetHist, _ = np.histogram(nontarget, bins=range(256))
        targetHist = np.cumsum(np.flip(targetHist), axis=0)
        nontargetHist = np.cumsum(np.flip(nontargetHist), axis=0)
        precision = targetHist / (targetHist + nontargetHist + 1e-8)
        recall = targetHist / np.sum(gt)
        return precision, recall, mfmeasure

    def show(self):
        assert self.num == self.idx, f"{self.num}, {self.idx}"
        precision = self.precision.mean(axis=0)
        recall = self.recall.mean(axis=0)
        fmeasure = 1.3 * precision * recall / (0.3 * precision + recall + 1e-8)
        mmfmeasure = self.meanF.mean()
        return fmeasure, fmeasure.max(), mmfmeasure, precision, recall


class CalMAE(object):
    # mean absolute error
    def __init__(self, num):
        # self.prediction = []
        self.prediction = np.zeros(num)
        self.idx = 0
        self.num = num

    def update(self, pred, gt):
        self.prediction[self.idx] = self.cal(pred, gt)
        self.idx += 1

    def cal(self, pred, gt):
        return np.mean(np.abs(pred - gt))

    def show(self):
        assert self.num == self.idx, f"{self.num}, {self.idx}"
        return self.prediction.mean()


class CalSM(object):
    # Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    def __init__(self, num, alpha=0.5):
        self.prediction = np.zeros(num)
        self.alpha = alpha
        self.idx = 0
        self.num = num

    def update(self, pred, gt):
        gt = gt > 0.5
        self.prediction[self.idx] = self.cal(pred, gt)
        self.idx += 1

    def show(self):
        assert self.num == self.idx, f"{self.num}, {self.idx}"
        return self.prediction.mean()

    def cal(self, pred, gt):
        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            score = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
        return score

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)

        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, np.logical_not(gt))

    def s_object(self, in1, in2):
        x = np.mean(in1[in2])
        sigma_x = np.std(in1[in2])
        return 2 * x / (pow(x, 2) + 1 + sigma_x + 1e-8)

    def region(self, pred, gt):
        [y, x] = center_of_mass(gt)
        y = int(round(y)) + 1
        x = int(round(x)) + 1
        [gt1, gt2, gt3, gt4, w1, w2, w3, w4] = self.divideGT(gt, x, y)
        pred1, pred2, pred3, pred4 = self.dividePred(pred, x, y)

        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def divideGT(self, gt, x, y):
        h, w = gt.shape
        area = h * w
        LT = gt[0:y, 0:x]
        RT = gt[0:y, x:w]
        LB = gt[y:h, 0:x]
        RB = gt[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area

        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePred(self, pred, x, y):
        h, w = pred.shape
        LT = pred[0:y, 0:x]
        RT = pred[0:y, x:w]
        LB = pred[y:h, 0:x]
        RB = pred[y:h, x:w]

        return LT, RT, LB, RB

    def ssim(self, in1, in2):
        in2 = np.float32(in2)
        h, w = in1.shape
        N = h * w

        x = np.mean(in1)
        y = np.mean(in2)
        sigma_x = np.var(in1)
        sigma_y = np.var(in2)
        sigma_xy = np.sum((in1 - x) * (in2 - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0

        return score


class CalEM(object):
    # Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    def __init__(self, num):
        self.prediction = np.zeros(num)
        self.idx = 0
        self.num = num

    def update(self, pred, gt):
        self.prediction[self.idx] = self.cal(pred, gt)
        self.idx += 1

    def cal(self, pred, gt):
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        FM = np.zeros(gt.shape)
        FM[pred >= th] = 1
        FM = np.array(FM, dtype=bool)
        GT = np.array(gt, dtype=bool)
        dFM = np.double(FM)
        if sum(sum(np.double(GT))) == 0:
            enhanced_matrix = 1.0 - dFM
        elif sum(sum(np.double(~GT))) == 0:
            enhanced_matrix = dFM
        else:
            dGT = np.double(GT)
            align_matrix = self.AlignmentTerm(dFM, dGT)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)
        [w, h] = np.shape(GT)
        score = sum(sum(enhanced_matrix)) / (w * h - 1 + 1e-8)
        return score

    def AlignmentTerm(self, dFM, dGT):
        mu_FM = np.mean(dFM)
        mu_GT = np.mean(dGT)
        align_FM = dFM - mu_FM
        align_GT = dGT - mu_GT
        align_Matrix = 2.0 * (align_GT * align_FM) / (align_GT * align_GT + align_FM * align_FM + 1e-8)
        return align_Matrix

    def EnhancedAlignmentTerm(self, align_Matrix):
        enhanced = np.power(align_Matrix + 1, 2) / 4
        return enhanced

    def show(self):
        assert self.num == self.idx, f"{self.num}, {self.idx}"
        return self.prediction.mean()


class CalWFM(object):
    def __init__(self, num, beta=1):
        self.scores_list = np.zeros(num)
        self.beta = beta
        self.eps = 1e-6
        self.idx = 0
        self.num = num

    def update(self, pred, gt):
        gt = gt > 0.5
        self.scores_list[self.idx] = 0 if gt.max() == 0 else self.cal(pred, gt)
        self.idx += 1

    def matlab_style_gauss2D(self, shape=(7, 7), sigma=5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def cal(self, pred, gt):
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        R = 1 - np.mean(Ew[gt])
        P = TPw / (self.eps + TPw + FPw)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (self.eps + R + self.beta * P)

        return Q

    def show(self):
        assert self.num == self.idx, f"{self.num}, {self.idx}"
        return self.scores_list.mean()


class CalTotalMetric(object):
    def __init__(self, num, beta_for_wfm=1):
        self.cal_mae = CalMAE(num=num)
        self.cal_fm = CalFM(num=num)
        self.cal_sm = CalSM(num=num)
        self.cal_em = CalEM(num=num)
        self.cal_wfm = CalWFM(num=num, beta=beta_for_wfm)

    def update(self, pred, gt):
        assert pred.ndim == gt.ndim and pred.shape == gt.shape
        assert pred.max() <= 1 and pred.min() >= 0
        assert gt.max() <= 1 and gt.min() >= 0

        self.cal_mae.update(pred, gt)
        self.cal_fm.update(pred, gt)
        self.cal_sm.update(pred, gt)
        self.cal_em.update(pred, gt)
        self.cal_wfm.update(pred, gt)

    def show(self):
        MAE = self.cal_mae.show()
        _, Maxf, Meanf, _, _, = self.cal_fm.show()
        SM = self.cal_sm.show()
        EM = self.cal_em.show()
        WFM = self.cal_wfm.show()
        results = {
            "MaxF": Maxf,
            "MeanF": Meanf,
            "WFM": WFM,
            "MAE": MAE,
            "SM": SM,
            "EM": EM,
        }
        return results


if __name__ == "__main__":
    pred = Image

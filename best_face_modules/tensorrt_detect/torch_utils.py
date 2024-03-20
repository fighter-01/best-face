from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms, nms


def seg_postprocess(
        data: Tuple[Tensor],
        shape: Union[Tuple, List],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = data[0][0], data[1][0]
    bboxes, scores, labels, maskconf = outputs.split([4, 1, 1, 32], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), labels.new_zeros((0, )), bboxes.new_zeros((0, 0, 0, 0))
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    idx = batched_nms(bboxes, scores, labels, iou_thres)
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx].int(), maskconf[idx]
    masks = (maskconf @ proto).sigmoid().view(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.)
    masks = F.interpolate(masks[None],
                          shape,
                          mode='bilinear',
                          align_corners=False)[0]
    masks = masks.gt_(0.5)[..., None]
    return bboxes, scores, labels, masks

def landmarks_postprocess(data, conf_thres=0.25, iou_thres=0.65) \
        -> Tuple[Tensor, Tensor, Tensor]:
    # 确保data包含三个张量：bboxes, scores, 和kpts
    assert len(data) == 3, "Expected data to contain 3 tensors, got {}".format(len(data))

    bboxes, scores, kpts = data

    # 处理Tensor格式和维度
    bboxes = bboxes.squeeze()  # 假定bboxes的形状为[1, N, 4]
    scores = scores.squeeze()  # 假定scores的形状为[1, N, 1]
    kpts = kpts.squeeze()  # 假定kpts的形状为[1, N, 51]

    # 应用置信度阈值过滤检测结果
    idx = scores > conf_thres
    if not torch.any(idx):
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0, 17, 3))

    bboxes, scores, kpts = bboxes[idx], scores[idx], kpts[idx]

    # 转换bbox格式从中心点+宽高到左上角+右下角
    xycenter, wh = bboxes.chunk(2, dim=-1)
    bboxes = torch.cat([xycenter - 0.5 * wh, xycenter + 0.5 * wh], dim=-1)

    # 应用NMS
    keep = nms(bboxes, scores.squeeze(-1), iou_thres)

    bboxes, scores, kpts = bboxes[keep], scores[keep], kpts[keep]

    return bboxes, scores, kpts.view(-1, 17, 3)

def pose_postprocess(
        data: Union[Tuple, Tensor],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[Tensor, Tensor, Tensor]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    outputs = torch.transpose(data[0], 0, 1).contiguous()
    bboxes, scores, kpts = outputs.split([4, 1, 51], 1)
    scores, kpts = scores.squeeze(), kpts.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), bboxes.new_zeros((0, 0, 0))
    bboxes, scores, kpts = bboxes[idx], scores[idx], kpts[idx]
    xycenter, wh = bboxes.chunk(2, -1)
    bboxes = torch.cat([xycenter - 0.5 * wh, xycenter + 0.5 * wh], -1)
    idx = nms(bboxes, scores, iou_thres)
    bboxes, scores, kpts = bboxes[idx], scores[idx], kpts[idx]
    return bboxes, scores, kpts.reshape(idx.shape[0], -1, 3)


def det_postprocess(data: Tuple[Tensor, Tensor, Tensor, Tensor]):
    assert len(data) == 4
    iou_thres: float = 0.65
    num_dets, bboxes, scores, labels = data[0][0], data[1][0], data[2][
        0], data[3][0]
    nums = num_dets.item()
    if nums == 0:
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), labels.new_zeros((0, ))
    # check score negative
    scores[scores < 0] = 1 + scores[scores < 0]
    # add nms
    idx = nms(bboxes, scores, iou_thres)
    bboxes, scores, labels = bboxes[idx], scores[idx], labels[idx]
    bboxes = bboxes[:nums]
    scores = scores[:nums]
    labels = labels[:nums]

    return bboxes, scores, labels


def det_postprocess_new(preds, scale_h, scale_w, padh, padw):
    preds = preds.transpose((0, 2, 1))
    bboxes = preds[:,:,:4]
    scores = preds[:,:,4:5]
    landmarks = preds[:,:,5:]

    bboxes = np.concatenate(bboxes, axis=0)
    x1y1 = (bboxes[:, 0:2] * 2 - bboxes[:, 2:]) / 2
    x2y2 = (bboxes[:, 0:2] * 2 + bboxes[:, 2:]) / 2
    bboxes[:, 0:2] = x1y1
    bboxes[:, 2:] = x2y2
    scores = np.concatenate(scores, axis=0)
    landmarks = np.concatenate(landmarks, axis=0)

    bboxes -= np.array([[padw, padh, padw, padh]])  ###合理使用广播法则
    bboxes *= np.array([[scale_w, scale_h, scale_w, scale_h]])
    landmarks -= np.tile(np.array([padw, padh, 0]), 5).reshape((1,15))
    landmarks *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1,15))

    bboxes_wh = bboxes.copy()
    bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh

    classIds = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)  ####max_class_confidence

    mask = confidences>self.conf_threshold
    bboxes_wh = bboxes_wh[mask]  ###合理使用广播法则
    confidences = confidences[mask]
    classIds = classIds[mask]
    landmarks = landmarks[mask]
    if len(landmarks) <= 0:
        print('nothing detect')
        return np.array([]), np.array([]), np.array([]), np.array([])

    indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold,
                                self.iou_threshold).flatten()
    if len(indices) > 0:
        mlvl_bboxes = bboxes_wh[indices]
        confidences = confidences[indices]
        classIds = classIds[indices]
        landmarks = landmarks[indices]
        return mlvl_bboxes, confidences, classIds, landmarks
    else:
        print('nothing detect')
        return np.array([]), np.array([]), np.array([]), np.array([])


def crop_mask(masks: Tensor, bboxes: Tensor) -> Tensor:
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(bboxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device,
                     dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device,
                     dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

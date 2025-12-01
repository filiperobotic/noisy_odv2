# custom_hooks/wandb_pred_buckets_hook.py
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import torch
import numpy as np
import inspect
import cv2

# Compat de bbox_overlaps (mudou entre versões)
def _bbox_overlaps(a, b, mode='iou'):
    # a: (Na, 4) xyxy ; b: (Nb, 4)
    # retorna (Na, Nb)
    try:
        # MMDet 3.x
        from mmdet.evaluation.functional import bbox_overlaps
    except Exception:
        try:
            # algumas builds
            from mmdet.structures.bbox import bbox_overlaps
        except Exception:
            bbox_overlaps = None

    if bbox_overlaps is not None:
        # Converte tensores para numpy se a implementação de bbox_overlaps esperar numpy
        a_in = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
        b_in = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else b
        try:
            sig = inspect.signature(bbox_overlaps)
            if 'is_aligned' in sig.parameters:
                out = bbox_overlaps(a_in, b_in, mode=mode, is_aligned=False)
            else:
                out = bbox_overlaps(a_in, b_in, mode)
        except Exception:
            # Fallbacks de assinatura
            try:
                out = bbox_overlaps(a_in, b_in, mode)
            except TypeError:
                out = bbox_overlaps(a_in, b_in)
        # Garante retorno como tensor torch
        if isinstance(out, np.ndarray):
            return torch.from_numpy(out)
        return out

    # Fallback puro-numpy
    Na = a.shape[0]
    Nb = b.shape[0]
    ious = np.zeros((Na, Nb), dtype=np.float32)
    for i in range(Na):
        x1, y1, x2, y2 = a[i]
        area_a = max(0, x2 - x1) * max(0, y2 - y1)
        for j in range(Nb):
            xx1 = max(x1, b[j, 0]); yy1 = max(y1, b[j, 1])
            xx2 = min(x2, b[j, 2]); yy2 = min(y2, b[j, 3])
            w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
            inter = w * h
            area_b = max(0, b[j, 2] - b[j, 0]) * max(0, b[j, 3] - b[j, 1])
            union = area_a + area_b - inter + 1e-6
            ious[i, j] = inter / union if union > 0 else 0.0
    return torch.from_numpy(ious)

# Filtra instâncias preditas por IoU com qualquer GT
# keep_matched=True mantém apenas preds com IoU >= thr; False mantém IoU < thr
# Filtra instâncias preditas por IoU com qualquer GT
# keep_matched=True  -> mantém apenas preds com IoU >= thr
# keep_matched=False -> mantém apenas preds com IoU <  thr
def _filter_pred_by_iou(ds, thr, keep_matched=True):
    pred = getattr(ds, 'pred_instances', None)
    gt = getattr(ds, 'gt_instances', None)

    if pred is None or not hasattr(pred, 'bboxes') or len(pred.bboxes) == 0:
        return ds, None

    device = pred.bboxes.device

    # Sem GT: se keep_matched, retorna vazio; senão, tudo é "unmatched"
    if gt is None or not hasattr(gt, 'bboxes') or len(gt.bboxes) == 0:
        if keep_matched:
            ds_out = ds.clone()
            empty_mask = torch.zeros((len(pred.bboxes),), dtype=torch.bool, device=device)
            ds_out.pred_instances = pred[empty_mask]
            return ds_out, empty_mask
        else:
            full_mask = torch.ones((len(pred.bboxes),), dtype=torch.bool, device=device)
            return ds, full_mask

    # IoU (usa helper que aceita torch/np e normaliza saída)
    pb = pred.bboxes.detach().cpu()
    gb = gt.bboxes.detach().cpu()
    ious = _bbox_overlaps(pb, gb)
    if not torch.is_tensor(ious):
        ious = torch.as_tensor(ious)

    if ious.numel() > 0:
        max_iou_pred, _ = ious.max(dim=1)   # para cada pred, melhor GT
    else:
        max_iou_pred = torch.zeros((len(pred.bboxes),))

    if keep_matched:
        mask = (max_iou_pred >= thr)
    else:
        mask = (max_iou_pred < thr)

    # garante bool + device correto
    mask = mask.to(torch.bool).to(device)

    ds_out = ds.clone()
    if mask.any().item():
        ds_out.pred_instances = pred[mask]
    else:
        empty_mask = torch.zeros((len(pred.bboxes),), dtype=torch.bool, device=device)
        ds_out.pred_instances = pred[empty_mask]

    return ds_out, mask



# Garante que pred.scores exista e esteja no intervalo [0,1], formatado com ndigits casas
def _ensure_scores(ds, ndigits=2):
    pred = getattr(ds, 'pred_instances', None)
    if pred is None:
        return ds
    scores = getattr(pred, 'scores', None)
    if scores is None:
        # Se não houver scores, cria scores dummy = 1.0 (evita usar outros tensores por engano)
        if hasattr(pred, 'bboxes') and len(pred.bboxes) > 0:
            pred.scores = torch.ones(len(pred.bboxes), device=pred.bboxes.device, dtype=torch.float32)
        return ds
    # Converte para float
    if not torch.is_floating_point(scores):
        scores = scores.float()
    # Se valores forem claramente fora de [0,1], normaliza de forma conservadora
    maxv = float(scores.max().detach().cpu()) if scores.numel() > 0 else 0.0
    minv = float(scores.min().detach().cpu()) if scores.numel() > 0 else 0.0
    if maxv > 1.5:  # pode ser logits/índices/percentuais
        if maxv <= 100.0:      # parece percentual (ex.: 57.0)
            scores = scores / 100.0
        else:                  # assume logits: aplica sigmoid
            scores = scores.sigmoid()
    elif minv < 0.0:
        # valores negativos: provável logit -> sigmoid
        scores = scores.sigmoid()
    # Arredonda
    factor = 10 ** ndigits
    scores = (scores * factor).round() / factor
    pred.scores = scores
    return ds

# Desenha caixas e rótulos personalizados: "classe: p_pred | gt(classe_gt): p_gt"
# Requer ds.pred_instances; se `cls_scores` (KxC) existir, calcula p_gt; caso contrário, exibe só p_pred.
# Retorna uma nova imagem renderizada.

# Filtra instâncias preditas por score (mantém apenas scores >= thr)

def _filter_pred_by_score(ds, thr: float):
    pred = getattr(ds, 'pred_instances', None)
    if pred is None or not hasattr(pred, 'bboxes') or len(pred.bboxes) == 0:
        return ds, None
    scores = getattr(pred, 'scores', None)
    if scores is None:
        return ds, None
    # Garante tensor e cria máscara booleana
    if not torch.is_tensor(scores):
        scores = torch.as_tensor(scores, device=pred.bboxes.device)
    mask = (scores >= thr).to(torch.bool)

    ds_out = ds.clone()
    if mask.any().item():
        ds_out.pred_instances = pred[mask]
    else:
        # seleção vazia segura
        empty_mask = torch.zeros((len(pred.bboxes),), dtype=torch.bool, device=pred.bboxes.device)
        ds_out.pred_instances = pred[empty_mask]
    return ds_out, mask


# Desenha caixas e rótulos personalizados: "classe: p_pred | gt(classe_gt): p_gt"
# Requer ds.pred_instances; se `cls_scores` (KxC) existir, calcula p_gt; caso contrário, exibe só p_pred.
# Retorna uma nova imagem renderizada.

def _render_preds_with_two_scores(img_rgb, ds, dataset_meta=None, iou_thr=0.5):
    img = img_rgb.copy()
    pred = getattr(ds, 'pred_instances', None)
    gt = getattr(ds, 'gt_instances', None)
    if pred is None or not hasattr(pred, 'bboxes') or len(pred.bboxes) == 0:
        return img

    # nomes de classe (se disponíveis)
    classes = None
    if isinstance(dataset_meta, dict):
        classes = dataset_meta.get('classes', None)

    pb = pred.bboxes.detach().cpu().numpy().astype(float)
    pl = pred.labels.detach().cpu().numpy().astype(int) if hasattr(pred, 'labels') else None
    ps = pred.scores.detach().cpu().numpy().astype(float) if hasattr(pred, 'scores') and pred.scores is not None else None

    cls_scores = getattr(pred, 'cls_scores', None)
    if cls_scores is not None:
        cls_scores = cls_scores.detach().cpu().numpy()

    # match pred->gt por IoU (para recuperar classe GT correspondente)
    best_gt_idx = None
    best_iou = None
    if gt is not None and hasattr(gt, 'bboxes') and len(gt.bboxes) > 0:
        gb = gt.bboxes.detach().cpu().numpy().astype(float)
        ious = _bbox_overlaps(torch.as_tensor(pb), torch.as_tensor(gb))
        if not torch.is_tensor(ious):
            ious = torch.as_tensor(ious)
        if ious.numel() > 0:
            best_iou_t, best_idx_t = ious.max(dim=1)  # por pred
            best_gt_idx = best_idx_t.detach().cpu().numpy()
            best_iou = best_iou_t.detach().cpu().numpy()

    # desenha
    H, W = img.shape[:2]
    for i in range(pb.shape[0]):
        x1, y1, x2, y2 = pb[i].astype(int)
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        color = (0, 102, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # classe predita e score da classe predita
        cls_id = int(pl[i]) if pl is not None else -1
        cls_name = classes[cls_id] if (classes is not None and 0 <= cls_id < len(classes)) else str(cls_id)
        p_pred = None
        if cls_scores is not None:
            p_pred = float(cls_scores[i, cls_id])
        elif ps is not None:
            p_pred = float(ps[i])

        # score da classe GT (se houver match e cls_scores existir)
        text = f"{cls_name}"
        if p_pred is not None:
            text += f": {p_pred:.2f}"
        if (best_gt_idx is not None) and (best_iou is not None) and (cls_scores is not None):
            if best_iou[i] >= iou_thr:
                gt_idx = int(best_gt_idx[i])
                gt_label = None
                if hasattr(gt, 'labels') and len(gt.labels) > gt_idx:
                    gt_label = int(gt.labels[gt_idx].item())
                if gt_label is not None and 0 <= gt_label < cls_scores.shape[1]:
                    p_gt = float(cls_scores[i, gt_label])
                    gt_name = classes[gt_label] if (classes is not None and 0 <= gt_label < len(classes)) else str(gt_label)
                    text += f" | gt({gt_name}): {p_gt:.2f}"

        # coloca texto
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bx1, by1 = x1, max(0, y1 - th - 6)
        cv2.rectangle(img, (bx1, by1), (bx1 + tw + 6, by1 + th + 6), (0, 0, 0), -1)
        cv2.putText(img, text, (bx1 + 3, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return img

@HOOKS.register_module()
class WandbPredBucketsHook(Hook):
    """Loga imagens por buckets: high-confidence (score>=high_conf_thr), mismatch (pred != GT), false_positive e false_negative. Usa dois limiares: high_conf_thr (ex.: 0.9) e min_pred_thr (ex.: 0.5)."""
    def __init__(self,
                 num_images=16,
                 high_conf_thr=0.90,
                 min_pred_thr=0.50,
                 iou_thr=0.50,
                 every_n_epochs=1,
                 random_sample=False):
        self.num_images = num_images
        self.high_conf_thr = high_conf_thr  # limiar do bucket high_conf
        self.min_pred_thr = min_pred_thr    # limiar mínimo para demais buckets
        self.iou_thr = iou_thr
        self.every_n_epochs = every_n_epochs
        self.random_sample = random_sample

    def after_val_epoch(self, runner, metrics=None):
        # Loga a cada N épocas
        if (runner.epoch + 1) % self.every_n_epochs != 0:
            return

        model = runner.model
        model.eval()

        dl = runner.val_dataloader
        it = iter(dl)
        batch = next(it)

        with torch.no_grad():
            data_samples = model.val_step(batch)

        visualizer = runner.visualizer
        imgs = batch['inputs']   # Tensor BxCxHxW
        B = len(data_samples)
        n = min(self.num_images, B)

        # Garante que o visualizer conheça os metadados do dataset (nomes de classes)
        if not getattr(visualizer, 'dataset_meta', None):
            try:
                visualizer.dataset_meta = runner.val_dataloader.dataset.metainfo
            except Exception:
                pass
        dataset_meta = getattr(visualizer, 'dataset_meta', None)

        for i in range(n):
            ds = data_samples[i]

            # Carrega a imagem original do caminho do dataset (preferível para manter escala e cores corretas)
            img_np = None
            try:
                img_path = None
                if hasattr(ds, 'metainfo') and isinstance(ds.metainfo, dict):
                    img_path = ds.metainfo.get('img_path', None)
                if img_path is not None:
                    # cv2 lê em BGR; convertendo para RGB para o visualizer (espera RGB)
                    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        img_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                img_np = None
            if img_np is None:
                # fallback para o tensor do batch (pode estar normalizado)
                img_np = imgs[i].detach().cpu().numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            pred = ds.pred_instances  # bboxes (N,4), scores (N,), labels (N,)
            gt = getattr(ds, 'gt_instances', None)

            # -------- Bucket 1: High confidence (≥ high_conf_thr) --------
            if pred is not None and hasattr(pred, 'scores'):
                high_idx = (pred.scores >= self.high_conf_thr).nonzero().flatten()
                if high_idx.numel() > 0:
                    # Clonar data_sample e filtrar só caixas high-conf
                    ds_high = ds.clone()
                    # Seleciona apenas as instâncias high-conf para desenhar
                    ds_high.pred_instances = pred[high_idx]
                    # Mantém apenas preds com IoU >= iou_thr
                    ds_high, keep_mask = _filter_pred_by_iou(ds_high, self.iou_thr, keep_matched=True)
                    # Se nada sobrou após o filtro por IoU, não loga
                    if (getattr(ds_high, 'pred_instances', None) is None) or \
                       (hasattr(ds_high.pred_instances, 'bboxes') and len(ds_high.pred_instances.bboxes) == 0):
                        continue
                    ds_high = _ensure_scores(ds_high, ndigits=2)
                    rendered = _render_preds_with_two_scores(img_np, ds_high, dataset_meta, iou_thr=self.iou_thr)
                    visualizer.add_datasample(
                        name=f'val_pred/high_conf/ep{runner.epoch+1}_idx{i}',
                        image=rendered,
                        data_sample=ds_high,
                        draw_gt=False,
                        draw_pred=False,
                        pred_score_thr=0.0,  # já filtramos
                        out_file=None,
                        show=False,
                        wait_time=0,
                        step=runner.epoch + 1
                    )

            # -------- Bucket 2: Mismatch pred != GT (com match por IoU) --------
            has_mismatch = False
            if (gt is not None and hasattr(gt, 'bboxes') and len(gt.bboxes) > 0
                and pred is not None and hasattr(pred, 'bboxes') and len(pred.bboxes) > 0):

                # Converte para cpu numpy
                pb = pred.bboxes.detach().cpu()
                pl = pred.labels.detach().cpu()
                gb = gt.bboxes.detach().cpu()
                gl = gt.labels.detach().cpu()

                # Matriz IoU: (Npred, Ngt)
                ious = _bbox_overlaps(pb, gb)  # tensor (Npred, Ngt) ou numpy
                if not torch.is_tensor(ious):
                    ious = torch.as_tensor(ious)

                # Para cada GT, pega melhor pred acima de iou_thr
                if ious.numel() > 0:
                    best_pred_iou, best_pred_idx = ious.max(dim=0)  # por GT
                    for gt_j in range(gb.shape[0]):
                        if best_pred_iou[gt_j] >= self.iou_thr:
                            pj = best_pred_idx[gt_j].item()
                            # compara labels
                            if int(pl[pj].item()) != int(gl[gt_j].item()):
                                has_mismatch = True
                                break

            if has_mismatch:
                # Mantém apenas preds com IoU >= iou_thr para exibição
                ds_mis, _ = _filter_pred_by_iou(ds, self.iou_thr, keep_matched=True)
                # Filtra preds por min_pred_thr (não high_conf_thr)
                ds_mis, _ = _filter_pred_by_score(ds_mis, self.min_pred_thr)
                # Se nada sobrou após o filtro por IoU, não loga
                if (getattr(ds_mis, 'pred_instances', None) is None) or \
                   (hasattr(ds_mis.pred_instances, 'bboxes') and len(ds_mis.pred_instances.bboxes) == 0):
                    pass
                else:
                    ds_mis = _ensure_scores(ds_mis, ndigits=2)
                    rendered = _render_preds_with_two_scores(img_np, ds_mis, dataset_meta, iou_thr=self.iou_thr)
                    visualizer.add_datasample(
                        name=f'val_pred/mismatch/ep{runner.epoch+1}_idx{i}',
                        image=rendered,
                        data_sample=ds_mis,
                        draw_gt=True,
                        draw_pred=False,
                        pred_score_thr=0.0,
                        out_file=None,
                        show=False,
                        wait_time=0,
                        step=runner.epoch + 1
                    )

            # -------- Bucket 3: True Positives (IoU >= thr e label correta) --------
            has_tp = False
            if (gt is not None and hasattr(gt, 'bboxes') and len(gt.bboxes) > 0
                and pred is not None and hasattr(pred, 'bboxes') and len(pred.bboxes) > 0):

                pb = pred.bboxes.detach().cpu()
                pl = pred.labels.detach().cpu()
                gb = gt.bboxes.detach().cpu()
                gl = gt.labels.detach().cpu()

                ious_tp = _bbox_overlaps(pb, gb)
                if not torch.is_tensor(ious_tp):
                    ious_tp = torch.as_tensor(ious_tp)
                if ious_tp.numel() > 0:
                    # melhor GT para cada pred
                    best_gt_iou, best_gt_idx = ious_tp.max(dim=1)  # por pred
                    tp_mask = (best_gt_iou >= self.iou_thr) & (pl == gl[best_gt_idx])
                    has_tp = tp_mask.any().item()
                else:
                    tp_mask = torch.zeros((len(pb),), dtype=torch.bool)

            if has_tp:
                ds_tp = ds.clone()
                ds_tp.pred_instances = pred[tp_mask]
                ds_tp, _ = _filter_pred_by_score(ds_tp, self.min_pred_thr)
                if (getattr(ds_tp, 'pred_instances', None) is None) or \
                   (hasattr(ds_tp.pred_instances, 'bboxes') and len(ds_tp.pred_instances.bboxes) == 0):
                    ds_tp = None
                if ds_tp is not None:
                    ds_tp = _ensure_scores(ds_tp, ndigits=2)
                    rendered = _render_preds_with_two_scores(img_np, ds_tp, dataset_meta, iou_thr=self.iou_thr)
                    visualizer.add_datasample(
                        name=f'val_pred/true_positive/ep{runner.epoch+1}_idx{i}',
                        image=rendered,
                        data_sample=ds_tp,
                        draw_gt=True,
                        draw_pred=False,
                        pred_score_thr=0.0,
                        out_file=None,
                        show=False,
                        wait_time=0,
                        step=runner.epoch + 1
                    )

            # -------- Bucket 4: False Positives (pred sem match >= IoU) --------
            if (pred is not None and hasattr(pred, 'bboxes') and len(pred.bboxes) > 0
                and gt is not None and hasattr(gt, 'bboxes')):
                # Se não houver GT, todos os preds são FP
                if gt is None or len(gt.bboxes) == 0:
                    fp_mask = torch.ones((len(pred.bboxes),), dtype=torch.bool)
                else:
                    # ious: (Npred, Ngt) já calculado acima quando possível; se não, recalcule
                    if 'ious' not in locals() or ious is None or ious.numel() == 0:
                        pb = pred.bboxes.detach().cpu()
                        gb = gt.bboxes.detach().cpu() if gt is not None else torch.empty((0, 4))
                        ious = _bbox_overlaps(pb, gb)
                        if not torch.is_tensor(ious):
                            ious = torch.as_tensor(ious)
                    # max IoU de cada pred com qualquer GT
                    if ious.numel() > 0:
                        max_iou_pred, _ = ious.max(dim=1)
                    else:
                        max_iou_pred = torch.zeros((len(pred.bboxes),))
                    fp_mask = max_iou_pred < self.iou_thr
                if fp_mask.any():
                    ds_fp = ds.clone()
                    ds_fp.pred_instances = pred[fp_mask]
                    ds_fp, _ = _filter_pred_by_score(ds_fp, self.min_pred_thr)
                    ds_fp = _ensure_scores(ds_fp, ndigits=2)
                    rendered = _render_preds_with_two_scores(img_np, ds_fp, dataset_meta, iou_thr=self.iou_thr)
                    # desenha apenas as preds FP (sem GT)
                    visualizer.add_datasample(
                        name=f'val_pred/false_positive/ep{runner.epoch+1}_idx{i}',
                        image=rendered,
                        data_sample=ds_fp,
                        draw_gt=False,
                        draw_pred=False,
                        pred_score_thr=0.0,
                        out_file=None,
                        show=False,
                        wait_time=0,
                        step=runner.epoch + 1
                    )

            # -------- Bucket 5: False Negatives (GT sem match >= IoU) --------
            if (gt is not None and hasattr(gt, 'bboxes') and len(gt.bboxes) > 0):
                # Se não houver preds, todos os GTs são FN
                if pred is None or not hasattr(pred, 'bboxes') or len(pred.bboxes) == 0:
                    fn_mask = torch.ones((len(gt.bboxes),), dtype=torch.bool)
                else:
                    # ious: (Npred, Ngt). Precisamos do max por GT
                    if 'ious' not in locals() or ious is None or ious.numel() == 0:
                        pb = pred.bboxes.detach().cpu()
                        gb = gt.bboxes.detach().cpu()
                        ious = _bbox_overlaps(pb, gb)
                        if not torch.is_tensor(ious):
                            ious = torch.as_tensor(ious)
                    if ious.numel() > 0:
                        max_iou_gt, _ = ious.max(dim=0)
                    else:
                        max_iou_gt = torch.zeros((len(gt.bboxes),))
                    fn_mask = max_iou_gt < self.iou_thr
                if fn_mask.any():
                    ds_fn = ds.clone()
                    # Para destacar apenas GTs FN, removemos as preds
                    empty_slice = slice(0, 0)
                    try:
                        ds_fn.pred_instances = pred[empty_slice]
                    except Exception:
                        ds_fn.pred_instances = None
                    visualizer.add_datasample(
                        name=f'val_pred/false_negative/ep{runner.epoch+1}_idx{i}',
                        image=img_np,
                        data_sample=ds_fn,
                        draw_gt=True,
                        draw_pred=False,
                        pred_score_thr=0.0,
                        out_file=None,
                        show=False,
                        wait_time=0,
                        step=runner.epoch + 1
                    )
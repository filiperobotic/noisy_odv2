from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import torch
from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.models.task_modules.assigners import MaxIoUAssigner
from collections import Counter
from torch.utils.data import DataLoader
import copy
import torch.nn.functional as F
from mmengine.structures import InstanceData
from collections import defaultdict
import numpy as np
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
from custom_configs.hooks.my_utils import weighted_knn
import matplotlib.pyplot as plt


import cv2
import os

# ---------------------------
# Helper: pairwise_iou_xyxy
# ---------------------------
def pairwise_iou_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU for boxes in xyxy format.

    Args:
        boxes: Tensor of shape (N, 4) in (x1, y1, x2, y2).
    Returns:
        iou: Tensor of shape (N, N).
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 0))

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # intersection
    xx1 = torch.max(x1[:, None], x1[None, :])
    yy1 = torch.max(y1[:, None], y1[None, :])
    xx2 = torch.min(x2[:, None], x2[None, :])
    yy2 = torch.min(y2[:, None], y2[None, :])

    inter_w = (xx2 - xx1).clamp(min=0)
    inter_h = (yy2 - yy1).clamp(min=0)
    inter = inter_w * inter_h

    union = area[:, None] + area[None, :] - inter
    iou = torch.where(union > 0, inter / union, union.new_zeros(()))
    return iou


def pairwise_containment_ratio_xyxy(boxes: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Matrix R where R[i,j] = area(intersection(box_i, box_j)) / area(box_i).
    Interpretação: fração do box_i que está dentro do box_j.
    boxes: (N,4) em xyxy
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 0))

    # (N,1)
    x1 = boxes[:, 0:1]
    y1 = boxes[:, 1:2]
    x2 = boxes[:, 2:3]
    y2 = boxes[:, 3:4]

    # broadcast (N,N)
    xx1 = torch.maximum(x1, x1.T)
    yy1 = torch.maximum(y1, y1.T)
    xx2 = torch.minimum(x2, x2.T)
    yy2 = torch.minimum(y2, y2.T)

    inter_w = (xx2 - xx1).clamp(min=0)
    inter_h = (yy2 - yy1).clamp(min=0)
    inter = inter_w * inter_h

    area_i = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)).squeeze(1).clamp(min=eps)
    return inter / area_i[:, None]


# ----------- Helper: match GT boxes to data_list indices -----------
def _match_gt_to_datalist_indices(gt_boxes: torch.Tensor,
                                 datalist_boxes: torch.Tensor,
                                 datalist_indices: list,
                                 tol: float = 1e-3) -> list:
    """Greedy match: for each gt box (Nx4 xyxy), find the closest box in datalist_boxes (Mx4 xyxy).

    Returns a list `gt_to_instidx` of length N with entries from `datalist_indices` (or -1 if unmatched).
    Matching uses max-abs coordinate difference and enforces one-to-one assignment.
    """
    if gt_boxes.numel() == 0:
        return []
    if datalist_boxes.numel() == 0:
        return [-1] * gt_boxes.shape[0]

    gt_to_instidx = [-1] * gt_boxes.shape[0]
    used = set()

    # compute max-abs diff cost matrix (N,M)
    # cost[n,m] = max(|gt[n]-dl[m]|)
    diff = (gt_boxes[:, None, :] - datalist_boxes[None, :, :]).abs()
    cost = diff.max(dim=-1).values

    for n in range(gt_boxes.shape[0]):
        # pick best unused match
        row = cost[n]
        # mask used columns
        if used:
            used_cols = torch.tensor(sorted(list(used)), device=row.device, dtype=torch.long)
            row = row.clone()
            row[used_cols] = float('inf')
        m = int(torch.argmin(row).item())
        best = float(row[m].item())
        if best <= tol:
            used.add(m)
            gt_to_instidx[n] = int(datalist_indices[m])

    return gt_to_instidx

# --- Helper: restore boxes from augmented (flipped+resized) to original image coordinates ---
def _restore_boxes_to_original_xyxy(boxes_xyxy: torch.Tensor, metainfo: dict) -> torch.Tensor:
    """Restore boxes from augmented image coords back to original image coords.

    Assumes boxes are xyxy in the current image space (after resize/flip).
    Uses metainfo keys: flip (bool), flip_direction (str), img_shape (h,w), scale_factor.

    Returns boxes in original (unresized, unflipped) xyxy.
    """
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy

    b = boxes_xyxy.clone()

    # 1) Undo flip in the *current* image space
    flip = metainfo.get('flip', False)
    if flip:
        direction = metainfo.get('flip_direction', 'horizontal')
        # img_shape is (h, w) of the current image (after resize)
        img_shape = metainfo.get('img_shape', None)
        if img_shape is None:
            # fallback: don't flip if we cannot determine width/height
            pass
        else:
            h, w = int(img_shape[0]), int(img_shape[1])
            if direction == 'horizontal':
                x1 = b[:, 0].clone()
                x2 = b[:, 2].clone()
                b[:, 0] = w - x2
                b[:, 2] = w - x1
            elif direction == 'vertical':
                y1 = b[:, 1].clone()
                y2 = b[:, 3].clone()
                b[:, 1] = h - y2
                b[:, 3] = h - y1
            elif direction == 'diagonal':
                # diagonal = horizontal + vertical
                x1 = b[:, 0].clone()
                x2 = b[:, 2].clone()
                y1 = b[:, 1].clone()
                y2 = b[:, 3].clone()
                b[:, 0] = w - x2
                b[:, 2] = w - x1
                b[:, 1] = h - y2
                b[:, 3] = h - y1

    # 2) Undo resize (back to original image coords)
    # scale_factor can be (4,) or (2,)
    sf = metainfo.get('scale_factor', None)
    if sf is not None:
        if isinstance(sf, (list, tuple)):
            sf_t = torch.tensor(sf, dtype=b.dtype, device=b.device)
        elif isinstance(sf, np.ndarray):
            sf_t = torch.tensor(sf.tolist(), dtype=b.dtype, device=b.device)
        else:
            # already tensor
            sf_t = sf.to(device=b.device, dtype=b.dtype)

        if sf_t.numel() == 4:
            # [w_scale, h_scale, w_scale, h_scale]
            w_scale = sf_t[0].clamp(min=1e-9)
            h_scale = sf_t[1].clamp(min=1e-9)
            b[:, 0] = b[:, 0] / w_scale
            b[:, 2] = b[:, 2] / w_scale
            b[:, 1] = b[:, 1] / h_scale
            b[:, 3] = b[:, 3] / h_scale
        elif sf_t.numel() == 2:
            w_scale = sf_t[0].clamp(min=1e-9)
            h_scale = sf_t[1].clamp(min=1e-9)
            b[:, 0] = b[:, 0] / w_scale
            b[:, 2] = b[:, 2] / w_scale
            b[:, 1] = b[:, 1] / h_scale
            b[:, 3] = b[:, 3] / h_scale

    return b

def tensor_to_numpy_img(tensor_img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """
    Converte tensor [3, H, W] (normalizado) em imagem numpy [H, W, 3] contígua para visualização.
    """
    img = tensor_img.clone().cpu().numpy()
    img = img.transpose(1, 2, 0)  # [C, H, W] → [H, W, C]
    img = (img * std) + mean     # desnormaliza
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = img[..., ::-1]  # ✅ converte RGB → BGR para o OpenCV
    return img.copy()  # <- AQUI é o segredo


def desenhar_bboxesv3(img_path, tensor_img, instances, save_path='bbox_debug.jpg', color=(0, 255, 0), thickness=2):
    # Carregar imagem
    # img_np = tensor_to_numpy_img(tensor_img)  # [H, W, 3] uint8
    img_np = tensor_to_numpy_img(tensor_img.cpu())

    # Agora você pode desenhar com OpenCV
    for inst in instances:
        bbox = inst['bboxes'].tensor[0].cpu().numpy().astype(int)
        # bbox = inst['bboxes'][0].cpu().numpy().astype(int)
        label = inst['labels'].cpu().item()
        ignore_flag = inst.get('ignore_flag', 0)
        color = (0, 0, 255) if ignore_flag else (0, 255, 0)

        cv2.rectangle(img_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(img_np, f"Cls:{label}", (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        

    # Criar pasta, se necessário
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Salvar a imagem
    # cv2.imwrite(save_path, img)
    cv2.imwrite(save_path, img_np)
    print(f"[INFO] Imagem salva em: {save_path}")

#def desenhar_bboxesv3_pred(img_path, tensor_img, gt_instances, pred_instances,associated_preds, save_path='bbox_debug.jpg', color=(0, 255, 0), thickness=2):
def desenhar_bboxesv3_pred(tensor_img, gt_instances, pred_instances,iou, associated_preds, gt_pred, save_path='bbox_debug.jpg', color=(0, 255, 0), thickness=2):
    # Carregar imagem
    # img_np = tensor_to_numpy_img(tensor_img)  # [H, W, 3] uint8
    img_np = tensor_to_numpy_img(tensor_img.cpu())

    # Agora você pode desenhar com OpenCV
    for inst in gt_instances:
        bbox = inst['bboxes'].tensor[0].cpu().numpy().astype(int)
        # bbox = inst['bboxes'][0].cpu().numpy().astype(int)
        label = inst['labels'].cpu().item()
        ignore_flag = inst.get('ignore_flag', 0)
        color = (0, 0, 255) if ignore_flag else (0, 255, 0)

        cv2.rectangle(img_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(img_np, f"Cls:{label}", (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    #  Desenhar predições (azul)
    for bbox, label, score, inter, logits in zip(pred_instances['priors'][associated_preds], pred_instances['labels'][associated_preds], pred_instances['scores'][associated_preds],iou[associated_preds], pred_instances['logits'][associated_preds]):
        bbox = bbox.cpu().numpy().astype(int)
        label = int(label)
        score = float(score)

        # import pdb; pdb.set_trace()
        myscores = torch.softmax(logits ,dim=-1)
        myscores_pred =  myscores.max(dim=0).values.item()
        myscore_gt = myscores[gt_pred].item()
        mylabel_pred_temp = myscores.argmax(dim=0).item()
        # if myscores_pred <0.7:
        #     import pdb; pdb.set_trace
        # if (label == gt_pred) and myscore_gt!=myscores_pred:
        #     import pdb; pdb.set_trace() 
        

        cv2.rectangle(img_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        #cv2.putText(img_np, f"Pred:{label} ({score:.2f}/{myscore_gt:.2f}/{inter:.2f})", (bbox[0], bbox[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(img_np, f"Pred:{mylabel_pred_temp}/{label}-{gt_pred} ({myscores_pred:.2f}/{myscore_gt:.2f}/{inter:.2f})", (bbox[0], bbox[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        

    # Criar pasta, se necessário
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Salvar a imagem
    # cv2.imwrite(save_path, img)
    cv2.imwrite(save_path, img_np)
    print(f"[INFO] Imagem salva em: {save_path}")

def calculate_gmm(c_class_scores, n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42  ):

    #gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
    gmm = GaussianMixture(n_components=n_components, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
    gmm.fit(c_class_scores)

    # Identificar o cluster com menor média (baixa confiança)
    low_confidence_component = np.argmin(gmm.means_)
    low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]

    return low_confidence_scores

def draw_score_histogram(c_class_scores, low_confidence_indices, save_path, epoch, c, threshold ):
    
    # Valor do ponto de corte nos scores (máximo dos scores das amostras de baixa confiança)
    # Tratamento para array vazio:
    if len(low_confidence_indices) > 0:
        score_cutoff = np.max(c_class_scores[low_confidence_indices])
    else:
        print(f"Aviso: Nenhuma amostra com confiança > {threshold} encontrada.")
        score_cutoff = np.min(c_class_scores)  # valor alternativo seguro

    # Dentro do seu for c in range(num_classes):
    # import pdb; pdb.set_trace()
    # ... após o fit do GMM
    plt.figure()
    plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
    # Adiciona linha vertical que separa as duas classes com base no GMM
    plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
    # plt.legend()
    #plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
    plt.title(f'Classe {c} - Histograma de scores (época {epoch} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
    plt.xlabel('Score')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.tight_layout()
    # Criar pasta, se necessário
    # save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
    plt.savefig(save_path)
    plt.close()




@HOOKS.register_module()
class MyHookCurrIouFilterPredGT_Class_Relabel(Hook):

    def __init__(self, reload_dataset=False, relabel_conf=0.95, double_thr = 2, filter_conf=0.8, filter_warmup=0, iou_assigner=0.5, low_quality=False, filter_thr=0.7, numGMM=2, filter_type = 'pred', group = False, selcand='max', overlap_filter_epochs=5, overlap_iou_thr=0.0):
        """
        Hook personalizado para relabeling de amostras.

        Args:
            reload_dataset (bool): Indica se o dataset deve ser recarregado a cada época.
            my_value (int): Um valor de exemplo para demonstrar parâmetros personalizados.
        """
        self.reload_dataset = reload_dataset
        self.relabel_conf = relabel_conf
        self.double_thr = double_thr
        self.filter_conf = filter_conf
        self.filter_warmup = filter_warmup
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        self.filter_thr = filter_thr
        self.numGMM = numGMM
        self.filter_type = filter_type
        self.group = group
        self.selcand = selcand
        # In the first epochs, ignore GT instances that overlap any other GT in the same image.
        # This is useful for very noisy data where co-occurrences (e.g., person on horse/sofa) can confuse early training.
        self.overlap_filter_epochs = overlap_filter_epochs
        self.overlap_iou_thr = overlap_iou_thr

    def before_train_epoch(self, runner, *args, **kwargs):
        
        """
        Atualiza os labels de classe (sem alterar os bounding boxes) a cada 100 iterações.
        """
        # import pdb; pdb.set_trace()
        #if (runner.epoch + 1) % 1 != 0:  
        if (runner.epoch + 1) >0 :  
        # if (runner.epoch + 1) >=5 :  

            print(f"[EPOCH]- runner.epoch: {runner.epoch}")
            print(f"[DEBUG] Antes do hook - iter: {runner.iter}")
            dataloader = runner.train_loop.dataloader
            dataset = dataloader.dataset

            #reload_dataset = True
            # my_value = 3
            #reload_dataset = getattr(runner.cfg, 'reload_dataset', False)  
            #my_value =  getattr(runner.cfg, 'my_value', 10)  
            reload_dataset = self.reload_dataset
            relabel_conf = self.relabel_conf
            
            # import pdb; pdb.set_trace()
            if reload_dataset:
                runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

                runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

            # Desencapsular RepeatDataset até o ConcatDataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset

            if not hasattr(dataset, 'datasets'):
                raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

            datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

            assigner = MaxIoUAssigner(
                pos_iou_thr=self.iou_assigner,
                neg_iou_thr=self.iou_assigner,
                min_pos_iou=self.iou_assigner,
                match_low_quality=self.low_quality
            )

            # Criar um dicionário para acesso rápido ao índice das imagens no dataset
            dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                            for sub_dataset_idx, sub_dataset in enumerate(datasets)
                            if hasattr(sub_dataset, 'data_list')
                            for data_idx, data_info in enumerate(sub_dataset.data_list)}
            
            # Dicionário para armazenar os scores médios por GT sem assumir sequência
            allbb_preds_map = defaultdict(dict)
            all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
            all_pred_instances_map = defaultdict(dict)
            # all_inputs_map = defaultdict(dict)

            num_classes = 20
            all_classes_preds_map=[]
            all_classes_gt_idx_map=[]
            for i in range(num_classes):
                all_classes_preds_map.append(defaultdict(list))
                all_classes_gt_idx_map.append(defaultdict(list))
            

            count_gts = 0
            sanity_count = 0
            global_index_counter = 0

            
            
            # Processar o batch de dados
            for batch_idx, data_batch in enumerate(dataloader):
                with torch.no_grad():
                    #data = runner.model.data_preprocessor(data_batch, False)
                    data = runner.model.data_preprocessor(data_batch, True)
                    
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                    
                    #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                    predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
                
                    #temp_filipe

                for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
                # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                    img_path = data_sample.img_path
                    
                    if img_path not in dataset_img_map:
                        print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                        continue

                    # ------------------------------------------------------------
                    # (NEW) Early-epoch overlap filtering on GT boxes
                    # ------------------------------------------------------------
                    if (runner.epoch + 1) <= self.overlap_filter_epochs:
                        sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                        sub_dataset = datasets[sub_dataset_idx]

                        # Indices in the underlying data_list that are currently NOT ignored
                        valid_instance_indices = [
                            idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                            if inst.get('ignore_flag', 0) == 0
                        ]

                        # Only proceed if we have at least 2 valid GTs
                        if len(valid_instance_indices) >= 2:
                            # Build GT boxes tensor in the same order as valid_instance_indices
                            gt_boxes = []
                            for vidx in valid_instance_indices:
                                b = sub_dataset.data_list[dataset_data_idx]['instances'][vidx]['bbox']
                                gt_boxes.append(b)
                            # IoU filtering does not require model predictions; keep it on CPU to avoid dependency on pred_instances
                            # (pred_instances is defined later in the loop)
                            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

                            iou_mat = pairwise_iou_xyxy(gt_boxes)
                            # ignore self-overlap
                            if iou_mat.numel() > 0:
                                iou_mat.fill_diagonal_(0.0)

                            overlapped_mask = (iou_mat > float(self.overlap_iou_thr)).any(dim=1)
                            overlapped_local = overlapped_mask.nonzero(as_tuple=True)[0].tolist()

                            if len(overlapped_local) > 0:
                                for local_j in overlapped_local:
                                    inst_idx = valid_instance_indices[local_j]
                                    sub_dataset.data_list[dataset_data_idx]['instances'][inst_idx]['ignore_flag'] = 1

                                print(
                                    f"[OVERLAP-FILTER] ep={runner.epoch + 1} img={os.path.basename(img_path)}: "
                                    f"ignored {len(overlapped_local)}/{len(valid_instance_indices)} GTs (IoU>{self.overlap_iou_thr})."
                                )

                    # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                    pred_instances = predictions_pred[i].pred_instances

                    # Pegar bounding boxes preditas
                    pred_instances.priors = pred_instances.pop('bboxes')

                    # Obter os ground truths
                    gt_instances = data_sample.gt_instances
                    gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que os GTs estão no mesmo dispositivo das predições
                    gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                    gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que as predições também estão no mesmo dispositivo
                    pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                    pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                    pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                    pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                    all_pred_instances_map[img_path] = pred_instances
                    

                    assign_result = assigner.assign(pred_instances, gt_instances)

                    updated_labels = gt_labels.clone()

                    # Criar lista de GTs para esta imagem
                    all_gt_idx_map[img_path] = []
                    count_gts += assign_result.num_gts

                    
                    
                    
                    # sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    # sub_dataset = datasets[sub_dataset_idx]
                    # import pdb; pdb.set_trace()
                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/sub_{os.path.basename(img_path)}')  
                    # desenhar_bboxesv3(img_path_real,inputs[i], gt_instances, save_path=f'debug_imgs/gt_{os.path.basename(img_path)}')  
                    
                    # if  len(data_samples[i].ignored_instances.labels)>0:
                    #     import pdb; pdb.set_trace()
                    

                    # **Vetorização do processo de associação**
                    for gt_idx in range(assign_result.num_gts):
                        # Skip GTs already marked ignored in the dataset instances (early epochs)
                        if (runner.epoch + 1) <= self.overlap_filter_epochs:
                            sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                            sub_dataset = datasets[sub_dataset_idx]
                            inst_all = sub_dataset.data_list[dataset_data_idx]['instances']
                            if gt_idx < len(inst_all) and inst_all[gt_idx].get('ignore_flag', 0) == 1:
                                continue
                        associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                        sanity_count += 1
                        
                        if associated_preds.numel() == 0:
                            continue 

                        # Calcular a média dos scores das bounding boxes associadas a este `gt_idx`
                        # mean_score = pred_instances.scores[associated_preds].mean().item()
                        

                        # max_score = pred_instances.scores[associated_preds].max().item()
                        given_max_score = pred_instances.scores[associated_preds].max().item()
                        # max_logit = pred_instances.logits[associated_preds].max().item()
                        # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                        logits_associated = pred_instances.logits[associated_preds] 
                        myscores = torch.softmax(logits_associated ,dim=-1)

                        
                        # import pdb;pdb.set_trace()
                        # all_pred_instances_map[img_path]['priors']=pred_instances.priors[associated_preds]
                        # all_pred_instances_map[img_path]['scores']=pred_instances.scores[associated_preds]
                        # all_pred_instances_map[img_path]['labels']=pred_instances.labels[associated_preds]
                        # import pdb;pdb.set_trace()
                        # myscores_pred =  myscores.max(dim=1).values.item()

                        # quero pegar o logit_gt da amostra que tem maior logit_pred
                        # import pdb; pdb.set_trace()
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        max_score_val = myscores.max()
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()
                        myious = assign_result.max_overlaps[associated_preds]
                        max_iou_idx = torch.argmax(myious)
                        amostra_id_iou, classe_id_iou = divmod(max_iou_idx.item(), myscores.size(1))
                        score_gt_max_iou = myscores_gt[amostra_id_iou].cpu().item()
                        score_pred_max_iou = myscores[amostra_id_iou, classe_id].cpu().item()

                        # Compute APS
                        row_scores = myscores[amostra_id]  # pega apenas a linha
                        p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                        S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        # p_y = myscores[amostra_id, classe_id].item()
                        # S_APS_score = -myscores[myscores >= p_y].sum().item()


                            

                        # import pdb; pdb.set_trace()
                        # allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score
                         
                        max_score_idx_local = torch.argmax(myscores_gt)
                        #----> remover depois do debug
                        #max_logit_idx_gt = associated_preds[max_score_idx_local]
                        max_logit_idx_gt = associated_preds[amostra_id]
                        logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()
                        


        
                        gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]

                        #---> logit com maior valor [OFICIAL]-voltar depois do debug
                        # max_logit = gt_logits.max().item()
                        # max_logit_idx_local = torch.argmax(gt_logits)
                        # max_logit_idx = associated_preds[max_logit_idx_local]

                        


                        #logit com maior IoU
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()

                        # Pegar idx local do maior IoU
                        max_iou_values = assign_result.max_overlaps[associated_preds]
                        max_iou_idx_local = torch.argmax(max_iou_values)
                        max_iou_idx = associated_preds[max_iou_idx_local]

                        # Vetor contendo ambos (pode repetir se for o mesmo)
                        important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                        important_associated_ids = torch.unique(important_associated_ids)

                        # import pdb; pdb.set_trace()


                        # gt_logits = [gt_logit[updated_labels[gt_idx]] for gt_logit in logits_associated]
                        
                        # Armazenar `gt_idx` na ordem correta
                        all_gt_idx_map[img_path].append(gt_idx)

                        # Armazenar `mean_score` corretamente
                        #---->[original]
                        # allbb_preds_map[img_path][gt_idx] = {'pred':max_logit, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #---->[temporario-filipe-debug] - remover depois do debug

                        if self.selcand == 'max':
                            allbb_preds_map[img_path][gt_idx] = {'pred': score_gt_max_pred, 'logit':logit_gt_max_pred , 
                                                             'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                             'aps':S_APS_score,
                                                              'max_pred':max_score_val.cpu().item(), 'pred_label': classe_id,
                                                                'filtered':False}
                        elif self.selcand == 'iou':
                            allbb_preds_map[img_path][gt_idx] = {'pred': score_gt_max_iou, 'logit':logit_gt_max_pred , 
                                                             'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                             'aps':S_APS_score,
                                                              'max_pred':score_pred_max_iou, 'pred_label': classe_id_iou,
                                                                'filtered':False}
                        
                        else:
                            print("Warning: selcand deve ser 'max' ou 'iou'. Usando 'max' por padrão.")
                        global_index_counter += 1



                        

                        #-->original   
                        #confident_preds =  associated_preds[pred_instances.scores[associated_preds] > relabel_conf]
                        #-->temporario-filipe-debug - remover depois
                        # import pdb; pdb.set_trace()

                        

                        confident_preds =  associated_preds[myscores.max(dim=1).values> relabel_conf]
                        

                        
                        if (batch_idx <2):

                            if confident_preds.numel() > 0:
                                # save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg'
                                # desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                pass
                            
                            
                                
                            elif len(associated_preds) > 1 and (max_score_val > 0.45) :
                                labels_group = myscores.argmax(dim=1)
                            
                                # _, qtd = Counter(labels_group.tolist()).most_common(1)[0]
                                most_common_label, qtd = Counter(labels_group.tolist()).most_common(1)[0]

                                scores_most_common = myscores[:,most_common_label]
                                confident_most_common =  associated_preds[scores_most_common> 0.45]
                                
                                # import pdb; pdb.set_trace()
                                # verifica se a quantidade é maior que 50% do total
                                if qtd > (len(associated_preds) / 2)  and len(confident_most_common) > 2:
                                    save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_grouped.jpg'
                                    #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                    desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, associated_preds,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  

                            else:
                                                                
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg'
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg')  
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, associated_preds, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                            

                        
                        
                                  

                        if confident_preds.numel() > 0:
                            
                            #---> original
                            # pred_labels_confident = pred_instances.labels[confident_preds]
                            #---> temporario-filipe-debug - remover depois do debug
                            #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                            pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                            most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        elif self.group and len(associated_preds) > 1 and (max_score_val > 0.45):
                            labels_group = myscores.argmax(dim=1)
                            most_common_label, qtd = Counter(labels_group.tolist()).most_common(1)[0]

                            scores_most_common = myscores[:,most_common_label]
                            confident_most_common =  associated_preds[scores_most_common> 0.45]
                            # import pdb; pdb.set_trace()

                            # verifica se a quantidade é maior que 50% do total
                            if qtd > (len(associated_preds) / 2) and len(confident_most_common) > 2:
                                
                                most_common_label = most_common_label
                            # senao verifica se tem algum elemento com score maior que o threshold
                            elif confident_preds.numel() > 0:
                                pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)
                                most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        # elif confident_preds.numel() == 0:
                        else:
                            continue  


                        updated_labels[gt_idx] = most_common_label
                        #atualiza também no mapeamento
                        allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                        #atualiza max logit-gt
                        gt_logits = logits_associated[:,most_common_label]
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                        max_logit = gt_logits.max().item()

                        ###-->debug -remover depois
                        # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        # max_score_val = myscores.max()
                        # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                        # Compute APS
                        # p_y = myscores[amostra_id, classe_id].item()
                        # S_APS_score = -myscores[myscores >= p_y].sum().item()
                        # row_scores = myscores[amostra_id]  # pega apenas a linha
                        # p_y = row_scores[classe_id].item()
                        # S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        row_scores = myscores[amostra_id]  # pega apenas a linha
                        p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                        S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        # if S_APS_score < -1:
                        #     print(S_APS_score)
                        #     print("a")
                        #     import pdb; pdb.set_trace()
                        #     print("ok")
                        allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score

                        #
                        #-->[original]
                        #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                        #-->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                        allbb_preds_map[img_path][gt_idx]['filtered'] = True



                        
                        

                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    inst_all = sub_dataset.data_list[dataset_data_idx]['instances']
                    N_upd = min(assign_result.num_gts, len(inst_all), int(updated_labels.numel()))
                    for gt_idx in range(N_upd):
                        # only update active (non-ignored) instances
                        if inst_all[gt_idx].get('ignore_flag', 0) == 1:
                            continue
                        inst_all[gt_idx]['bbox_label'] = int(updated_labels[gt_idx].item())
            

            # só faz filter depois do warmup
            if (runner.epoch + 1) >= self.filter_warmup:

                # filtering
                #Unir todas as probabilidades das imagens e treinar o GMM**
                # all_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values()
                # ]).reshape(-1, 1)
                # import pdb; pdb.set_trace()
                print("[DEBUG1]: Entrou no filtro")
                all_classes_low_confidence_scores_global_idx = []

                for c in range(num_classes):
                    # c_class_scores = np.array([
                    #     score['pred'] for img_scores in allbb_preds_map.values()
                    #     for score in img_scores.values() if score['gt_label'] == c
                    # ]).reshape(-1, 1)

                    # if runner.epoch + 1 >1:
                    #     import pdb; pdb.set_trace()

                    scores = np.array([])
                    # img_indexes = np.array([])
                    scores_dict = {'pred': np.array([]),
                                   'logit': np.array([]),
                                   'aps': np.array([])}
                    low_conf_dict = {'pred': np.array([]),
                                   'logit': np.array([]),
                                   'aps': np.array([])}
                    img_paths = np.array([], dtype=object)
                    c_global_indexes = np.array([])

                    for img_path, img_info in allbb_preds_map.items():
                        #for pred, gt_label, index  in img_info.values():
                        for temp_gt_idx, values in img_info.items():
                            #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                            #if gt_label== c:
                            if values['gt_label'] == c:
                                scores_dict['pred'] = np.append(scores_dict['pred'], values['pred'])
                                scores_dict['logit'] = np.append(scores_dict['logit'], values['logit'])
                                scores_dict['aps'] = np.append(scores_dict['aps'], values['aps'])

                                # calculando como pred apenas para ver depois se len(scores) == 0
                                scores = np.append(scores, values['pred'])

                                
                                # if self.filter_type == 'pred':
                                #     scores = np.append(scores, values['pred'])
                                # elif self.filter_type == 'logit':
                                #     scores = np.append(scores, values['logit'])
                                # elif self.filter_type == 'aps':
                                #     scores = np.append(scores, values['aps'])

                                c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                                if img_path not in img_paths:
                                    img_paths = np.append(img_paths, img_path)
                    
                    if len(scores) == 0:
                        print(f"Aviso: Nenhuma amostra encontrada para a classe {c}.")
                        continue

                    
                    

                    # low_confidence_scores = calculate_gmm(c_class_scores, n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                                
                    low_conf_dict['pred'] = calculate_gmm(scores_dict['pred'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    low_conf_dict['logit'] = calculate_gmm(scores_dict['logit'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    low_conf_dict['aps'] = calculate_gmm(scores_dict['aps'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)

                    if self.filter_type == 'pred':
                        scores = scores_dict['pred']
                        low_confidence_scores = low_conf_dict['pred']
                    elif self.filter_type == 'logit':
                        scores = scores_dict['logit']
                        low_confidence_scores = low_conf_dict['logit']
                    elif self.filter_type == 'aps':
                        scores = scores_dict['aps']
                        low_confidence_scores = low_conf_dict['aps']


                    # print("[DEBUG1.5]: INICIANDO GMM")

                    c_class_scores = c_class_scores = scores.reshape(-1, 1)


                    #gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    # gmm = GaussianMixture(n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    # gmm.fit(c_class_scores)

                    # # Identificar o cluster com menor média (baixa confiança)
                    # low_confidence_component = np.argmin(gmm.means_)
                    # low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]


                    threshold = self.filter_conf
                
                    low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                    all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                    

                    

                    # draw pred hist
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_pred_ep{runner.epoch + 1}.png"
                    low_confidence_indices_temp = np.where(low_conf_dict['pred'] > threshold)[0]
                    draw_score_histogram(scores_dict['pred'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_conf)

                    # draw logit hist
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_logit_ep{runner.epoch + 1}.png"
                    low_confidence_indices_temp = np.where(low_conf_dict['logit'] > threshold)[0]
                    draw_score_histogram(scores_dict['logit'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_conf)

                    # draw aps hist
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_aps_ep{runner.epoch + 1}.png"
                    low_confidence_indices_temp = np.where(low_conf_dict['aps'] > threshold)[0]
                    draw_score_histogram(scores_dict['aps'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_conf)

                    # Dentro do seu for c in range(num_classes):
                    # import pdb; pdb.set_trace()
                    # ... após o fit do GMM
                    # plt.figure()
                    # plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                    # # Adiciona linha vertical que separa as duas classes com base no GMM
                    # plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                    # # plt.legend()
                    # plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                    # plt.xlabel('Score')
                    # plt.ylabel('Frequência')
                    # plt.grid(True)
                    # plt.tight_layout()
                    # # Criar pasta, se necessário
                    # save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    # #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                    # plt.savefig(save_path)
                    # plt.close()
                   

                # print("[DEBUG2]: SAIU do GMM")
                # if len(all_scores) > 0:
                    

                # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
                # index = 0
                # import pdb; pdb.set_trace()
                my_counter = 0
                for img_path, gt_scores in allbb_preds_map.items():
                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    

                    valid_instance_indices = [
                        idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                        if inst['ignore_flag'] == 0
                    ]

                    # **Mapear GTs reais na ordem correta**
                    gt_idx_list = all_gt_idx_map[img_path]  

                    # import pdb; pdb.set_trace()

                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path)}')  
                    # save_path = f'debug_imgs/epoch{runner.epoch+1}_{os.path.basename(img_path)}'
                    # import pdb; pdb.set_trace()
                    

                    for gt_idx in gt_idx_list:
                        related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']

                        if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                            if my_counter < 5:
                                import shutil
                                base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                                possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg", "_grouped.jpg"]
                                for suffix in possible_suffixes:
                                    base_debug_path = base_prefix + suffix
                                    if os.path.exists(base_debug_path):
                                        my_counter += 1
                                        filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                                        shutil.copy(base_debug_path, filtered_debug_path)
                                        print(f"[INFO] Cópia criada: {filtered_debug_path}")
                                        break
                            # Robust mapping for valid_idx (fix IndexError/GT mapping)
                            # Map gt_idx -> data_list instance index robustly.
                            # Prefer direct mapping by gt_idx (VOC order), but fall back to position within gt_idx_list if needed.
                            inst_all = sub_dataset.data_list[dataset_data_idx]['instances']
                            valid_idx = None
                            if gt_idx < len(inst_all):
                                valid_idx = gt_idx
                            else:
                                try:
                                    pos = gt_idx_list.index(gt_idx)
                                    if pos < len(valid_instance_indices):
                                        valid_idx = valid_instance_indices[pos]
                                except ValueError:
                                    valid_idx = None
                            if valid_idx is None:
                                print(
                                    f"[GMM-FILTER][WARN] Could not map gt_idx={gt_idx} to instance index "
                                    f"(len(inst_all)={len(inst_all)}, len(valid)={len(valid_instance_indices)}) img={os.path.basename(img_path)}"
                                )
                                continue
                            # self.double_thr
                            if allbb_preds_map[img_path][gt_idx]['max_pred'] >= self.double_thr:
                                # update
                                sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = allbb_preds_map[img_path][gt_idx]['pred_label']
                            else:
                                # filtra
                                sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                        
                    
                                
                                
                            
                        # index += 1
                        
        # --- Remove all remaining occurrences of valid_instance_indices[gt_idx] ---
        # (Handled in above blocks. If any remain, replace with robust mapping below.)

            print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")


@HOOKS.register_module()
class MyHookCurrIntoFilterPredGT_Class_Relabel(Hook):

    def __init__(self, reload_dataset=False, relabel_conf=0.95, double_thr = 2, filter_conf=0.8, filter_warmup=0, iou_assigner=0.5, low_quality=False, filter_thr=0.7, numGMM=2, filter_type = 'pred', group = False, selcand='max', overlap_filter_epochs=5, overlap_iou_thr=0.0):
        """
        Hook personalizado para relabeling de amostras.

        Args:
            reload_dataset (bool): Indica se o dataset deve ser recarregado a cada época.
            my_value (int): Um valor de exemplo para demonstrar parâmetros personalizados.
        """
        self.reload_dataset = reload_dataset
        self.relabel_conf = relabel_conf
        self.double_thr = double_thr
        self.filter_conf = filter_conf
        self.filter_warmup = filter_warmup
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        self.filter_thr = filter_thr
        self.numGMM = numGMM
        self.filter_type = filter_type
        self.group = group
        self.selcand = selcand
        # In the first epochs, ignore GT instances that overlap any other GT in the same image.
        # This is useful for very noisy data where co-occurrences (e.g., person on horse/sofa) can confuse early training.
        self.overlap_filter_epochs = overlap_filter_epochs
        self.overlap_iou_thr = overlap_iou_thr

    def before_train_epoch(self, runner, *args, **kwargs):
        
        """
        Atualiza os labels de classe (sem alterar os bounding boxes) a cada 100 iterações.
        """
        # import pdb; pdb.set_trace()
        #if (runner.epoch + 1) % 1 != 0:  
        if (runner.epoch + 1) >0 :  
        # if (runner.epoch + 1) >=5 :  

            print(f"[EPOCH]- runner.epoch: {runner.epoch}")
            print(f"[DEBUG] Antes do hook - iter: {runner.iter}")
            dataloader = runner.train_loop.dataloader
            dataset = dataloader.dataset

            #reload_dataset = True
            # my_value = 3
            #reload_dataset = getattr(runner.cfg, 'reload_dataset', False)  
            #my_value =  getattr(runner.cfg, 'my_value', 10)  
            reload_dataset = self.reload_dataset
            relabel_conf = self.relabel_conf
            
            # import pdb; pdb.set_trace()
            if reload_dataset:
                runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

                runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

            # Desencapsular RepeatDataset até o ConcatDataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset

            if not hasattr(dataset, 'datasets'):
                raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

            datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

            assigner = MaxIoUAssigner(
                pos_iou_thr=self.iou_assigner,
                neg_iou_thr=self.iou_assigner,
                min_pos_iou=self.iou_assigner,
                match_low_quality=self.low_quality
            )

            # Criar um dicionário para acesso rápido ao índice das imagens no dataset
            dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                            for sub_dataset_idx, sub_dataset in enumerate(datasets)
                            if hasattr(sub_dataset, 'data_list')
                            for data_idx, data_info in enumerate(sub_dataset.data_list)}
            
            # Dicionário para armazenar os scores médios por GT sem assumir sequência
            allbb_preds_map = defaultdict(dict)
            all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
            all_pred_instances_map = defaultdict(dict)
            # all_inputs_map = defaultdict(dict)

            num_classes = 20
            all_classes_preds_map=[]
            all_classes_gt_idx_map=[]
            for i in range(num_classes):
                all_classes_preds_map.append(defaultdict(list))
                all_classes_gt_idx_map.append(defaultdict(list))
            

            count_gts = 0
            sanity_count = 0
            global_index_counter = 0

            
            
            # Processar o batch de dados
            for batch_idx, data_batch in enumerate(dataloader):
                with torch.no_grad():
                    #data = runner.model.data_preprocessor(data_batch, False)
                    data = runner.model.data_preprocessor(data_batch, True)
                    
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                    
                    #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                    predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
                
                    #temp_filipe

                for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
                # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                    img_path = data_sample.img_path
                    
                    if img_path not in dataset_img_map:
                        print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                        continue

                    

                    # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                    pred_instances = predictions_pred[i].pred_instances

                    # Pegar bounding boxes preditas
                    pred_instances.priors = pred_instances.pop('bboxes')

                    # Obter os ground truths
                    gt_instances = data_sample.gt_instances
                    gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que os GTs estão no mesmo dispositivo das predições
                    gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                    gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que as predições também estão no mesmo dispositivo
                    pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                    pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                    pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                    pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                    all_pred_instances_map[img_path] = pred_instances
                    

                    assign_result = assigner.assign(pred_instances, gt_instances)

                    updated_labels = gt_labels.clone()

                    # Track GT indices filtered by containment in this image, to skip later processing safely
                    contain_filtered_gt = set()

                    # ------------------------------------------------------------
                    # (NEW) Early-epoch containment-based filtering using
                    # prediction-vs-annotation disagreement (robust indexing)
                    # ------------------------------------------------------------
                    to_ignore_gt = set()
                    if ((runner.epoch + 1) <= self.overlap_filter_epochs) and  ((runner.epoch + 1) >= self.filter_warmup) and (float(self.overlap_iou_thr) > 0.0):
                        sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                        sub_dataset = datasets[sub_dataset_idx]

                        # IMPORTANT:
                        # - `gt_idx` used by `gt_instances/assign_result` typically follows the dataset instance order.
                        # - Do NOT remap indices through a "non-ignored" list for mapping; that can shrink the list and cause IndexError.
                        inst_all = sub_dataset.data_list[dataset_data_idx]['instances']
                        active_gt_indices = [i for i, inst in enumerate(inst_all) if inst.get('ignore_flag', 0) == 0]

                        # Need at least 2 GTs for containment relations
                        if assign_result.num_gts >= 2 and len(active_gt_indices) >= 2:
                            # GT boxes/labels aligned with assign_result indexing
                            gt_boxes = gt_instances.bboxes.tensor
                            ann_labels_t = gt_instances.labels

                            # Index-based mapping: assume GT order matches the order of non-ignored instances in data_list.
                            # We will use direct indexing by gt_idx for ignore_flag.

                            # For each GT, infer a predicted label from its associated predictions
                            # If there are no associated preds, set pred label to -1 (skip filtering for that GT)
                            pred_labels_t = torch.full((assign_result.num_gts,), -1, dtype=torch.long, device=gt_boxes.device)
                            for gt_idx in range(assign_result.num_gts):
                                associated = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                                if associated.numel() == 0:
                                    continue
                                logits_assoc = pred_instances.logits[associated]
                                scores_assoc = torch.softmax(logits_assoc, dim=-1)
                                best_row = torch.argmax(scores_assoc.max(dim=1).values)
                                pred_labels_t[gt_idx] = int(torch.argmax(scores_assoc[best_row]).item())

                            # Containment ratio matrix: fraction of box_i inside box_j
                            contain_mat = pairwise_containment_ratio_xyxy(gt_boxes)
                            if contain_mat.numel() > 0:
                                contain_mat.fill_diagonal_(0.0)

                            thr = float(self.overlap_iou_thr)
                            areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0))

                            # For pairs where a smaller box is mostly inside a larger one, apply per-box rule:
                            # keep if pred==ann; otherwise ignore.
                            
                            for i_gt in active_gt_indices:
                                for j_gt in active_gt_indices:
                                    if i_gt == j_gt:
                                        continue
                                    if areas[i_gt] > areas[j_gt]:
                                        continue
                                    if float(contain_mat[i_gt, j_gt].item()) >= thr:
                                        pi = int(pred_labels_t[i_gt].item())
                                        pj = int(pred_labels_t[j_gt].item())
                                        ai = int(ann_labels_t[i_gt].item())
                                        aj = int(ann_labels_t[j_gt].item())
                                        if pi != -1 and pi != ai:
                                            to_ignore_gt.add(i_gt)
                                        if pj != -1 and pj != aj:
                                            to_ignore_gt.add(j_gt)

                            # Apply ignore_flag directly by gt_idx (dataset instance order)
                            applied = 0
                            for gt_idx in sorted(to_ignore_gt):
                                # if gt_idx >= len(inst_all):
                                    # print(
                                    #     f"[CONTAIN-FILTER][WARN] gt_idx={gt_idx} out of range for instances "
                                    #     f"(len={len(inst_all)}) img={os.path.basename(img_path)}"
                                    # )
                                    # continue
                                # inst_all[gt_idx]['ignore_flag'] = 1
                                sub_dataset.data_list[dataset_data_idx]['instances'][gt_idx]['ignore_flag'] = 1
                                
                                contain_filtered_gt.add(gt_idx)
                                applied += 1

                            if applied > 0:
                                print(
                                    f"[CONTAIN-FILTER] ep={runner.epoch + 1} img={os.path.basename(img_path)}: "
                                    f"ignored {applied}/{assign_result.num_gts} GTs (contain>={thr})."
                                )


                    # Criar lista de GTs para esta imagem
                    all_gt_idx_map[img_path] = []
                    count_gts += assign_result.num_gts

                    
                    
                    
                    # sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    # sub_dataset = datasets[sub_dataset_idx]
                    # import pdb; pdb.set_trace()
                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/sub_{os.path.basename(img_path)}')  
                    # desenhar_bboxesv3(img_path_real,inputs[i], gt_instances, save_path=f'debug_imgs/gt_{os.path.basename(img_path)}')  
                    
                    # if  len(data_samples[i].ignored_instances.labels)>0:
                    #     import pdb; pdb.set_trace()
                    

                    # **Vetorização do processo de associação**
                    for gt_idx in range(assign_result.num_gts):

                        #filipe2
                        if gt_idx  in to_ignore_gt:
                            # import pdb; pdb.set_trace()
                            continue
                        #endfilipe2

                        # Skip GTs filtered by containment in early epochs (robust, no index guessing)
                        # if (runner.epoch + 1) <= self.overlap_filter_epochs and gt_idx in contain_filtered_gt:
                        #     continue
                        associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                        sanity_count += 1
                        
                        if associated_preds.numel() == 0:
                            continue 

                        # Calcular a média dos scores das bounding boxes associadas a este `gt_idx`
                        # mean_score = pred_instances.scores[associated_preds].mean().item()
                        

                        # max_score = pred_instances.scores[associated_preds].max().item()
                        given_max_score = pred_instances.scores[associated_preds].max().item()
                        # max_logit = pred_instances.logits[associated_preds].max().item()
                        # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                        logits_associated = pred_instances.logits[associated_preds] 
                        myscores = torch.softmax(logits_associated ,dim=-1)

                        
                        # import pdb;pdb.set_trace()
                        # all_pred_instances_map[img_path]['priors']=pred_instances.priors[associated_preds]
                        # all_pred_instances_map[img_path]['scores']=pred_instances.scores[associated_preds]
                        # all_pred_instances_map[img_path]['labels']=pred_instances.labels[associated_preds]
                        # import pdb;pdb.set_trace()
                        # myscores_pred =  myscores.max(dim=1).values.item()

                        # quero pegar o logit_gt da amostra que tem maior logit_pred
                        # import pdb; pdb.set_trace()
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        max_score_val = myscores.max()
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()
                        myious = assign_result.max_overlaps[associated_preds]
                        max_iou_idx = torch.argmax(myious)
                        amostra_id_iou, classe_id_iou = divmod(max_iou_idx.item(), myscores.size(1))
                        score_gt_max_iou = myscores_gt[amostra_id_iou].cpu().item()
                        score_pred_max_iou = myscores[amostra_id_iou, classe_id].cpu().item()

                        # Compute APS
                        row_scores = myscores[amostra_id]  # pega apenas a linha
                        p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                        S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        # p_y = myscores[amostra_id, classe_id].item()
                        # S_APS_score = -myscores[myscores >= p_y].sum().item()


                            

                        # import pdb; pdb.set_trace()
                        # allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score
                         
                        max_score_idx_local = torch.argmax(myscores_gt)
                        #----> remover depois do debug
                        #max_logit_idx_gt = associated_preds[max_score_idx_local]
                        max_logit_idx_gt = associated_preds[amostra_id]
                        logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()
                        


        
                        gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]

                        #---> logit com maior valor [OFICIAL]-voltar depois do debug
                        # max_logit = gt_logits.max().item()
                        # max_logit_idx_local = torch.argmax(gt_logits)
                        # max_logit_idx = associated_preds[max_logit_idx_local]

                        


                        #logit com maior IoU
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()

                        # Pegar idx local do maior IoU
                        max_iou_values = assign_result.max_overlaps[associated_preds]
                        max_iou_idx_local = torch.argmax(max_iou_values)
                        max_iou_idx = associated_preds[max_iou_idx_local]

                        # Vetor contendo ambos (pode repetir se for o mesmo)
                        important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                        important_associated_ids = torch.unique(important_associated_ids)

                        # import pdb; pdb.set_trace()


                        # gt_logits = [gt_logit[updated_labels[gt_idx]] for gt_logit in logits_associated]
                        
                        # Armazenar `gt_idx` na ordem correta
                        all_gt_idx_map[img_path].append(gt_idx)

                        # Armazenar `mean_score` corretamente
                        #---->[original]
                        # allbb_preds_map[img_path][gt_idx] = {'pred':max_logit, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #---->[temporario-filipe-debug] - remover depois do debug

                        if self.selcand == 'max':
                            allbb_preds_map[img_path][gt_idx] = {'pred': score_gt_max_pred, 'logit':logit_gt_max_pred , 
                                                             'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                             'aps':S_APS_score,
                                                              'max_pred':max_score_val.cpu().item(), 'pred_label': classe_id,
                                                                'filtered':False}
                        elif self.selcand == 'iou':
                            allbb_preds_map[img_path][gt_idx] = {'pred': score_gt_max_iou, 'logit':logit_gt_max_pred , 
                                                             'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                             'aps':S_APS_score,
                                                              'max_pred':score_pred_max_iou, 'pred_label': classe_id_iou,
                                                                'filtered':False}
                        
                        else:
                            print("Warning: selcand deve ser 'max' ou 'iou'. Usando 'max' por padrão.")
                        global_index_counter += 1



                        

                        #-->original   
                        #confident_preds =  associated_preds[pred_instances.scores[associated_preds] > relabel_conf]
                        #-->temporario-filipe-debug - remover depois
                        # import pdb; pdb.set_trace()

                        

                        confident_preds =  associated_preds[myscores.max(dim=1).values> relabel_conf]
                        

                        
                        # if (batch_idx <2):

                        #     if confident_preds.numel() > 0:
                        #         # save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg'
                        #         # desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                        #         pass
                            
                            
                                
                        #     elif len(associated_preds) > 1 and (max_score_val > 0.45) :
                        #         labels_group = myscores.argmax(dim=1)
                            
                        #         # _, qtd = Counter(labels_group.tolist()).most_common(1)[0]
                        #         most_common_label, qtd = Counter(labels_group.tolist()).most_common(1)[0]

                        #         scores_most_common = myscores[:,most_common_label]
                        #         confident_most_common =  associated_preds[scores_most_common> 0.45]
                                
                        #         # import pdb; pdb.set_trace()
                        #         # verifica se a quantidade é maior que 50% do total
                        #         if qtd > (len(associated_preds) / 2)  and len(confident_most_common) > 2:
                        #             save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_grouped.jpg'
                        #             #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                        #             desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, associated_preds,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  

                        #     else:
                                                                
                        #         save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg'
                        #         #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg')  
                        #         #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                        #         desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, associated_preds, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                            

                        
                        
                                  

                        if confident_preds.numel() > 0:
                            
                            #---> original
                            # pred_labels_confident = pred_instances.labels[confident_preds]
                            #---> temporario-filipe-debug - remover depois do debug
                            #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                            pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                            most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        elif self.group and len(associated_preds) > 1 and (max_score_val > 0.45):
                            labels_group = myscores.argmax(dim=1)
                            most_common_label, qtd = Counter(labels_group.tolist()).most_common(1)[0]

                            scores_most_common = myscores[:,most_common_label]
                            confident_most_common =  associated_preds[scores_most_common> 0.45]
                            # import pdb; pdb.set_trace()

                            # verifica se a quantidade é maior que 50% do total
                            if qtd > (len(associated_preds) / 2) and len(confident_most_common) > 2:
                                
                                most_common_label = most_common_label
                            # senao verifica se tem algum elemento com score maior que o threshold
                            elif confident_preds.numel() > 0:
                                pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)
                                most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        # elif confident_preds.numel() == 0:
                        else:
                            continue  


                        updated_labels[gt_idx] = most_common_label
                        #atualiza também no mapeamento
                        allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                        #atualiza max logit-gt
                        gt_logits = logits_associated[:,most_common_label]
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                        max_logit = gt_logits.max().item()

                        ###-->debug -remover depois
                        # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        # max_score_val = myscores.max()
                        # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                        # Compute APS
                        # p_y = myscores[amostra_id, classe_id].item()
                        # S_APS_score = -myscores[myscores >= p_y].sum().item()
                        # row_scores = myscores[amostra_id]  # pega apenas a linha
                        # p_y = row_scores[classe_id].item()
                        # S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        row_scores = myscores[amostra_id]  # pega apenas a linha
                        p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                        S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        # if S_APS_score < -1:
                        #     print(S_APS_score)
                        #     print("a")
                        #     import pdb; pdb.set_trace()
                        #     print("ok")
                        allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score

                        #
                        #-->[original]
                        #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                        #-->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                        allbb_preds_map[img_path][gt_idx]['filtered'] = True



                        
                        

                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                    valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                    # Atualizar os labels corretamente usando o mapeamento
                    for gt_idx, valid_idx in enumerate(valid_instance_indices):
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                        # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
            

            # só faz filter depois do warmup
            if (runner.epoch + 1) >= self.filter_warmup:
            # if (runner.epoch + 1) >= 0:

                # filtering
                #Unir todas as probabilidades das imagens e treinar o GMM**
                # all_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values()
                # ]).reshape(-1, 1)
                # import pdb; pdb.set_trace()
                print("[DEBUG1]: Entrou no filtro")
                all_classes_low_confidence_scores_global_idx = []

                for c in range(num_classes):
                    # c_class_scores = np.array([
                    #     score['pred'] for img_scores in allbb_preds_map.values()
                    #     for score in img_scores.values() if score['gt_label'] == c
                    # ]).reshape(-1, 1)

                    # if runner.epoch + 1 >1:
                    #     import pdb; pdb.set_trace()

                    scores = np.array([])
                    # img_indexes = np.array([])
                    scores_dict = {'pred': np.array([]),
                                   'logit': np.array([]),
                                   'aps': np.array([])}
                    low_conf_dict = {'pred': np.array([]),
                                   'logit': np.array([]),
                                   'aps': np.array([])}
                    img_paths = np.array([], dtype=object)
                    c_global_indexes = np.array([])

                    for img_path, img_info in allbb_preds_map.items():
                        #for pred, gt_label, index  in img_info.values():
                        for temp_gt_idx, values in img_info.items():
                            #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                            #if gt_label== c:
                            if values['gt_label'] == c:
                                scores_dict['pred'] = np.append(scores_dict['pred'], values['pred'])
                                scores_dict['logit'] = np.append(scores_dict['logit'], values['logit'])
                                scores_dict['aps'] = np.append(scores_dict['aps'], values['aps'])

                                # calculando como pred apenas para ver depois se len(scores) == 0
                                scores = np.append(scores, values['pred'])

                                
                                # if self.filter_type == 'pred':
                                #     scores = np.append(scores, values['pred'])
                                # elif self.filter_type == 'logit':
                                #     scores = np.append(scores, values['logit'])
                                # elif self.filter_type == 'aps':
                                #     scores = np.append(scores, values['aps'])

                                c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                                if img_path not in img_paths:
                                    img_paths = np.append(img_paths, img_path)
                    
                    if len(scores) == 0:
                        print(f"Aviso: Nenhuma amostra encontrada para a classe {c}.")
                        continue

                    
                    

                    # low_confidence_scores = calculate_gmm(c_class_scores, n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                                
                    low_conf_dict['pred'] = calculate_gmm(scores_dict['pred'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    low_conf_dict['logit'] = calculate_gmm(scores_dict['logit'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    low_conf_dict['aps'] = calculate_gmm(scores_dict['aps'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)

                    if self.filter_type == 'pred':
                        scores = scores_dict['pred']
                        low_confidence_scores = low_conf_dict['pred']
                    elif self.filter_type == 'logit':
                        scores = scores_dict['logit']
                        low_confidence_scores = low_conf_dict['logit']
                    elif self.filter_type == 'aps':
                        scores = scores_dict['aps']
                        low_confidence_scores = low_conf_dict['aps']


                    # print("[DEBUG1.5]: INICIANDO GMM")

                    c_class_scores = c_class_scores = scores.reshape(-1, 1)


                    #gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    # gmm = GaussianMixture(n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    # gmm.fit(c_class_scores)

                    # # Identificar o cluster com menor média (baixa confiança)
                    # low_confidence_component = np.argmin(gmm.means_)
                    # low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]


                    threshold = self.filter_conf
                
                    low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                    all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                    

                    

                    # draw pred hist
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_pred_ep{runner.epoch + 1}.png"
                    low_confidence_indices_temp = np.where(low_conf_dict['pred'] > threshold)[0]
                    draw_score_histogram(scores_dict['pred'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_conf)

                    # draw logit hist
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_logit_ep{runner.epoch + 1}.png"
                    low_confidence_indices_temp = np.where(low_conf_dict['logit'] > threshold)[0]
                    draw_score_histogram(scores_dict['logit'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_conf)

                    # draw aps hist
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_aps_ep{runner.epoch + 1}.png"
                    low_confidence_indices_temp = np.where(low_conf_dict['aps'] > threshold)[0]
                    draw_score_histogram(scores_dict['aps'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_conf)

                    # Dentro do seu for c in range(num_classes):
                    # import pdb; pdb.set_trace()
                    # ... após o fit do GMM
                    # plt.figure()
                    # plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                    # # Adiciona linha vertical que separa as duas classes com base no GMM
                    # plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                    # # plt.legend()
                    # plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                    # plt.xlabel('Score')
                    # plt.ylabel('Frequência')
                    # plt.grid(True)
                    # plt.tight_layout()
                    # # Criar pasta, se necessário
                    # save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    # #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                    # plt.savefig(save_path)
                    # plt.close()
                   

                # print("[DEBUG2]: SAIU do GMM")
                # if len(all_scores) > 0:
                    

                # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
                # index = 0
                # import pdb; pdb.set_trace()
                my_counter = 0
                for img_path, gt_scores in allbb_preds_map.items():
                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    

                    valid_instance_indices = [
                        idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                        if inst['ignore_flag'] == 0
                    ]

                    # **Mapear GTs reais na ordem correta**
                    gt_idx_list = all_gt_idx_map[img_path]  

                    # import pdb; pdb.set_trace()

                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path)}')  
                    # save_path = f'debug_imgs/epoch{runner.epoch+1}_{os.path.basename(img_path)}'
                    # import pdb; pdb.set_trace()
                    

                    for gt_idx in gt_idx_list:
                    # for gt_idx in valid_instance_indices:

                        related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']  

                        

                        #if index in low_confidence_indices:
                        #if related_global_index in all_classes_low_confidence_scores_global_idx:
                        # if low confidence and not too high confident
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.relabel_conf):
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.5):
                        
                        # if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.3):
                        if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                            # if my_counter<5:

                                                              
                                # import shutil

                                # # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                                
                                # base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                                # possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg", "_grouped.jpg"]

                                # for suffix in possible_suffixes:
                                    
                                #     base_debug_path = base_prefix + suffix
                                #     if os.path.exists(base_debug_path):
                                #         my_counter+=1 
                                #         filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                                #         # if suffix == "_relabeled.jpg":
                                #         #     import pdb; pdb.set_trace()
                                #         shutil.copy(base_debug_path, filtered_debug_path)
                                #         print(f"[INFO] Cópia criada: {filtered_debug_path}")
                                #         break  # Para no primeiro que encontrar 

                                # desenhar_bboxesv3_filtered(all_inputs_map[img_path], sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_filtered.jpg')  
                            # Encontrar `valid_idx` correspondente ao `gt_idx`
                            # if gt_idx in gt_idx_list:
                            #[ME PARECE ERRADO]
                            # valid_idx = valid_instance_indices[gt_idx_list.index(gt_idx)]
                            #[TESTAR ESSE]
                            # import pdb; pdb.set_trace()

                            # if gt_idx >= len(valid_instance_indices):
                            #     import pdb; pdb.set_trace()

                            #valid_idx = valid_instance_indices[gt_idx]
                            valid_idx = gt_idx # alterei pra nao dar bug. Na pratica o id real dele já é o idx. Do jeito que tava antes, funcionava se nunca tivesse ignore_flag=1, mas como agora tem, tinha deixado de funcionar

                            # self.double_thr
                            if allbb_preds_map[img_path][gt_idx]['max_pred'] >= self.double_thr:
                                #update
                                sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = allbb_preds_map[img_path][gt_idx]['pred_label']
                            else:    
                                #filtra
                                sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1

                            
                            # import pdb; pdb.set_trace()
                            # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                            # sub_dataset.data_list[dataset_data_idx]['instances'][gt_idx]['ignore_flag'] = 1
                                #print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, GT: {gt_idx}")
                                #bbox = sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox']
                                # print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, BBOX: {bbox} GT: {gt_idx}")
                        
                    
                                
                                
                            
                        # index += 1
                        

            print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")

@HOOKS.register_module()
class MyHookDoubleFilterPredGT_Class_Relabel(Hook):

    def __init__(self, reload_dataset=False, relabel_conf=0.95, double_thr = 2, filter_conf=0.8, filter_warmup=0, iou_assigner=0.5, low_quality=False, filter_thr=0.7, numGMM=2, filter_type = 'pred', group = False, selcand='max', backup = False):
        """
        Hook personalizado para relabeling de amostras.

        Args:
            reload_dataset (bool): Indica se o dataset deve ser recarregado a cada época.
            my_value (int): Um valor de exemplo para demonstrar parâmetros personalizados.
        """
        self.reload_dataset = reload_dataset
        self.relabel_conf = relabel_conf
        self.double_thr = double_thr
        self.filter_conf = filter_conf
        self.filter_warmup = filter_warmup
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        self.filter_thr = filter_thr
        self.numGMM = numGMM
        self.filter_type = filter_type
        self.group = group
        self.selcand = selcand
        self.backup = backup
        
        
        

    def before_train_epoch(self, runner, *args, **kwargs):
        
        """
        Atualiza os labels de classe (sem alterar os bounding boxes) a cada 100 iterações.
        """
        # import pdb; pdb.set_trace()
        #if (runner.epoch + 1) % 1 != 0:  
        if (runner.epoch + 1) >0 :  
        # if (runner.epoch + 1) >=5 :  

            print(f"[EPOCH]- runner.epoch: {runner.epoch}")
            print(f"[DEBUG] Antes do hook - iter: {runner.iter}")
            dataloader = runner.train_loop.dataloader
            dataset = dataloader.dataset

            #reload_dataset = True
            # my_value = 3
            #reload_dataset = getattr(runner.cfg, 'reload_dataset', False)  
            #my_value =  getattr(runner.cfg, 'my_value', 10)  
            reload_dataset = self.reload_dataset
            relabel_conf = self.relabel_conf
            
            # import pdb; pdb.set_trace()
            if reload_dataset:
                runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

                runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

            # Desencapsular RepeatDataset até o ConcatDataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset

            if not hasattr(dataset, 'datasets'):
                raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

            datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

            assigner = MaxIoUAssigner(
                pos_iou_thr=self.iou_assigner,
                neg_iou_thr=self.iou_assigner,
                min_pos_iou=self.iou_assigner,
                match_low_quality=self.low_quality
            )

            # Criar um dicionário para acesso rápido ao índice das imagens no dataset
            dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                            for sub_dataset_idx, sub_dataset in enumerate(datasets)
                            if hasattr(sub_dataset, 'data_list')
                            for data_idx, data_info in enumerate(sub_dataset.data_list)}
            
            # Dicionário para armazenar os scores médios por GT sem assumir sequência
            allbb_preds_map = defaultdict(dict)
            all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
            all_pred_instances_map = defaultdict(dict)
            # all_inputs_map = defaultdict(dict)

            num_classes = 20
            all_classes_preds_map=[]
            all_classes_gt_idx_map=[]
            for i in range(num_classes):
                all_classes_preds_map.append(defaultdict(list))
                all_classes_gt_idx_map.append(defaultdict(list))
            

            count_gts = 0
            sanity_count = 0
            global_index_counter = 0

            
            
            # Processar o batch de dados
            for batch_idx, data_batch in enumerate(dataloader):
                with torch.no_grad():
                    #data = runner.model.data_preprocessor(data_batch, False)
                    data = runner.model.data_preprocessor(data_batch, True)
                    
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                    
                    #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                    predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
                
                    #temp_filipe

                for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
                # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                    img_path = data_sample.img_path
                    
                    if img_path not in dataset_img_map:
                        print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                        continue

                    # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                    pred_instances = predictions_pred[i].pred_instances

                    # Pegar bounding boxes preditas
                    pred_instances.priors = pred_instances.pop('bboxes')

                    # Obter os ground truths
                    gt_instances = data_sample.gt_instances
                    gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que os GTs estão no mesmo dispositivo das predições
                    gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                    gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que as predições também estão no mesmo dispositivo
                    pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                    pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                    pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                    pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                    all_pred_instances_map[img_path] = pred_instances
                    

                    assign_result = assigner.assign(pred_instances, gt_instances)

                    updated_labels = gt_labels.clone()

                    # Criar lista de GTs para esta imagem
                    all_gt_idx_map[img_path] = []
                    count_gts += assign_result.num_gts

                    
                    
                    
                    # sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    # sub_dataset = datasets[sub_dataset_idx]
                    # import pdb; pdb.set_trace()
                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/sub_{os.path.basename(img_path)}')  
                    # desenhar_bboxesv3(img_path_real,inputs[i], gt_instances, save_path=f'debug_imgs/gt_{os.path.basename(img_path)}')  
                    
                    # if  len(data_samples[i].ignored_instances.labels)>0:
                    #     import pdb; pdb.set_trace()
                    

                    # **Vetorização do processo de associação**
                    for gt_idx in range(assign_result.num_gts):
                        associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                        sanity_count += 1
                        
                        if associated_preds.numel() == 0:
                            continue 

                        # Calcular a média dos scores das bounding boxes associadas a este `gt_idx`
                        # mean_score = pred_instances.scores[associated_preds].mean().item()
                        

                        # max_score = pred_instances.scores[associated_preds].max().item()
                        given_max_score = pred_instances.scores[associated_preds].max().item()
                        # max_logit = pred_instances.logits[associated_preds].max().item()
                        # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                        logits_associated = pred_instances.logits[associated_preds] 
                        myscores = torch.softmax(logits_associated ,dim=-1)

                        
                        # import pdb;pdb.set_trace()
                        # all_pred_instances_map[img_path]['priors']=pred_instances.priors[associated_preds]
                        # all_pred_instances_map[img_path]['scores']=pred_instances.scores[associated_preds]
                        # all_pred_instances_map[img_path]['labels']=pred_instances.labels[associated_preds]
                        # import pdb;pdb.set_trace()
                        # myscores_pred =  myscores.max(dim=1).values.item()

                        # quero pegar o logit_gt da amostra que tem maior logit_pred
                        # import pdb; pdb.set_trace()
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        max_score_val = myscores.max()
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()
                        myious = assign_result.max_overlaps[associated_preds]
                        max_iou_idx = torch.argmax(myious)
                        amostra_id_iou, classe_id_iou = divmod(max_iou_idx.item(), myscores.size(1))
                        score_gt_max_iou = myscores_gt[amostra_id_iou].cpu().item()
                        score_pred_max_iou = myscores[amostra_id_iou, classe_id].cpu().item()

                        # Compute APS
                        row_scores = myscores[amostra_id]  # pega apenas a linha
                        p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                        S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        # p_y = myscores[amostra_id, classe_id].item()
                        # S_APS_score = -myscores[myscores >= p_y].sum().item()


                            

                        # import pdb; pdb.set_trace()
                        # allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score
                         
                        max_score_idx_local = torch.argmax(myscores_gt)
                        #----> remover depois do debug
                        #max_logit_idx_gt = associated_preds[max_score_idx_local]
                        max_logit_idx_gt = associated_preds[amostra_id]
                        logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()
                        


        
                        gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]

                        #---> logit com maior valor [OFICIAL]-voltar depois do debug
                        # max_logit = gt_logits.max().item()
                        # max_logit_idx_local = torch.argmax(gt_logits)
                        # max_logit_idx = associated_preds[max_logit_idx_local]

                        


                        #logit com maior IoU
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()

                        # Pegar idx local do maior IoU
                        max_iou_values = assign_result.max_overlaps[associated_preds]
                        max_iou_idx_local = torch.argmax(max_iou_values)
                        max_iou_idx = associated_preds[max_iou_idx_local]

                        # Vetor contendo ambos (pode repetir se for o mesmo)
                        important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                        important_associated_ids = torch.unique(important_associated_ids)

                        # import pdb; pdb.set_trace()


                        # gt_logits = [gt_logit[updated_labels[gt_idx]] for gt_logit in logits_associated]
                        
                        # Armazenar `gt_idx` na ordem correta
                        all_gt_idx_map[img_path].append(gt_idx)

                        # Armazenar `mean_score` corretamente
                        #---->[original]
                        # allbb_preds_map[img_path][gt_idx] = {'pred':max_logit, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #---->[temporario-filipe-debug] - remover depois do debug

                        other_labels_pred = 0
                        # import pdb;
                        # pdb.set_trace()
                        for ii in range(assign_result.num_gts):
                            if (ii != gt_idx) and (updated_labels[ii].cpu().item() == classe_id):
                                other_labels_pred += 1

                        # if other_labels_pred > 0:
                        #     print(f"outras {other_labels_pred} amostras com a mesma classe {classe_id} para o GT {gt_idx} da imagem {img_path}")
                        #     pdb.set_trace()
                        

                        if self.selcand == 'iou':
                            allbb_preds_map[img_path][gt_idx] = {'pred':score_gt_max_iou, 'logit':logit_gt_max_pred , 
                                                             'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                             'aps':S_APS_score,
                                                              'max_pred':max_score_val.cpu().item(), 'pred_label': classe_id,
                                                                'filtered':False, 'backup_pred_label': other_labels_pred}
                        elif self.selcand == 'max':
                            allbb_preds_map[img_path][gt_idx] = {'pred':score_gt_max_pred, 'logit':logit_gt_max_pred , 
                                                             'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                             'aps':S_APS_score,
                                                              'max_pred':score_pred_max_iou, 'pred_label': classe_id_iou,
                                                                'filtered':False, 'backup_pred_label': other_labels_pred}
                        
                        else:
                            print("Warning: selcand deve ser 'max' ou 'iou'. Usando 'max' por padrão.")
                        global_index_counter += 1



                        

                        #-->original   
                        #confident_preds =  associated_preds[pred_instances.scores[associated_preds] > relabel_conf]
                        #-->temporario-filipe-debug - remover depois
                        # import pdb; pdb.set_trace()

                        

                        confident_preds =  associated_preds[myscores.max(dim=1).values> relabel_conf]
                        

                        
                        if (batch_idx <100):

                            if confident_preds.numel() > 0:
                                # save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg'
                                # desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                pass
                            
                            
                                
                            elif len(associated_preds) > 1 and (max_score_val > 0.45) :
                                labels_group = myscores.argmax(dim=1)
                            
                                # _, qtd = Counter(labels_group.tolist()).most_common(1)[0]
                                most_common_label, qtd = Counter(labels_group.tolist()).most_common(1)[0]

                                scores_most_common = myscores[:,most_common_label]
                                confident_most_common =  associated_preds[scores_most_common> 0.45]
                                
                                # import pdb; pdb.set_trace()
                                # verifica se a quantidade é maior que 50% do total
                                if qtd > (len(associated_preds) / 2)  and len(confident_most_common) > 2:
                                    save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_grouped.jpg'
                                    #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                    desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, associated_preds,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  

                            else:
                                                                
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg'
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg')  
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, associated_preds, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                            

                        
                        
                                  

                        if confident_preds.numel() > 0:
                            
                            #---> original
                            # pred_labels_confident = pred_instances.labels[confident_preds]
                            #---> temporario-filipe-debug - remover depois do debug
                            #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                            pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                            most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        elif self.group and len(associated_preds) > 1 and (max_score_val > 0.45):
                            labels_group = myscores.argmax(dim=1)
                            most_common_label, qtd = Counter(labels_group.tolist()).most_common(1)[0]

                            scores_most_common = myscores[:,most_common_label]
                            confident_most_common =  associated_preds[scores_most_common> 0.45]
                            # import pdb; pdb.set_trace()

                            # verifica se a quantidade é maior que 50% do total
                            if qtd > (len(associated_preds) / 2) and len(confident_most_common) > 2:
                                
                                most_common_label = most_common_label
                            # senao verifica se tem algum elemento com score maior que o threshold
                            elif confident_preds.numel() > 0:
                                pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)
                                most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        # elif confident_preds.numel() == 0:
                        # check if there is any other GT with the same pred_label in the image
                        elif self.backup and (other_labels_pred > 0) and (( runner.epoch + 1) >= self.filter_warmup):
                            most_common_label = classe_id
                        else:
                            continue  


                        updated_labels[gt_idx] = most_common_label
                        #atualiza também no mapeamento
                        allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                        #atualiza max logit-gt
                        gt_logits = logits_associated[:,most_common_label]
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                        max_logit = gt_logits.max().item()

                        ###-->debug -remover depois
                        # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        # max_score_val = myscores.max()
                        # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                        # Compute APS
                        # p_y = myscores[amostra_id, classe_id].item()
                        # S_APS_score = -myscores[myscores >= p_y].sum().item()
                        # row_scores = myscores[amostra_id]  # pega apenas a linha
                        # p_y = row_scores[classe_id].item()
                        # S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        row_scores = myscores[amostra_id]  # pega apenas a linha
                        p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                        S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        # if S_APS_score < -1:
                        #     print(S_APS_score)
                        #     print("a")
                        #     import pdb; pdb.set_trace()
                        #     print("ok")
                        allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score

                        #
                        #-->[original]
                        #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                        #-->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                        allbb_preds_map[img_path][gt_idx]['filtered'] = True



                        
                        

                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                    valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                    # Atualizar os labels corretamente usando o mapeamento
                    for gt_idx, valid_idx in enumerate(valid_instance_indices):
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                        # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
            

            # só faz filter depois do warmup
            if (runner.epoch + 1) >= self.filter_warmup:

                # filtering
                #Unir todas as probabilidades das imagens e treinar o GMM**
                # all_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values()
                # ]).reshape(-1, 1)
                # import pdb; pdb.set_trace()
                print("[DEBUG1]: Entrou no filtro")
                all_classes_low_confidence_scores_global_idx = []

                all_classes_pred_cutoff =  [0.0 for _ in range(num_classes)]

                for c in range(num_classes):
                    # c_class_scores = np.array([
                    #     score['pred'] for img_scores in allbb_preds_map.values()
                    #     for score in img_scores.values() if score['gt_label'] == c
                    # ]).reshape(-1, 1)

                    # if runner.epoch + 1 >1:
                    #     import pdb; pdb.set_trace()

                    scores = np.array([])
                    # img_indexes = np.array([])
                    scores_dict = {'pred': np.array([]),
                                   'logit': np.array([]),
                                   'aps': np.array([])}
                    low_conf_dict = {'pred': np.array([]),
                                   'logit': np.array([]),
                                   'aps': np.array([])}
                    img_paths = np.array([], dtype=object)
                    c_global_indexes = np.array([])

                    for img_path, img_info in allbb_preds_map.items():
                        #for pred, gt_label, index  in img_info.values():
                        for temp_gt_idx, values in img_info.items():
                            #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                            #if gt_label== c:
                            if values['gt_label'] == c:
                                scores_dict['pred'] = np.append(scores_dict['pred'], values['pred'])
                                scores_dict['logit'] = np.append(scores_dict['logit'], values['logit'])
                                scores_dict['aps'] = np.append(scores_dict['aps'], values['aps'])

                                # calculando como pred apenas para ver depois se len(scores) == 0
                                scores = np.append(scores, values['pred'])

                                
                                # if self.filter_type == 'pred':
                                #     scores = np.append(scores, values['pred'])
                                # elif self.filter_type == 'logit':
                                #     scores = np.append(scores, values['logit'])
                                # elif self.filter_type == 'aps':
                                #     scores = np.append(scores, values['aps'])

                                c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                                if img_path not in img_paths:
                                    img_paths = np.append(img_paths, img_path)
                    
                    if len(scores) == 0:
                        print(f"Aviso: Nenhuma amostra encontrada para a classe {c}.")
                        continue

                    
                    

                    # low_confidence_scores = calculate_gmm(c_class_scores, n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                                
                    low_conf_dict['pred'] = calculate_gmm(scores_dict['pred'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    low_conf_dict['logit'] = calculate_gmm(scores_dict['logit'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    low_conf_dict['aps'] = calculate_gmm(scores_dict['aps'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)

                    if self.filter_type == 'pred':
                        scores = scores_dict['pred']
                        low_confidence_scores = low_conf_dict['pred']
                    elif self.filter_type == 'logit':
                        scores = scores_dict['logit']
                        low_confidence_scores = low_conf_dict['logit']
                    elif self.filter_type == 'aps':
                        scores = scores_dict['aps']
                        low_confidence_scores = low_conf_dict['aps']

                    

                    # print("[DEBUG1.5]: INICIANDO GMM")

                    c_class_scores  = scores.reshape(-1, 1)


                    #gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    # gmm = GaussianMixture(n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    # gmm.fit(c_class_scores)

                    # # Identificar o cluster com menor média (baixa confiança)
                    # low_confidence_component = np.argmin(gmm.means_)
                    # low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]


                    threshold = self.filter_conf
                
                    low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                    all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                    
                    if len(low_confidence_indices) > 0:
                        all_classes_pred_cutoff[c] = np.max(scores_dict['pred'].reshape(-1, 1)[low_confidence_indices])
                    else:
                        print(f"Aviso: Nenhuma amostra com confiança > {threshold} encontrada.")
                        all_classes_pred_cutoff[c] = np.min(scores_dict['pred'].reshape(-1, 1))  # valor alternativo seguro

                    

                    # draw pred hist
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_pred_ep{runner.epoch + 1}.png"
                    low_confidence_indices_temp = np.where(low_conf_dict['pred'] > threshold)[0]
                    draw_score_histogram(scores_dict['pred'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_conf)

                    # draw logit hist
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_logit_ep{runner.epoch + 1}.png"
                    low_confidence_indices_temp = np.where(low_conf_dict['logit'] > threshold)[0]
                    draw_score_histogram(scores_dict['logit'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_conf)

                    # draw aps hist
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_aps_ep{runner.epoch + 1}.png"
                    low_confidence_indices_temp = np.where(low_conf_dict['aps'] > threshold)[0]
                    draw_score_histogram(scores_dict['aps'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_conf)

                    # Dentro do seu for c in range(num_classes):
                    # import pdb; pdb.set_trace()
                    # ... após o fit do GMM
                    # plt.figure()
                    # plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                    # # Adiciona linha vertical que separa as duas classes com base no GMM
                    # plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                    # # plt.legend()
                    # plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                    # plt.xlabel('Score')
                    # plt.ylabel('Frequência')
                    # plt.grid(True)
                    # plt.tight_layout()
                    # # Criar pasta, se necessário
                    # save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    # #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                    # plt.savefig(save_path)
                    # plt.close()
                   

                # print("[DEBUG2]: SAIU do GMM")
                # if len(all_scores) > 0:
                    

                # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
                # index = 0
                # import pdb; pdb.set_trace()
                my_counter = 0
                for img_path, gt_scores in allbb_preds_map.items():
                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    

                    valid_instance_indices = [
                        idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                        if inst['ignore_flag'] == 0
                    ]

                    # **Mapear GTs reais na ordem correta**
                    gt_idx_list = all_gt_idx_map[img_path]  

                    # import pdb; pdb.set_trace()

                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path)}')  
                    # save_path = f'debug_imgs/epoch{runner.epoch+1}_{os.path.basename(img_path)}'
                    # import pdb; pdb.set_trace()
                    import shutil
                    possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg", "_grouped.jpg"]

                    for gt_idx in gt_idx_list:
                    # for gt_idx in valid_instance_indices:

                        related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']  

                        

                        #if index in low_confidence_indices:
                        #if related_global_index in all_classes_low_confidence_scores_global_idx:
                        # if low confidence and not too high confident
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.relabel_conf):
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.5):
                        
                        # if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.3):
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                        
                        class_pred = allbb_preds_map[img_path][gt_idx]['pred_label']
                        class_gt = allbb_preds_map[img_path][gt_idx]['gt_label']
                        valid_idx = valid_instance_indices[gt_idx]

                        msg_filter = "empty"
                        # pred é ruidoso e gt é clean 
                        # if (allbb_preds_map[img_path][gt_idx]['max_pred'] <= all_classes_pred_cutoff[class_pred]) and (allbb_preds_map[img_path][gt_idx]['pred'] > all_classes_pred_cutoff[class_gt]):
                        #     #deixa como está
                        #     msg_filter = "low_high"
                        #     pass
                        # pred é clean e gt é ruidoso 
                        # elif (allbb_preds_map[img_path][gt_idx]['max_pred'] > all_classes_pred_cutoff[class_pred]) and (allbb_preds_map[img_path][gt_idx]['pred'] <= all_classes_pred_cutoff[class_gt]):
                        #     #atualiza label gt
                        #     sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = allbb_preds_map[img_path][gt_idx]['pred_label']
                        #     msg_filter = "high_low"
                        
                        #if (allbb_preds_map[img_path][gt_idx]['max_pred'] <= all_classes_pred_cutoff[class_pred]) and (allbb_preds_map[img_path][gt_idx]['pred'] <= all_classes_pred_cutoff[class_gt]):
                        if (allbb_preds_map[img_path][gt_idx]['pred'] <= all_classes_pred_cutoff[class_gt]) and (allbb_preds_map[img_path][gt_idx]['max_pred']<= self.filter_thr): 
                            sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                            msg_filter = "low_low"
                        #pred clean e gt clean
                        # elif (allbb_preds_map[img_path][gt_idx]['max_pred'] > all_classes_pred_cutoff[class_pred]) and (allbb_preds_map[img_path][gt_idx]['pred'] > all_classes_pred_cutoff[class_gt]):
                        #     sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = allbb_preds_map[img_path][gt_idx]['pred_label']
                        #     msg_filter = "high_high"
                            
                            

                           

                        if my_counter<100:

                                                            
                            

                            # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                            
                            base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                            possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg", "_grouped.jpg"]

                            for suffix in possible_suffixes:
                                
                                base_debug_path = base_prefix + suffix
                                if os.path.exists(base_debug_path):
                                    my_counter+=1 
                                   # filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                                    filtered_debug_path = base_debug_path[:-4] + "_" + msg_filter + ".jpg"
                                    shutil.copy(base_debug_path, filtered_debug_path)
                                    print(f"[INFO] Cópia criada: {filtered_debug_path}")
                                    break  # Para no primeiro que encontrar 

                            # desenhar_bboxesv3_filtered(all_inputs_map[img_path], sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_filtered.jpg')  
                        # Encontrar `valid_idx` correspondente ao `gt_idx`
                        # if gt_idx in gt_idx_list:
                        #[ME PARECE ERRADO]
                        # valid_idx = valid_instance_indices[gt_idx_list.index(gt_idx)]
                        #[TESTAR ESSE]
                        # import pdb; pdb.set_trace()
                        # valid_idx = valid_instance_indices[gt_idx]

                        # self.double_thr
                        # if allbb_preds_map[img_path][gt_idx]['max_pred'] >= self.double_thr:
                        #     #update
                        #     sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = allbb_preds_map[img_path][gt_idx]['pred_label']
                        # else:    
                        #     #filtra
                        #     sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1

                            
                            # import pdb; pdb.set_trace()
                            # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                            # sub_dataset.data_list[dataset_data_idx]['instances'][gt_idx]['ignore_flag'] = 1
                                #print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, GT: {gt_idx}")
                                #bbox = sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox']
                                # print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, BBOX: {bbox} GT: {gt_idx}")
                        
                    
                                
                                
                            
                        # index += 1
                        

            print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")


@HOOKS.register_module()
class MyHookMixFilter_Class_Relabel(Hook):

    def __init__(self, reload_dataset=False, relabel_conf=0.95, relabel_conf2=0.9, filter_conf=0.8, filter_warmup=0, iou_assigner=0.5, low_quality=False, filter_thr=0.7, numGMM=2, filter_type = 'pred', ep_pred=5):
        """
        Hook personalizado para relabeling de amostras.

        Args:
            reload_dataset (bool): Indica se o dataset deve ser recarregado a cada época.
            my_value (int): Um valor de exemplo para demonstrar parâmetros personalizados.
        """
        self.reload_dataset = reload_dataset
        self.relabel_conf = relabel_conf
        self.filter_conf = filter_conf
        self.filter_warmup = filter_warmup
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        self.filter_thr = filter_thr
        self.numGMM = numGMM
        self.filter_type = filter_type
        self.ep_pred = ep_pred
        self.relabel_conf2 = relabel_conf2
        

    def before_train_epoch(self, runner, *args, **kwargs):
        
        """
        Atualiza os labels de classe (sem alterar os bounding boxes) a cada 100 iterações.
        """
        
        if (runner.epoch + 1) >0 :  
        

            # print(f"[EPOCH]- runner.epoch: {runner.epoch}")
            # print(f"[DEBUG] Antes do hook - iter: {runner.iter}")
            dataloader = runner.train_loop.dataloader
            dataset = dataloader.dataset

            #reload_dataset = True
            # my_value = 3
            #reload_dataset = getattr(runner.cfg, 'reload_dataset', False)  
            #my_value =  getattr(runner.cfg, 'my_value', 10)  
            reload_dataset = self.reload_dataset
            # relabel_conf = self.relabel_conf

            if (runner.epoch + 1) >= 5:
                relabel_conf = self.relabel_conf2
            else:
                relabel_conf = self.relabel_conf
            
            # import pdb; pdb.set_trace()
            if reload_dataset:
                runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

                runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

            # Desencapsular RepeatDataset até o ConcatDataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset

            if not hasattr(dataset, 'datasets'):
                raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

            datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

            assigner = MaxIoUAssigner(
                pos_iou_thr=self.iou_assigner,
                neg_iou_thr=self.iou_assigner,
                min_pos_iou=self.iou_assigner,
                match_low_quality=self.low_quality
            )

            # Criar um dicionário para acesso rápido ao índice das imagens no dataset
            dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                            for sub_dataset_idx, sub_dataset in enumerate(datasets)
                            if hasattr(sub_dataset, 'data_list')
                            for data_idx, data_info in enumerate(sub_dataset.data_list)}
            
            # Dicionário para armazenar os scores médios por GT sem assumir sequência
            allbb_preds_map = defaultdict(dict)
            all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
            all_pred_instances_map = defaultdict(dict)
            # all_inputs_map = defaultdict(dict)

            num_classes = 20
            all_classes_preds_map=[]
            all_classes_gt_idx_map=[]
            for i in range(num_classes):
                all_classes_preds_map.append(defaultdict(list))
                all_classes_gt_idx_map.append(defaultdict(list))
            

            count_gts = 0
            sanity_count = 0
            global_index_counter = 0

            
            
            # Processar o batch de dados
            for batch_idx, data_batch in enumerate(dataloader):
                with torch.no_grad():
                    #data = runner.model.data_preprocessor(data_batch, False)
                    data = runner.model.data_preprocessor(data_batch, True)
                    
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                    
                    #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                    predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
                
                    #temp_filipe

                for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
                # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                    img_path = data_sample.img_path
                    
                    if img_path not in dataset_img_map:
                        print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                        continue

                    # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                    pred_instances = predictions_pred[i].pred_instances

                    # Pegar bounding boxes preditas
                    pred_instances.priors = pred_instances.pop('bboxes')

                    # Obter os ground truths
                    gt_instances = data_sample.gt_instances
                    gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que os GTs estão no mesmo dispositivo das predições
                    gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                    gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que as predições também estão no mesmo dispositivo
                    pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                    pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                    pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                    pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                    all_pred_instances_map[img_path] = pred_instances
                    

                    assign_result = assigner.assign(pred_instances, gt_instances)

                    updated_labels = gt_labels.clone()

                    # Criar lista de GTs para esta imagem
                    all_gt_idx_map[img_path] = []
                    count_gts += assign_result.num_gts

                    
                   
                    

                    # **Vetorização do processo de associação**
                    for gt_idx in range(assign_result.num_gts):
                        associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                        sanity_count += 1
                        
                        if associated_preds.numel() == 0:
                            continue 

                        # Calcular a média dos scores das bounding boxes associadas a este `gt_idx`
                        # mean_score = pred_instances.scores[associated_preds].mean().item()
                        

                        # max_score = pred_instances.scores[associated_preds].max().item()
                        given_max_score = pred_instances.scores[associated_preds].max().item()
                        # max_logit = pred_instances.logits[associated_preds].max().item()
                        # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                        logits_associated = pred_instances.logits[associated_preds] 
                        myscores = torch.softmax(logits_associated ,dim=-1)

                        
                        
                        # import pdb; pdb.set_trace()
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        max_score_val = myscores.max()
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()
                         
                        max_score_idx_local = torch.argmax(myscores_gt)
                        #----> remover depois do debug
                        #max_logit_idx_gt = associated_preds[max_score_idx_local]
                        max_logit_idx_gt = associated_preds[amostra_id]
                        logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()
                        


        
                        gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]


                        # Pegar idx local do maior IoU
                        max_iou_values = assign_result.max_overlaps[associated_preds]
                        max_iou_idx_local = torch.argmax(max_iou_values)
                        max_iou_idx = associated_preds[max_iou_idx_local]

                        # Vetor contendo ambos (pode repetir se for o mesmo)
                        important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                        important_associated_ids = torch.unique(important_associated_ids)

                        # import pdb; pdb.set_trace()


                        # gt_logits = [gt_logit[updated_labels[gt_idx]] for gt_logit in logits_associated]
                        
                        # Armazenar `gt_idx` na ordem correta
                        all_gt_idx_map[img_path].append(gt_idx)

                        # Armazenar `mean_score` corretamente
                        #---->[original]
                        # allbb_preds_map[img_path][gt_idx] = {'pred':max_logit, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #---->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx] = {'pred':score_gt_max_pred, 
                                                             'logit':logit_gt_max_pred , 
                                                             'gt_label':gt_labels[gt_idx].item(), 
                                                             'global_index_counter':global_index_counter, 
                                                             'filtered':False,
                                                             'max_pred':max_score_val.cpu().item(),
                                                             'pred_label': classe_id}
                        global_index_counter += 1



                        # import pdb; pdb.set_trace()

                        #-->original   
                        #confident_preds =  associated_preds[pred_instances.scores[associated_preds] > relabel_conf]
                        #-->temporario-filipe-debug - remover depois
                        # import pdb; pdb.set_trace()
                        confident_preds =  associated_preds[myscores.max(dim=1).values> relabel_conf]

                        
                        if (batch_idx <1):
                            
                            if confident_preds.numel() == 0:
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg'
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg')  
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                # pass
                            else:
                                
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg'
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                            


                        if confident_preds.numel() == 0:
                            continue  

                        
                        
                        # if (runner.epoch + 1 >= 0) and (i <100):
                        #     import pdb; pdb.set_trace()
                        #     img_path_real = img_path
                        #     # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/sub_{os.path.basename(img_path)}')  
                            # desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}.jpg')  
                        

                        #---> original
                        # pred_labels_confident = pred_instances.labels[confident_preds]
                        #---> temporario-filipe-debug - remover depois do debug
                        #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                        pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                        most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        updated_labels[gt_idx] = most_common_label
                        #atualiza também no mapeamento
                        allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                        #atualiza max logit-gt
                        gt_logits = logits_associated[:,most_common_label]
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                        max_logit = gt_logits.max().item()

                        ###-->debug -remover depois
                        # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        # max_score_val = myscores.max()
                        # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                        #
                        #-->[original]
                        #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                        #-->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                        allbb_preds_map[img_path][gt_idx]['filtered'] = True
                        
                        

                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                    valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                    # Atualizar os labels corretamente usando o mapeamento
                    for gt_idx, valid_idx in enumerate(valid_instance_indices):
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                        # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
            

            # só faz filter depois do warmup
            if (runner.epoch + 1) >= self.filter_warmup:

                # filtering
                #Unir todas as probabilidades das imagens e treinar o GMM**
                # all_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values()
                # ]).reshape(-1, 1)
                # import pdb; pdb.set_trace()
                # print("[DEBUG1]: Entrou no filtro")
                all_classes_low_confidence_scores_global_idx = []
                filtered_proportion_class = np.array([])

                for c in range(num_classes):


                    scores = np.array([])
                    # img_indexes = np.array([])
                    img_paths = np.array([], dtype=object)
                    c_global_indexes = np.array([])

                    for img_path, img_info in allbb_preds_map.items():
                        #for pred, gt_label, index  in img_info.values():
                        for temp_gt_idx, values in img_info.items():
                            #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                            #if gt_label== c:
                            if values['gt_label'] == c:
                                
                                # if self.filter_type == 'pred':
                                if runner.epoch + 1 >= self.ep_pred:
                                    scores = np.append(scores, values['pred'])
                                    self.numGMM = 4
                               # elif self.filter_type == 'logit':
                                else:
                                    
                                    scores = np.append(scores, values['logit'])
                                    self.numGMM = 2
                                # img_indexes = np.append(img_indexes, values['global_index_counter'])
                                c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                                if img_path not in img_paths:
                                    img_paths = np.append(img_paths, img_path)
                    # if runner.epoch + 1 >1:
                    #     import pdb; pdb.set_trace()
                    # print("[DEBUG1.5]: INICIANDO GMM")
                    c_class_scores = scores.reshape(-1, 1)
                                
                    
                    #gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    gmm = GaussianMixture(n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    gmm.fit(c_class_scores)

                    # Identificar o cluster com menor média (baixa confiança)
                    low_confidence_component = np.argmin(gmm.means_)
                    low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]
                    threshold = self.filter_conf
                
                    low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                    all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])
                    filtered_proportion_class = np.append(filtered_proportion_class, len(low_confidence_indices) / len(c_class_scores))

                    # Valor do ponto de corte nos scores (máximo dos scores das amostras de baixa confiança)
                    # Tratamento para array vazio:
                    if len(low_confidence_indices) > 0:
                        score_cutoff = np.max(c_class_scores[low_confidence_indices])
                    else:
                        print(f"Aviso: Nenhuma amostra com confiança > {threshold} encontrada.")
                        score_cutoff = np.min(c_class_scores)  # valor alternativo seguro

                    ## save import pickle
                    # import pickle
                    # with open(f"c{c}_scores_ep{runner.epoch + 1}.pkl", "wb") as f:
                    #     pickle.dump(c_class_scores, f)
                    #end saving

                    import matplotlib.pyplot as plt

                    # Dentro do seu for c in range(num_classes):
                    # import pdb; pdb.set_trace()
                    # ... após o fit do GMM
                    plt.figure()
                    plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                    # Adiciona linha vertical que separa as duas classes com base no GMM
                    plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                    # plt.legend()
                    plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                    plt.xlabel('Score')
                    plt.ylabel('Frequência')
                    plt.grid(True)
                    plt.tight_layout()
                    # Criar pasta, se necessário
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                    plt.savefig(save_path)
                    plt.close()
                   

                # print("[DEBUG2]: SAIU do GMM")
                # if len(all_scores) > 0:
                    

                # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
                # index = 0
                # import pdb; pdb.set_trace()
                my_counter = 0
                for img_path, gt_scores in allbb_preds_map.items():
                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    

                    valid_instance_indices = [
                        idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                        if inst['ignore_flag'] == 0
                    ]

                    # **Mapear GTs reais na ordem correta**
                    gt_idx_list = all_gt_idx_map[img_path]  

   
                    

                    for gt_idx in gt_idx_list:
                    # for gt_idx in valid_instance_indices:

                        related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']  

                        
                        if runner.epoch + 1 >= self.ep_pred:
                            if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                                if my_counter<5:

                                    my_counter+=1                                
                                    import shutil

                                    # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                                    base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                                    possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg"]

                                    for suffix in possible_suffixes:
                                        
                                        base_debug_path = base_prefix + suffix
                                        if os.path.exists(base_debug_path):
                                            filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                                            
                                            shutil.copy(base_debug_path, filtered_debug_path)
                                            print(f"[INFO] Cópia criada: {filtered_debug_path}")
                                            break  # Para no primeiro que encontrar 

                                
                                valid_idx = valid_instance_indices[gt_idx]

                                sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                            
                            # Double Relabel adaptative
                            else:
                                valid_idx = valid_instance_indices[gt_idx]
                                gt_label_temp = sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label']
                                #if allbb_preds_map[img_path][gt_idx]['max_pred'] >= (0.5 + filtered_proportion_class[gt_label_temp]):
                                if allbb_preds_map[img_path][gt_idx]['max_pred'] >= (0.7 + filtered_proportion_class[gt_label_temp]):
                                    # if (sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] != allbb_preds_map[img_path][gt_idx]['pred_label']):
                                    #     import pdb; pdb.set_trace()

                                    #update
                                    sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = allbb_preds_map[img_path][gt_idx]['pred_label']
                        #[logit mode] Filter all bellow 0 logit
                        else:
                            if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr) :
                            # if  (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr) and (allbb_preds_map[img_path][gt_idx]['logit'] < 0):
                                if my_counter<5:

                                    my_counter+=1                                
                                    import shutil

                                    # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                                    base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                                    possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg"]

                                    for suffix in possible_suffixes:
                                        
                                        base_debug_path = base_prefix + suffix
                                        if os.path.exists(base_debug_path):
                                            filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                                            
                                            shutil.copy(base_debug_path, filtered_debug_path)
                                            print(f"[INFO] Cópia criada: {filtered_debug_path}")
                                            break  # Para no primeiro que encontrar 

                                
                                valid_idx = valid_instance_indices[gt_idx]

                                

                                

                                sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                            
                            
                       

            print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")


@HOOKS.register_module()
class MyHookFilterPredGT_Class_GivenRelabel(Hook):

    def __init__(self, reload_dataset=False, relabel_conf=0.95, filter_conf=0.8, filter_warmup=0, iou_assigner=0.5, low_quality=False, filter_thr=0.7, numGMM=2, filter_type = 'pred'):
        """
        Hook personalizado para relabeling de amostras.

        Args:
            reload_dataset (bool): Indica se o dataset deve ser recarregado a cada época.
            my_value (int): Um valor de exemplo para demonstrar parâmetros personalizados.
        """
        self.reload_dataset = reload_dataset
        self.relabel_conf = relabel_conf
        self.filter_conf = filter_conf
        self.filter_warmup = filter_warmup
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        self.filter_thr = filter_thr
        self.numGMM = numGMM
        self.filter_type = filter_type
        

    def before_train_epoch(self, runner, *args, **kwargs):
        
        """
        Atualiza os labels de classe (sem alterar os bounding boxes) a cada 100 iterações.
        """
        # import pdb; pdb.set_trace()
        #if (runner.epoch + 1) % 1 != 0:  
        if (runner.epoch + 1) >0 :  
        # if (runner.epoch + 1) >=5 :  

            print(f"[EPOCH]- runner.epoch: {runner.epoch}")
            print(f"[DEBUG] Antes do hook - iter: {runner.iter}")
            dataloader = runner.train_loop.dataloader
            dataset = dataloader.dataset

            #reload_dataset = True
            # my_value = 3
            #reload_dataset = getattr(runner.cfg, 'reload_dataset', False)  
            #my_value =  getattr(runner.cfg, 'my_value', 10)  
            reload_dataset = self.reload_dataset
            relabel_conf = self.relabel_conf
            
            # import pdb; pdb.set_trace()
            if reload_dataset:
                runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

                runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

            # Desencapsular RepeatDataset até o ConcatDataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset

            if not hasattr(dataset, 'datasets'):
                raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

            datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

            assigner = MaxIoUAssigner(
                pos_iou_thr=self.iou_assigner,
                neg_iou_thr=self.iou_assigner,
                min_pos_iou=self.iou_assigner,
                match_low_quality=self.low_quality
            )

            # Criar um dicionário para acesso rápido ao índice das imagens no dataset
            dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                            for sub_dataset_idx, sub_dataset in enumerate(datasets)
                            if hasattr(sub_dataset, 'data_list')
                            for data_idx, data_info in enumerate(sub_dataset.data_list)}
            
            # Dicionário para armazenar os scores médios por GT sem assumir sequência
            allbb_preds_map = defaultdict(dict)
            all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
            all_pred_instances_map = defaultdict(dict)
            # all_inputs_map = defaultdict(dict)

            num_classes = 20
            all_classes_preds_map=[]
            all_classes_gt_idx_map=[]
            for i in range(num_classes):
                all_classes_preds_map.append(defaultdict(list))
                all_classes_gt_idx_map.append(defaultdict(list))
            

            count_gts = 0
            sanity_count = 0
            global_index_counter = 0

            
            
            # Processar o batch de dados
            for batch_idx, data_batch in enumerate(dataloader):
                with torch.no_grad():
                    #data = runner.model.data_preprocessor(data_batch, False)
                    data = runner.model.data_preprocessor(data_batch, True)
                    
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                    
                    #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                    predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
                
                    #temp_filipe

                for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
                # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                    img_path = data_sample.img_path
                    
                    if img_path not in dataset_img_map:
                        print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                        continue

                    # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                    pred_instances = predictions_pred[i].pred_instances

                    # Pegar bounding boxes preditas
                    pred_instances.priors = pred_instances.pop('bboxes')

                    # Obter os ground truths
                    gt_instances = data_sample.gt_instances
                    gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que os GTs estão no mesmo dispositivo das predições
                    gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                    gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que as predições também estão no mesmo dispositivo
                    pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                    pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                    pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                    pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                    all_pred_instances_map[img_path] = pred_instances
                    

                    assign_result = assigner.assign(pred_instances, gt_instances)

                    updated_labels = gt_labels.clone()

                    # Criar lista de GTs para esta imagem
                    all_gt_idx_map[img_path] = []
                    count_gts += assign_result.num_gts

                    
                    
                    
                    # sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    # sub_dataset = datasets[sub_dataset_idx]
                    # import pdb; pdb.set_trace()
                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/sub_{os.path.basename(img_path)}')  
                    # desenhar_bboxesv3(img_path_real,inputs[i], gt_instances, save_path=f'debug_imgs/gt_{os.path.basename(img_path)}')  
                    
                    # if  len(data_samples[i].ignored_instances.labels)>0:
                    #     import pdb; pdb.set_trace()
                    

                    # **Vetorização do processo de associação**
                    for gt_idx in range(assign_result.num_gts):
                        associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                        sanity_count += 1
                        
                        if associated_preds.numel() == 0:
                            continue 

                        # Calcular a média dos scores das bounding boxes associadas a este `gt_idx`
                        # mean_score = pred_instances.scores[associated_preds].mean().item()
                        

                        # max_score = pred_instances.scores[associated_preds].max().item()
                        given_max_score = pred_instances.scores[associated_preds].max().item()
                        # max_logit = pred_instances.logits[associated_preds].max().item()
                        # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                        logits_associated = pred_instances.logits[associated_preds] 
                        myscores = torch.softmax(logits_associated ,dim=-1)

                        
                        # import pdb;pdb.set_trace()
                        # all_pred_instances_map[img_path]['priors']=pred_instances.priors[associated_preds]
                        # all_pred_instances_map[img_path]['scores']=pred_instances.scores[associated_preds]
                        # all_pred_instances_map[img_path]['labels']=pred_instances.labels[associated_preds]
                        # import pdb;pdb.set_trace()
                        # myscores_pred =  myscores.max(dim=1).values.item()

                        # quero pegar o logit_gt da amostra que tem maior logit_pred
                        # import pdb; pdb.set_trace()
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        max_score_val = myscores.max()
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()
                         
                        max_score_idx_local = torch.argmax(myscores_gt)
                        #----> remover depois do debug
                        #max_logit_idx_gt = associated_preds[max_score_idx_local]
                        max_logit_idx_gt = associated_preds[amostra_id]
                        logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()
                        


        
                        gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]

                        #---> logit com maior valor [OFICIAL]-voltar depois do debug
                        # max_logit = gt_logits.max().item()
                        # max_logit_idx_local = torch.argmax(gt_logits)
                        # max_logit_idx = associated_preds[max_logit_idx_local]

                        


                        #logit com maior IoU
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()

                        # Pegar idx local do maior IoU
                        max_iou_values = assign_result.max_overlaps[associated_preds]
                        max_iou_idx_local = torch.argmax(max_iou_values)
                        max_iou_idx = associated_preds[max_iou_idx_local]

                        # Vetor contendo ambos (pode repetir se for o mesmo)
                        important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                        important_associated_ids = torch.unique(important_associated_ids)

                        # import pdb; pdb.set_trace()


                        # gt_logits = [gt_logit[updated_labels[gt_idx]] for gt_logit in logits_associated]
                        
                        # Armazenar `gt_idx` na ordem correta
                        all_gt_idx_map[img_path].append(gt_idx)

                        # Armazenar `mean_score` corretamente
                        #---->[original]
                        # allbb_preds_map[img_path][gt_idx] = {'pred':max_logit, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #---->[temporario-filipe-debug] - remover depois do debug
                        given_max_score
                        allbb_preds_map[img_path][gt_idx] = {'pred':score_gt_max_pred, 'logit':logit_gt_max_pred , 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 'filtered':False}
                        
                        global_index_counter += 1



                        # import pdb; pdb.set_trace()

                        #-->original   
                        confident_preds =  associated_preds[pred_instances.scores[associated_preds] > relabel_conf]
                        #-->temporario-filipe-debug - remover depois
                        # import pdb; pdb.set_trace()
                        # confident_preds =  associated_preds[myscores.max(dim=1).values> relabel_conf]

                        
                        if (batch_idx <200):
                            
                            if confident_preds.numel() == 0:
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg'
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg')  
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                # pass
                            else:
                                
                                
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg')  
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg'
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                            


                        if confident_preds.numel() == 0:
                            continue  

                        

                        #---> original
                        pred_labels_confident = pred_instances.labels[confident_preds]
                        #---> temporario-filipe-debug - remover depois do debug
                        #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                        # pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                        most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        updated_labels[gt_idx] = most_common_label
                        #atualiza também no mapeamento
                        allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                        #atualiza max logit-gt
                        gt_logits = logits_associated[:,most_common_label]
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                        max_logit = gt_logits.max().item()

                        ###-->debug -remover depois
                        # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        # max_score_val = myscores.max()
                        # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                        #
                        #-->[original]
                        #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                        #-->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                        allbb_preds_map[img_path][gt_idx]['filtered'] = True
                        
                        

                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                    valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                    # Atualizar os labels corretamente usando o mapeamento
                    for gt_idx, valid_idx in enumerate(valid_instance_indices):
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                        # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
            

            # só faz filter depois do warmup
            if (runner.epoch + 1) >= self.filter_warmup:

                # filtering
                #Unir todas as probabilidades das imagens e treinar o GMM**
                # all_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values()
                # ]).reshape(-1, 1)
                # import pdb; pdb.set_trace()
                print("[DEBUG1]: Entrou no filtro")
                all_classes_low_confidence_scores_global_idx = []

                for c in range(num_classes):
                    # c_class_scores = np.array([
                    #     score['pred'] for img_scores in allbb_preds_map.values()
                    #     for score in img_scores.values() if score['gt_label'] == c
                    # ]).reshape(-1, 1)

                    # if runner.epoch + 1 >1:
                    #     import pdb; pdb.set_trace()

                    scores = np.array([])
                    # img_indexes = np.array([])
                    img_paths = np.array([], dtype=object)
                    c_global_indexes = np.array([])

                    for img_path, img_info in allbb_preds_map.items():
                        #for pred, gt_label, index  in img_info.values():
                        for temp_gt_idx, values in img_info.items():
                            #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                            #if gt_label== c:
                            if values['gt_label'] == c:
                                
                                if self.filter_type == 'pred':
                                    scores = np.append(scores, values['pred'])
                                elif self.filter_type == 'logit':
                                    scores = np.append(scores, values['logit'])
                                # img_indexes = np.append(img_indexes, values['global_index_counter'])
                                c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                                if img_path not in img_paths:
                                    img_paths = np.append(img_paths, img_path)
                    # if runner.epoch + 1 >1:
                    #     import pdb; pdb.set_trace()
                    # print("[DEBUG1.5]: INICIANDO GMM")
                    c_class_scores = scores.reshape(-1, 1)
                                
                    
                    #gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    gmm = GaussianMixture(n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    gmm.fit(c_class_scores)

                    # Identificar o cluster com menor média (baixa confiança)
                    low_confidence_component = np.argmin(gmm.means_)
                    low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]
                    threshold = self.filter_conf
                
                    low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                    all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                    # Valor do ponto de corte nos scores (máximo dos scores das amostras de baixa confiança)
                    # Tratamento para array vazio:
                    if len(low_confidence_indices) > 0:
                        score_cutoff = np.max(c_class_scores[low_confidence_indices])
                    else:
                        print(f"Aviso: Nenhuma amostra com confiança > {threshold} encontrada.")
                        score_cutoff = np.min(c_class_scores)  # valor alternativo seguro

                    ## save import pickle
                    # import pickle
                    # with open(f"c{c}_scores_ep{runner.epoch + 1}.pkl", "wb") as f:
                    #     pickle.dump(c_class_scores, f)
                    #end saving

                    import matplotlib.pyplot as plt

                    # Dentro do seu for c in range(num_classes):
                    # import pdb; pdb.set_trace()
                    # ... após o fit do GMM
                    plt.figure()
                    plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                    # Adiciona linha vertical que separa as duas classes com base no GMM
                    plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                    # plt.legend()
                    plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                    plt.xlabel('Score')
                    plt.ylabel('Frequência')
                    plt.grid(True)
                    plt.tight_layout()
                    # Criar pasta, se necessário
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                    plt.savefig(save_path)
                    plt.close()
                   

                # print("[DEBUG2]: SAIU do GMM")
                # if len(all_scores) > 0:
                    

                # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
                # index = 0
                # import pdb; pdb.set_trace()
                my_counter = 0
                for img_path, gt_scores in allbb_preds_map.items():
                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    

                    valid_instance_indices = [
                        idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                        if inst['ignore_flag'] == 0
                    ]

                    # **Mapear GTs reais na ordem correta**
                    gt_idx_list = all_gt_idx_map[img_path]  

                    # import pdb; pdb.set_trace()

                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path)}')  
                    # save_path = f'debug_imgs/epoch{runner.epoch+1}_{os.path.basename(img_path)}'
                    # import pdb; pdb.set_trace()
                    

                    for gt_idx in gt_idx_list:
                    # for gt_idx in valid_instance_indices:

                        related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']  

                        

                        #if index in low_confidence_indices:
                        #if related_global_index in all_classes_low_confidence_scores_global_idx:
                        # if low confidence and not too high confident
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.relabel_conf):
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.5):
                        
                        # if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.3):
                        if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                            if my_counter<400:

                                my_counter+=1                                
                                import shutil

                                # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                                base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                                possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg"]

                                for suffix in possible_suffixes:
                                    
                                    base_debug_path = base_prefix + suffix
                                    if os.path.exists(base_debug_path):
                                        filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                                        # if suffix == "_relabeled.jpg":
                                        #     import pdb; pdb.set_trace()
                                        shutil.copy(base_debug_path, filtered_debug_path)
                                        print(f"[INFO] Cópia criada: {filtered_debug_path}")
                                        break  # Para no primeiro que encontrar 

                                # desenhar_bboxesv3_filtered(all_inputs_map[img_path], sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_filtered.jpg')  
                            # Encontrar `valid_idx` correspondente ao `gt_idx`
                            # if gt_idx in gt_idx_list:
                            #[ME PARECE ERRADO]
                            # valid_idx = valid_instance_indices[gt_idx_list.index(gt_idx)]
                            #[TESTAR ESSE]
                            # import pdb; pdb.set_trace()
                            valid_idx = valid_instance_indices[gt_idx]

                            
                            # import pdb; pdb.set_trace()
                            sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                            # sub_dataset.data_list[dataset_data_idx]['instances'][gt_idx]['ignore_flag'] = 1
                                #print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, GT: {gt_idx}")
                                #bbox = sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox']
                                # print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, BBOX: {bbox} GT: {gt_idx}")
                        
                    
                                
                                
                            
                        # index += 1
                        

            print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")


@HOOKS.register_module()
class MyHookFilterPredGT_Class_DoubleRelabel(Hook):

    def __init__(self, reload_dataset=False, relabel_conf=0.95, filter_conf=0.8, filter_warmup=0, iou_assigner=0.5, low_quality=False, filter_thr=0.7, numGMM=2, filter_type = 'pred', double_thr=0.5):
        """
        Hook personalizado para relabeling de amostras.

        Args:
            reload_dataset (bool): Indica se o dataset deve ser recarregado a cada época.
            my_value (int): Um valor de exemplo para demonstrar parâmetros personalizados.
        """
        self.reload_dataset = reload_dataset
        self.relabel_conf = relabel_conf
        self.filter_conf = filter_conf
        self.filter_warmup = filter_warmup
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        self.filter_thr = filter_thr
        self.numGMM = numGMM
        self.filter_type = filter_type
        self.double_thr = double_thr
        self.group = False
        

    def before_train_epoch(self, runner, *args, **kwargs):
        
        """
        Atualiza os labels de classe (sem alterar os bounding boxes) a cada 100 iterações.
        """
        # import pdb; pdb.set_trace()
        #if (runner.epoch + 1) % 1 != 0:  
        if (runner.epoch + 1) >0 :  
        # if (runner.epoch + 1) >=5 :  

            print(f"[EPOCH]- runner.epoch: {runner.epoch}")
            print(f"[DEBUG] Antes do hook - iter: {runner.iter}")
            dataloader = runner.train_loop.dataloader
            dataset = dataloader.dataset

            #reload_dataset = True
            # my_value = 3
            #reload_dataset = getattr(runner.cfg, 'reload_dataset', False)  
            #my_value =  getattr(runner.cfg, 'my_value', 10)  
            reload_dataset = self.reload_dataset
            relabel_conf = self.relabel_conf
            
            # import pdb; pdb.set_trace()
            if reload_dataset:
                runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

                runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

            # Desencapsular RepeatDataset até o ConcatDataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset

            if not hasattr(dataset, 'datasets'):
                raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

            datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

            assigner = MaxIoUAssigner(
                pos_iou_thr=self.iou_assigner,
                neg_iou_thr=self.iou_assigner,
                min_pos_iou=self.iou_assigner,
                match_low_quality=self.low_quality
            )

            # Criar um dicionário para acesso rápido ao índice das imagens no dataset
            dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                            for sub_dataset_idx, sub_dataset in enumerate(datasets)
                            if hasattr(sub_dataset, 'data_list')
                            for data_idx, data_info in enumerate(sub_dataset.data_list)}
            
            # Dicionário para armazenar os scores médios por GT sem assumir sequência
            allbb_preds_map = defaultdict(dict)
            all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
            all_pred_instances_map = defaultdict(dict)
            # all_inputs_map = defaultdict(dict)

            num_classes = 20
            all_classes_preds_map=[]
            all_classes_gt_idx_map=[]
            for i in range(num_classes):
                all_classes_preds_map.append(defaultdict(list))
                all_classes_gt_idx_map.append(defaultdict(list))
            

            count_gts = 0
            sanity_count = 0
            global_index_counter = 0

            
            
            # Processar o batch de dados
            for batch_idx, data_batch in enumerate(dataloader):
                with torch.no_grad():
                    #data = runner.model.data_preprocessor(data_batch, False)
                    data = runner.model.data_preprocessor(data_batch, True)
                    
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                    
                    #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                    predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
                
                    #temp_filipe

                for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
                # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                    img_path = data_sample.img_path
                    
                    if img_path not in dataset_img_map:
                        print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                        continue

                    # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                    pred_instances = predictions_pred[i].pred_instances

                    # Pegar bounding boxes preditas
                    pred_instances.priors = pred_instances.pop('bboxes')

                    # Obter os ground truths
                    gt_instances = data_sample.gt_instances
                    gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que os GTs estão no mesmo dispositivo das predições
                    gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                    gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que as predições também estão no mesmo dispositivo
                    pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                    pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                    pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                    pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                    all_pred_instances_map[img_path] = pred_instances
                    

                    assign_result = assigner.assign(pred_instances, gt_instances)

                    updated_labels = gt_labels.clone()

                    # Criar lista de GTs para esta imagem
                    all_gt_idx_map[img_path] = []
                    count_gts += assign_result.num_gts

                    
                    
                    
                    # sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    # sub_dataset = datasets[sub_dataset_idx]
                    # import pdb; pdb.set_trace()
                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/sub_{os.path.basename(img_path)}')  
                    # desenhar_bboxesv3(img_path_real,inputs[i], gt_instances, save_path=f'debug_imgs/gt_{os.path.basename(img_path)}')  
                    
                    # if  len(data_samples[i].ignored_instances.labels)>0:
                    #     import pdb; pdb.set_trace()
                    

                    # **Vetorização do processo de associação**
                    for gt_idx in range(assign_result.num_gts):
                        associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                        sanity_count += 1
                        
                        if associated_preds.numel() == 0:
                            continue 

                        # Calcular a média dos scores das bounding boxes associadas a este `gt_idx`
                        # mean_score = pred_instances.scores[associated_preds].mean().item()
                        

                        # max_score = pred_instances.scores[associated_preds].max().item()
                        given_max_score = pred_instances.scores[associated_preds].max().item()
                        # max_logit = pred_instances.logits[associated_preds].max().item()
                        # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                        logits_associated = pred_instances.logits[associated_preds] 
                        myscores = torch.softmax(logits_associated ,dim=-1)

                        
                        # import pdb;pdb.set_trace()
                        # all_pred_instances_map[img_path]['priors']=pred_instances.priors[associated_preds]
                        # all_pred_instances_map[img_path]['scores']=pred_instances.scores[associated_preds]
                        # all_pred_instances_map[img_path]['labels']=pred_instances.labels[associated_preds]
                        # import pdb;pdb.set_trace()
                        # myscores_pred =  myscores.max(dim=1).values.item()

                        # quero pegar o logit_gt da amostra que tem maior logit_pred
                        # import pdb; pdb.set_trace()
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        max_score_val = myscores.max()
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                        # Compute APS
                        row_scores = myscores[amostra_id]  # pega apenas a linha
                        p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                        S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        # p_y = myscores[amostra_id, classe_id].item()
                        # S_APS_score = -myscores[myscores >= p_y].sum().item()

                        # if S_APS_score < -1:
                        #     print(S_APS_score)
                        #     print("a")
                        #     import pdb; pdb.set_trace()
                        #     print("ok")

                            

                        # import pdb; pdb.set_trace()
                        # allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score
                         
                        max_score_idx_local = torch.argmax(myscores_gt)
                        #----> remover depois do debug
                        #max_logit_idx_gt = associated_preds[max_score_idx_local]
                        max_logit_idx_gt = associated_preds[amostra_id]
                        logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()
                        


        
                        gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]

                        #---> logit com maior valor [OFICIAL]-voltar depois do debug
                        # max_logit = gt_logits.max().item()
                        # max_logit_idx_local = torch.argmax(gt_logits)
                        # max_logit_idx = associated_preds[max_logit_idx_local]

                        


                        #logit com maior IoU
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()

                        # Pegar idx local do maior IoU
                        max_iou_values = assign_result.max_overlaps[associated_preds]
                        max_iou_idx_local = torch.argmax(max_iou_values)
                        max_iou_idx = associated_preds[max_iou_idx_local]

                        # Vetor contendo ambos (pode repetir se for o mesmo)
                        important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                        important_associated_ids = torch.unique(important_associated_ids)

                        # import pdb; pdb.set_trace()


                        # gt_logits = [gt_logit[updated_labels[gt_idx]] for gt_logit in logits_associated]
                        
                        # Armazenar `gt_idx` na ordem correta
                        all_gt_idx_map[img_path].append(gt_idx)

                        # Armazenar `mean_score` corretamente
                        #---->[original]
                        # allbb_preds_map[img_path][gt_idx] = {'pred':max_logit, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #---->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx] = {'pred':score_gt_max_pred, 'logit':logit_gt_max_pred , 
                                                             'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                             'aps':S_APS_score,
                                                              'max_pred':max_score_val.cpu().item(), 'pred_label': classe_id,
                                                                'filtered':False}
                        global_index_counter += 1



                        # import pdb; pdb.set_trace()

                        #-->original   
                        #confident_preds =  associated_preds[pred_instances.scores[associated_preds] > relabel_conf]
                        #-->temporario-filipe-debug - remover depois
                        # import pdb; pdb.set_trace()

                        

                        confident_preds =  associated_preds[myscores.max(dim=1).values> relabel_conf]
                        

                        
                        if (batch_idx <200):
                            
                            if confident_preds.numel() == 0:
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg'
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg')  
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                # pass
                            else:
                                
                                # for _, label_t, score_t, inter, logits_t in zip(pred_instances['priors'][important_associated_ids], pred_instances['labels'][important_associated_ids], pred_instances['scores'][important_associated_ids],assign_result.max_overlaps[important_associated_ids], pred_instances['logits'][important_associated_ids]):
                                #     label_temp = int(label_t)
                                #     score_temp = float(score_t)
                                #     gt_pred_temp = updated_labels[gt_idx].cpu().item()
                                #     myscores_temp = torch.softmax(logits_t ,dim=-1)
                                #     myscores_pred_temp =  myscores_temp.max(dim=0).values.item()
                                #     mylabel_pred_temp = myscores_temp.argmax(dim=0).item()
                                #     myscore_gt_temp = myscores_temp[gt_pred_temp].item()
                                #     if myscores_pred_temp <0.7:
                                #         import pdb; pdb.set_trace
                                    # if (label_temp == gt_pred_temp) and myscore_gt_temp!=myscores_pred_temp:
                                    #     import pdb; pdb.set_trace() 
                                # pass
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg')  
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg'
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                            

                        
                        if self.group and len(associated_preds) > 1:
                            labels_group = myscores.argmax(dim=1)
                            # print(labels_group)
                            # import pdb; pdb.set_trace()
                            #most_common_label = Counter(labels_group.tolist()).most_common(1)[0][0]
                            most_common_label, qtd = Counter(labels_group.tolist()).most_common(1)[0]
                            # print(most_common_label, qtd)
                            # encontra o elemento mais comum e sua quantidade
                            # most_common, qtd = Counter.most_common(1)[0]

                            # verifica se a quantidade é maior que 50% do total
                            if qtd > len(associated_preds) / 2:
                                most_common_label = most_common_label
                            # senao verifica se tem algum elemento com score maior que o threshold
                            elif confident_preds.numel() > 0:
                                pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)
                                most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]
                            else:
                                continue          

                        elif confident_preds.numel() > 0:
                            # if (runner.epoch + 1 >= 0) and (i <100):
                            #     import pdb; pdb.set_trace()
                            #     img_path_real = img_path
                            #     # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/sub_{os.path.basename(img_path)}')  
                                # desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}.jpg')  
                            

                            #---> original
                            # pred_labels_confident = pred_instances.labels[confident_preds]
                            #---> temporario-filipe-debug - remover depois do debug
                            #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                            pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                            most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        elif confident_preds.numel() == 0:
                            continue  

                        
                        
                        

                        updated_labels[gt_idx] = most_common_label
                        #atualiza também no mapeamento
                        allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                        #atualiza max logit-gt
                        gt_logits = logits_associated[:,most_common_label]
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                        max_logit = gt_logits.max().item()

                        ###-->debug -remover depois
                        # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        # max_score_val = myscores.max()
                        # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                        # Compute APS
                        # p_y = myscores[amostra_id, classe_id].item()
                        # S_APS_score = -myscores[myscores >= p_y].sum().item()
                        # row_scores = myscores[amostra_id]  # pega apenas a linha
                        # p_y = row_scores[classe_id].item()
                        # S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        row_scores = myscores[amostra_id]  # pega apenas a linha
                        p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                        S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        # if S_APS_score < -1:
                        #     print(S_APS_score)
                        #     print("a")
                        #     import pdb; pdb.set_trace()
                        #     print("ok")
                        allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score

                        #
                        #-->[original]
                        #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                        #-->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                        allbb_preds_map[img_path][gt_idx]['filtered'] = True
                        
                        

                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                    valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                    # Atualizar os labels corretamente usando o mapeamento
                    for gt_idx, valid_idx in enumerate(valid_instance_indices):
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                        # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
            

            # só faz filter depois do warmup
            if (runner.epoch + 1) >= self.filter_warmup:

                # filtering
                #Unir todas as probabilidades das imagens e treinar o GMM**
                # all_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values()
                # ]).reshape(-1, 1)
                # import pdb; pdb.set_trace()
                print("[DEBUG1]: Entrou no filtro")
                all_classes_low_confidence_scores_global_idx = []

                for c in range(num_classes):
                    # c_class_scores = np.array([
                    #     score['pred'] for img_scores in allbb_preds_map.values()
                    #     for score in img_scores.values() if score['gt_label'] == c
                    # ]).reshape(-1, 1)

                    # if runner.epoch + 1 >1:
                    #     import pdb; pdb.set_trace()

                    scores = np.array([])
                    # img_indexes = np.array([])
                    img_paths = np.array([], dtype=object)
                    c_global_indexes = np.array([])

                    for img_path, img_info in allbb_preds_map.items():
                        #for pred, gt_label, index  in img_info.values():
                        for temp_gt_idx, values in img_info.items():
                            #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                            #if gt_label== c:
                            if values['gt_label'] == c:
                                
                                if self.filter_type == 'pred':
                                    scores = np.append(scores, values['pred'])
                                elif self.filter_type == 'logit':
                                    scores = np.append(scores, values['logit'])
                                elif self.filter_type == 'aps':
                                    scores = np.append(scores, values['aps'])
                                # img_indexes = np.append(img_indexes, values['global_index_counter'])
                                c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                                if img_path not in img_paths:
                                    img_paths = np.append(img_paths, img_path)
                    # if runner.epoch + 1 >1:
                    print(scores)
                    # import pdb; pdb.set_trace()
                    if len(scores) == 0:
                        print(f"Aviso: Nenhuma amostra encontrada para a classe {c}.")
                        continue
                    # print("[DEBUG1.5]: INICIANDO GMM")
                    c_class_scores = scores.reshape(-1, 1)
                                
                    
                    #gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    gmm = GaussianMixture(n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    gmm.fit(c_class_scores)

                    # Identificar o cluster com menor média (baixa confiança)
                    low_confidence_component = np.argmin(gmm.means_)
                    low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]
                    threshold = self.filter_conf
                
                    low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                    all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                    # Valor do ponto de corte nos scores (máximo dos scores das amostras de baixa confiança)
                    # Tratamento para array vazio:
                    if len(low_confidence_indices) > 0:
                        score_cutoff = np.max(c_class_scores[low_confidence_indices])
                    else:
                        print(f"Aviso: Nenhuma amostra com confiança > {threshold} encontrada.")
                        score_cutoff = np.min(c_class_scores)  # valor alternativo seguro

                    ## save import pickle
                    # import pickle
                    # with open(f"c{c}_scores_ep{runner.epoch + 1}.pkl", "wb") as f:
                    #     pickle.dump(c_class_scores, f)
                    #end saving

                    import matplotlib.pyplot as plt

                    # Dentro do seu for c in range(num_classes):
                    # import pdb; pdb.set_trace()
                    # ... após o fit do GMM
                    plt.figure()
                    plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                    # Adiciona linha vertical que separa as duas classes com base no GMM
                    plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                    # plt.legend()
                    plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                    plt.xlabel('Score')
                    plt.ylabel('Frequência')
                    plt.grid(True)
                    plt.tight_layout()
                    # Criar pasta, se necessário
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                    plt.savefig(save_path)
                    plt.close()
                   

                # print("[DEBUG2]: SAIU do GMM")
                # if len(all_scores) > 0:
                    

                # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
                # index = 0
                # import pdb; pdb.set_trace()
                my_counter = 0
                for img_path, gt_scores in allbb_preds_map.items():
                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    

                    valid_instance_indices = [
                        idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                        if inst['ignore_flag'] == 0
                    ]

                    # **Mapear GTs reais na ordem correta**
                    gt_idx_list = all_gt_idx_map[img_path]  

                    # import pdb; pdb.set_trace()

                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path)}')  
                    # save_path = f'debug_imgs/epoch{runner.epoch+1}_{os.path.basename(img_path)}'
                    # import pdb; pdb.set_trace()
                    

                    for gt_idx in gt_idx_list:
                    # for gt_idx in valid_instance_indices:

                        related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']  

                        

                        #if index in low_confidence_indices:
                        #if related_global_index in all_classes_low_confidence_scores_global_idx:
                        # if low confidence and not too high confident
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.relabel_conf):
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.5):
                        
                        # if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.3):
                        if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                            if my_counter<400:

                                my_counter+=1                                
                                import shutil

                                # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                                
                                base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                                possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg"]

                                for suffix in possible_suffixes:
                                    
                                    base_debug_path = base_prefix + suffix
                                    if os.path.exists(base_debug_path):
                                        filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                                        # if suffix == "_relabeled.jpg":
                                        #     import pdb; pdb.set_trace()
                                        shutil.copy(base_debug_path, filtered_debug_path)
                                        print(f"[INFO] Cópia criada: {filtered_debug_path}")
                                        break  # Para no primeiro que encontrar 

                                # desenhar_bboxesv3_filtered(all_inputs_map[img_path], sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_filtered.jpg')  
                            # Encontrar `valid_idx` correspondente ao `gt_idx`
                            # if gt_idx in gt_idx_list:
                            #[ME PARECE ERRADO]
                            # valid_idx = valid_instance_indices[gt_idx_list.index(gt_idx)]
                            #[TESTAR ESSE]
                            # import pdb; pdb.set_trace()
                            valid_idx = valid_instance_indices[gt_idx]

                           # self.double_thr
                            if allbb_preds_map[img_path][gt_idx]['max_pred'] >= self.double_thr:
                                #update
                                sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = allbb_preds_map[img_path][gt_idx]['pred_label']

                            else:    
                                #filtra
                                sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                                # sub_dataset.data_list[dataset_data_idx]['instances'][gt_idx]['ignore_flag'] = 1
                                    #print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, GT: {gt_idx}")
                                    #bbox = sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox']
                                    # print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, BBOX: {bbox} GT: {gt_idx}")
                        
                    
                                
                                
                            
                        # index += 1
                        

            print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")

@HOOKS.register_module()
class MyHookFilterLOSS_Class_Relabel(Hook):

    def __init__(self, reload_dataset=False, relabel_conf=0.95, filter_conf=0.8, filter_warmup=0, iou_assigner=0.5, low_quality=False):
        """
        Hook personalizado para relabeling de amostras.

        Args:
            reload_dataset (bool): Indica se o dataset deve ser recarregado a cada época.
            my_value (int): Um valor de exemplo para demonstrar parâmetros personalizados.
        """
        self.reload_dataset = reload_dataset
        self.relabel_conf = relabel_conf
        self.filter_conf = filter_conf
        self.filter_warmup = filter_warmup
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        

    def before_train_epoch(self, runner, *args, **kwargs):
        
        """
        Atualiza os labels de classe (sem alterar os bounding boxes) a cada 100 iterações.
        """

        #if (runner.epoch + 1) % 1 != 0:  
        if (runner.epoch + 1) >0 :  
        # if (runner.epoch + 1) >=5 :  

            print(f"[EPOCH]- runner.epoch: {runner.epoch}")
            print(f"[DEBUG] Antes do hook - iter: {runner.iter}")
            dataloader = runner.train_loop.dataloader
            dataset = dataloader.dataset

            #reload_dataset = True
            # my_value = 3
            #reload_dataset = getattr(runner.cfg, 'reload_dataset', False)  
            #my_value =  getattr(runner.cfg, 'my_value', 10)  
            reload_dataset = self.reload_dataset
            relabel_conf = self.relabel_conf
            
            # import pdb; pdb.set_trace()
            if reload_dataset:
                runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

                runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

            # Desencapsular RepeatDataset até o ConcatDataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset

            if not hasattr(dataset, 'datasets'):
                raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

            datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

            # assigner = MaxIoUAssigner(
            #     pos_iou_thr=0.5,
            #     neg_iou_thr=0.5,
            #     min_pos_iou=0.5,
            #     match_low_quality=False
            # )

            assigner = MaxIoUAssigner(
                pos_iou_thr=self.iou_assigner,
                neg_iou_thr=self.iou_assigner,
                min_pos_iou=self.iou_assigner,
                match_low_quality=self.low_quality
            )

            # Criar um dicionário para acesso rápido ao índice das imagens no dataset
            dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                            for sub_dataset_idx, sub_dataset in enumerate(datasets)
                            if hasattr(sub_dataset, 'data_list')
                            for data_idx, data_info in enumerate(sub_dataset.data_list)}
            
            # Dicionário para armazenar os scores médios por GT sem assumir sequência
            allbb_preds_map = defaultdict(dict)
            all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais

            num_classes = 20
            all_classes_preds_map=[]
            all_classes_gt_idx_map=[]
            for i in range(num_classes):
                all_classes_preds_map.append(defaultdict(list))
                all_classes_gt_idx_map.append(defaultdict(list))
            

            count_gts = 0
            sanity_count = 0
            global_index_counter = 0
            
            # Processar o batch de dados
            for batch_idx, data_batch in enumerate(dataloader):
                with torch.no_grad():
                    #data = runner.model.data_preprocessor(data_batch, False)
                    data = runner.model.data_preprocessor(data_batch, True)
                    
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                    
                    #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                    predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
                
                for i, data_sample in enumerate(data_batch['data_samples']):
                    img_path = data_sample.img_path
                    
                    if img_path not in dataset_img_map:
                        print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                        continue

                    # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                    pred_instances = predictions_pred[i].pred_instances

                    # Pegar bounding boxes preditas
                    pred_instances.priors = pred_instances.pop('bboxes')

                    # Obter os ground truths
                    gt_instances = data_sample.gt_instances
                    gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que os GTs estão no mesmo dispositivo das predições
                    gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                    gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que as predições também estão no mesmo dispositivo
                    pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                    pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                    pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                    pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                    assign_result = assigner.assign(pred_instances, gt_instances)

                    updated_labels = gt_labels.clone()

                    # Criar lista de GTs para esta imagem
                    all_gt_idx_map[img_path] = []
                    count_gts += assign_result.num_gts
                    
                    

                    # **Vetorização do processo de associação**
                    for gt_idx in range(assign_result.num_gts):
                        associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                        sanity_count += 1
                        
                        if associated_preds.numel() == 0:
                            continue 

                        
                        # Calcular a média dos scores das bounding boxes associadas a este `gt_idx`
                        # mean_score = pred_instances.scores[associated_preds].mean().item()
                        max_score = pred_instances.scores[associated_preds].max().item()
                        logits_associated = pred_instances.logits[associated_preds]  # shape: (N, num_classes)

                        # import pdb;pdb.set_trace()
                        target_labels = gt_instances.labels[gt_idx].expand(logits_associated.size(0))  # shape: (N,)
                        # import pdb; pdb.set_trace()
                        # Calcular a CE loss para cada predição associada
                        ce_losses = F.cross_entropy(logits_associated, target_labels, reduction='none')  # shape: (N,)

                        # import pdb; pdb.set_trace()
                        mean_ce_loss = ce_losses.mean().item()
                        min_ce_loss = ce_losses.min().item()
                        
                        # Armazenar `gt_idx` na ordem correta
                        all_gt_idx_map[img_path].append(gt_idx)

                        # Armazenar `mean_score` corretamente
                        # allbb_preds_map[img_path][gt_idx] = mean_score
                        #allbb_preds_map[img_path][gt_idx] = max_score
                        # allbb_preds_map[img_path][gt_idx] = {'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        allbb_preds_map[img_path][gt_idx] = {'pred':min_ce_loss, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        global_index_counter += 1
                            
                        confident_preds =  associated_preds[pred_instances.scores[associated_preds] > relabel_conf]

                        if confident_preds.numel() == 0:
                            continue  

                        pred_labels_confident = pred_instances.labels[confident_preds]
                        most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        updated_labels[gt_idx] = most_common_label
                        #atualiza também no mapeamento
                        allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                        

                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                    valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                    # Atualizar os labels corretamente usando o mapeamento
                    for gt_idx, valid_idx in enumerate(valid_instance_indices):
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                        # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
            

            # só faz filter depois do warmup
            if (runner.epoch + 1) >= self.filter_warmup:

                # filtering
                #Unir todas as probabilidades das imagens e treinar o GMM**
                # all_scores = np.array([
                #     score['pred'] for img_scoresf in allbb_preds_map.values()
                #     for score in img_scores.values()
                # ]).reshape(-1, 1)
                # import pdb; pdb.set_trace()
                all_classes_low_confidence_scores_global_idx = []

                for c in range(num_classes):
                    # c_class_scores = np.array([
                    #     score['pred'] for img_scores in allbb_preds_map.values()
                    #     for score in img_scores.values() if score['gt_label'] == c
                    # ]).reshape(-1, 1)
                    # import pdb; pdb.set_trace()

                    scores = np.array([])
                    # img_indexes = np.array([])
                    img_paths = np.array([], dtype=object)
                    c_global_indexes = np.array([])

                    for img_path, img_info in allbb_preds_map.items():
                        #for pred, gt_label, index  in img_info.values():
                        for temp_gt_idx, values in img_info.items():
                            #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                            #if gt_label== c:
                            if values['gt_label'] == c:
                                scores = np.append(scores, values['pred'])
                                # img_indexes = np.append(img_indexes, values['global_index_counter'])
                                c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                                if img_path not in img_paths:
                                    img_paths = np.append(img_paths, img_path)
                    c_class_scores = scores.reshape(-1, 1)
                                

                    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    gmm.fit(c_class_scores)

                    # Identificar o cluster com menor média (baixa confiança)
                    #low_confidence_component = np.argmin(gmm.means_)
                    low_confidence_component = np.argmax(gmm.means_)
                    low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]
                    threshold = self.filter_conf
                
                    low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                    all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                    ## save import pickle
                    import pickle
                    with open(f"c{c}_scores_ep{runner.epoch + 1}.pkl", "wb") as f:
                        pickle.dump(c_class_scores, f)
                    #end saving
                   


                # if len(all_scores) > 0:
                    

                # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
                # index = 0
                # import pdb; pdb.set_trace()
                for img_path, gt_scores in allbb_preds_map.items():
                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    

                    valid_instance_indices = [
                        idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                        if inst['ignore_flag'] == 0
                    ]

                    # **Mapear GTs reais na ordem correta**
                    gt_idx_list = all_gt_idx_map[img_path]  

                    # import pdb; pdb.set_trace()

                    for gt_idx in gt_idx_list:
                    # for gt_idx in valid_instance_indices:

                        related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']    

                        #if index in low_confidence_indices:
                        if related_global_index in all_classes_low_confidence_scores_global_idx:
                            # Encontrar `valid_idx` correspondente ao `gt_idx`
                            # if gt_idx in gt_idx_list:
                            valid_idx = valid_instance_indices[gt_idx_list.index(gt_idx)]
                            sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                            # sub_dataset.data_list[dataset_data_idx]['instances'][gt_idx]['ignore_flag'] = 1
                                #print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, GT: {gt_idx}")
                                #bbox = sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox']
                                # print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, BBOX: {bbox} GT: {gt_idx}")

                        # index += 1
                        

            print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")

@HOOKS.register_module()
class MyHookFilterPredGT_Class_AdaptiveRelabel(Hook):

    def __init__(self, reload_dataset=False, relabel_conf=0.95, relabel_conf2= 0.8, filter_conf=0.8, filter_warmup=0, iou_assigner=0.5, low_quality=False, filter_thr=0.7, numGMM=2, filter_type = 'pred'):
        """
        Hook personalizado para relabeling de amostras.

        Args:
            reload_dataset (bool): Indica se o dataset deve ser recarregado a cada época.
            my_value (int): Um valor de exemplo para demonstrar parâmetros personalizados.
        """
        self.reload_dataset = reload_dataset
        self.relabel_conf = relabel_conf
        self.filter_conf = filter_conf
        self.filter_warmup = filter_warmup
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        self.filter_thr = filter_thr
        self.numGMM = numGMM
        self.filter_type = filter_type
        self.relabel_conf2 = relabel_conf2
        

    def before_train_epoch(self, runner, *args, **kwargs):
        
        """
        Atualiza os labels de classe (sem alterar os bounding boxes) a cada 100 iterações.
        """
        
        
            

        if (runner.epoch + 1) >0 :  
        # if (runner.epoch + 1) >=5 :  

            print(f"[EPOCH]- runner.epoch: {runner.epoch}")
            print(f"[DEBUG] Antes do hook - iter: {runner.iter}")
            dataloader = runner.train_loop.dataloader
            dataset = dataloader.dataset

            #reload_dataset = True
            # my_value = 3
            #reload_dataset = getattr(runner.cfg, 'reload_dataset', False)  
            #my_value =  getattr(runner.cfg, 'my_value', 10)  
            reload_dataset = self.reload_dataset
            if (runner.epoch + 1) >= 5:
                relabel_conf = self.relabel_conf2
            else:
                relabel_conf = self.relabel_conf
            
            # import pdb; pdb.set_trace()
            if reload_dataset:
                runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

                runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

            # Desencapsular RepeatDataset até o ConcatDataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset

            if not hasattr(dataset, 'datasets'):
                raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

            datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

            assigner = MaxIoUAssigner(
                pos_iou_thr=self.iou_assigner,
                neg_iou_thr=self.iou_assigner,
                min_pos_iou=self.iou_assigner,
                match_low_quality=self.low_quality
            )

            # Criar um dicionário para acesso rápido ao índice das imagens no dataset
            dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                            for sub_dataset_idx, sub_dataset in enumerate(datasets)
                            if hasattr(sub_dataset, 'data_list')
                            for data_idx, data_info in enumerate(sub_dataset.data_list)}
            
            # Dicionário para armazenar os scores médios por GT sem assumir sequência
            allbb_preds_map = defaultdict(dict)
            all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
            all_pred_instances_map = defaultdict(dict)
            # all_inputs_map = defaultdict(dict)

            num_classes = 20
            all_classes_preds_map=[]
            all_classes_gt_idx_map=[]
            for i in range(num_classes):
                all_classes_preds_map.append(defaultdict(list))
                all_classes_gt_idx_map.append(defaultdict(list))
            

            count_gts = 0
            sanity_count = 0
            global_index_counter = 0

            
            
            # Processar o batch de dados
            for batch_idx, data_batch in enumerate(dataloader):
                with torch.no_grad():
                    #data = runner.model.data_preprocessor(data_batch, False)
                    data = runner.model.data_preprocessor(data_batch, True)
                    
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                    
                    #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                    predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
                
                    #temp_filipe

                for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
                # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                    img_path = data_sample.img_path
                    
                    if img_path not in dataset_img_map:
                        print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                        continue

                    # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                    pred_instances = predictions_pred[i].pred_instances

                    # Pegar bounding boxes preditas
                    pred_instances.priors = pred_instances.pop('bboxes')

                    # Obter os ground truths
                    gt_instances = data_sample.gt_instances
                    gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que os GTs estão no mesmo dispositivo das predições
                    gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                    gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que as predições também estão no mesmo dispositivo
                    pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                    pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                    pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                    pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                    all_pred_instances_map[img_path] = pred_instances
                    

                    assign_result = assigner.assign(pred_instances, gt_instances)

                    updated_labels = gt_labels.clone()

                    # Criar lista de GTs para esta imagem
                    all_gt_idx_map[img_path] = []
                    count_gts += assign_result.num_gts

                    
                    
                    
                    # sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    # sub_dataset = datasets[sub_dataset_idx]
                    # import pdb; pdb.set_trace()
                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/sub_{os.path.basename(img_path)}')  
                    # desenhar_bboxesv3(img_path_real,inputs[i], gt_instances, save_path=f'debug_imgs/gt_{os.path.basename(img_path)}')  
                    
                    # if  len(data_samples[i].ignored_instances.labels)>0:
                    #     import pdb; pdb.set_trace()
                    

                    # **Vetorização do processo de associação**
                    for gt_idx in range(assign_result.num_gts):
                        associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                        sanity_count += 1
                        
                        if associated_preds.numel() == 0:
                            continue 

                        # Calcular a média dos scores das bounding boxes associadas a este `gt_idx`
                        # mean_score = pred_instances.scores[associated_preds].mean().item()
                        

                        # max_score = pred_instances.scores[associated_preds].max().item()
                        given_max_score = pred_instances.scores[associated_preds].max().item()
                        # max_logit = pred_instances.logits[associated_preds].max().item()
                        # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                        logits_associated = pred_instances.logits[associated_preds] 
                        myscores = torch.softmax(logits_associated ,dim=-1)

                        
                        # import pdb;pdb.set_trace()
                        # all_pred_instances_map[img_path]['priors']=pred_instances.priors[associated_preds]
                        # all_pred_instances_map[img_path]['scores']=pred_instances.scores[associated_preds]
                        # all_pred_instances_map[img_path]['labels']=pred_instances.labels[associated_preds]
                        # import pdb;pdb.set_trace()
                        # myscores_pred =  myscores.max(dim=1).values.item()

                        # quero pegar o logit_gt da amostra que tem maior logit_pred
                        # import pdb; pdb.set_trace()
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        max_score_val = myscores.max()
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()
                         
                        max_score_idx_local = torch.argmax(myscores_gt)
                        #----> remover depois do debug
                        #max_logit_idx_gt = associated_preds[max_score_idx_local]
                        max_logit_idx_gt = associated_preds[amostra_id]
                        logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()
                        


        
                        gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]

                        #---> logit com maior valor [OFICIAL]-voltar depois do debug
                        # max_logit = gt_logits.max().item()
                        # max_logit_idx_local = torch.argmax(gt_logits)
                        # max_logit_idx = associated_preds[max_logit_idx_local]

                        


                        #logit com maior IoU
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()

                        # Pegar idx local do maior IoU
                        max_iou_values = assign_result.max_overlaps[associated_preds]
                        max_iou_idx_local = torch.argmax(max_iou_values)
                        max_iou_idx = associated_preds[max_iou_idx_local]

                        # Vetor contendo ambos (pode repetir se for o mesmo)
                        important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                        important_associated_ids = torch.unique(important_associated_ids)

                        # import pdb; pdb.set_trace()


                        # gt_logits = [gt_logit[updated_labels[gt_idx]] for gt_logit in logits_associated]
                        
                        # Armazenar `gt_idx` na ordem correta
                        all_gt_idx_map[img_path].append(gt_idx)

                        # Armazenar `mean_score` corretamente
                        #---->[original]
                        # allbb_preds_map[img_path][gt_idx] = {'pred':max_logit, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #---->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx] = {'pred':score_gt_max_pred, 'logit':logit_gt_max_pred , 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 'filtered':False}
                        global_index_counter += 1



                        # import pdb; pdb.set_trace()

                        #-->original   
                        #confident_preds =  associated_preds[pred_instances.scores[associated_preds] > relabel_conf]
                        #-->temporario-filipe-debug - remover depois
                        # import pdb; pdb.set_trace()
                        confident_preds =  associated_preds[myscores.max(dim=1).values> relabel_conf]

                        
                        if (batch_idx <200):
                            
                            if confident_preds.numel() == 0:
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg'
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg')  
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                # pass
                            else:
                                
                                # for _, label_t, score_t, inter, logits_t in zip(pred_instances['priors'][important_associated_ids], pred_instances['labels'][important_associated_ids], pred_instances['scores'][important_associated_ids],assign_result.max_overlaps[important_associated_ids], pred_instances['logits'][important_associated_ids]):
                                #     label_temp = int(label_t)
                                #     score_temp = float(score_t)
                                #     gt_pred_temp = updated_labels[gt_idx].cpu().item()
                                #     myscores_temp = torch.softmax(logits_t ,dim=-1)
                                #     myscores_pred_temp =  myscores_temp.max(dim=0).values.item()
                                #     mylabel_pred_temp = myscores_temp.argmax(dim=0).item()
                                #     myscore_gt_temp = myscores_temp[gt_pred_temp].item()
                                #     if myscores_pred_temp <0.7:
                                #         import pdb; pdb.set_trace
                                    # if (label_temp == gt_pred_temp) and myscore_gt_temp!=myscores_pred_temp:
                                    #     import pdb; pdb.set_trace() 
                                # pass
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg')  
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg'
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                            


                        if confident_preds.numel() == 0:
                            continue  

                        
                        
                        # if (runner.epoch + 1 >= 0) and (i <100):
                        #     import pdb; pdb.set_trace()
                        #     img_path_real = img_path
                        #     # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/sub_{os.path.basename(img_path)}')  
                            # desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}.jpg')  
                        

                        #---> original
                        # pred_labels_confident = pred_instances.labels[confident_preds]
                        #---> temporario-filipe-debug - remover depois do debug
                        #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                        pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                        most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        updated_labels[gt_idx] = most_common_label
                        #atualiza também no mapeamento
                        allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                        #atualiza max logit-gt
                        gt_logits = logits_associated[:,most_common_label]
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                        max_logit = gt_logits.max().item()

                        ###-->debug -remover depois
                        # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        # max_score_val = myscores.max()
                        # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                        #
                        #-->[original]
                        #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                        #-->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                        allbb_preds_map[img_path][gt_idx]['filtered'] = True
                        
                        

                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                    valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                    # Atualizar os labels corretamente usando o mapeamento
                    for gt_idx, valid_idx in enumerate(valid_instance_indices):
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                        # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
            

            # só faz filter depois do warmup
            if (runner.epoch + 1) >= self.filter_warmup:

                # filtering
                #Unir todas as probabilidades das imagens e treinar o GMM**
                # all_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values()
                # ]).reshape(-1, 1)
                # import pdb; pdb.set_trace()
                print("[DEBUG1]: Entrou no filtro")
                all_classes_low_confidence_scores_global_idx = []

                for c in range(num_classes):
                    # c_class_scores = np.array([
                    #     score['pred'] for img_scores in allbb_preds_map.values()
                    #     for score in img_scores.values() if score['gt_label'] == c
                    # ]).reshape(-1, 1)

                    # if runner.epoch + 1 >1:
                    #     import pdb; pdb.set_trace()

                    scores = np.array([])
                    # img_indexes = np.array([])
                    img_paths = np.array([], dtype=object)
                    c_global_indexes = np.array([])

                    for img_path, img_info in allbb_preds_map.items():
                        #for pred, gt_label, index  in img_info.values():
                        for temp_gt_idx, values in img_info.items():
                            #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                            #if gt_label== c:
                            if values['gt_label'] == c:
                                
                                if self.filter_type == 'pred':
                                    scores = np.append(scores, values['pred'])
                                elif self.filter_type == 'logit':
                                    scores = np.append(scores, values['logit'])
                                # img_indexes = np.append(img_indexes, values['global_index_counter'])
                                c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                                if img_path not in img_paths:
                                    img_paths = np.append(img_paths, img_path)
                    # if runner.epoch + 1 >1:
                    #     import pdb; pdb.set_trace()
                    # print("[DEBUG1.5]: INICIANDO GMM")
                    c_class_scores = scores.reshape(-1, 1)
                                
                    
                    #gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    gmm = GaussianMixture(n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    gmm.fit(c_class_scores)

                    # Identificar o cluster com menor média (baixa confiança)
                    low_confidence_component = np.argmin(gmm.means_)
                    low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]
                    threshold = self.filter_conf
                
                    low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                    all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                    # Valor do ponto de corte nos scores (máximo dos scores das amostras de baixa confiança)
                    # Tratamento para array vazio:
                    if len(low_confidence_indices) > 0:
                        score_cutoff = np.max(c_class_scores[low_confidence_indices])
                    else:
                        print(f"Aviso: Nenhuma amostra com confiança > {threshold} encontrada.")
                        score_cutoff = np.min(c_class_scores)  # valor alternativo seguro

                    ## save import pickle
                    # import pickle
                    # with open(f"c{c}_scores_ep{runner.epoch + 1}.pkl", "wb") as f:
                    #     pickle.dump(c_class_scores, f)
                    #end saving

                    import matplotlib.pyplot as plt

                    # Dentro do seu for c in range(num_classes):
                    # import pdb; pdb.set_trace()
                    # ... após o fit do GMM
                    plt.figure()
                    plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                    # Adiciona linha vertical que separa as duas classes com base no GMM
                    plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                    # plt.legend()
                    plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                    plt.xlabel('Score')
                    plt.ylabel('Frequência')
                    plt.grid(True)
                    plt.tight_layout()
                    # Criar pasta, se necessário
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                    plt.savefig(save_path)
                    plt.close()
                   

                # print("[DEBUG2]: SAIU do GMM")
                # if len(all_scores) > 0:
                    

                # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
                # index = 0
                # import pdb; pdb.set_trace()
                my_counter = 0
                for img_path, gt_scores in allbb_preds_map.items():
                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    

                    valid_instance_indices = [
                        idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                        if inst['ignore_flag'] == 0
                    ]

                    # **Mapear GTs reais na ordem correta**
                    gt_idx_list = all_gt_idx_map[img_path]  

                    # import pdb; pdb.set_trace()

                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path)}')  
                    # save_path = f'debug_imgs/epoch{runner.epoch+1}_{os.path.basename(img_path)}'
                    # import pdb; pdb.set_trace()
                    

                    for gt_idx in gt_idx_list:
                    # for gt_idx in valid_instance_indices:

                        related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']  

                        

                        #if index in low_confidence_indices:
                        #if related_global_index in all_classes_low_confidence_scores_global_idx:
                        # if low confidence and not too high confident
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.relabel_conf):
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.5):
                        
                        # if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.3):
                        if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                            if my_counter<400:

                                my_counter+=1                                
                                import shutil

                                # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                                
                                base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                                possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg"]

                                for suffix in possible_suffixes:
                                    
                                    base_debug_path = base_prefix + suffix
                                    if os.path.exists(base_debug_path):
                                        filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                                        # if suffix == "_relabeled.jpg":
                                        #     import pdb; pdb.set_trace()
                                        shutil.copy(base_debug_path, filtered_debug_path)
                                        print(f"[INFO] Cópia criada: {filtered_debug_path}")
                                        break  # Para no primeiro que encontrar 

                                # desenhar_bboxesv3_filtered(all_inputs_map[img_path], sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_filtered.jpg')  
                            # Encontrar `valid_idx` correspondente ao `gt_idx`
                            # if gt_idx in gt_idx_list:
                            #[ME PARECE ERRADO]
                            # valid_idx = valid_instance_indices[gt_idx_list.index(gt_idx)]
                            #[TESTAR ESSE]
                            # import pdb; pdb.set_trace()
                            valid_idx = valid_instance_indices[gt_idx]

                            
                            # import pdb; pdb.set_trace()
                            sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                            # sub_dataset.data_list[dataset_data_idx]['instances'][gt_idx]['ignore_flag'] = 1
                                #print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, GT: {gt_idx}")
                                #bbox = sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox']
                                # print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, BBOX: {bbox} GT: {gt_idx}")
                        
                    
                                
                                
                            
                        # index += 1
                        

            print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")

@HOOKS.register_module()
class MyHookFilterLM_Class_Relabel(Hook):

    def __init__(self, reload_dataset=False, relabel_conf=0.95, filter_conf=0.8, filter_warmup=0, iou_assigner=0.5, low_quality=False, filter_thr=0.7, numGMM=2, filter_type = 'pred'):
        """
        Hook personalizado para relabeling de amostras.

        Args:
            reload_dataset (bool): Indica se o dataset deve ser recarregado a cada época.
            my_value (int): Um valor de exemplo para demonstrar parâmetros personalizados.
        """
        self.reload_dataset = reload_dataset
        self.relabel_conf = relabel_conf
        self.filter_conf = filter_conf
        self.filter_warmup = filter_warmup
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        self.filter_thr = filter_thr
        self.numGMM = numGMM
        self.filter_type = filter_type
        

    def before_train_epoch(self, runner, *args, **kwargs):
        
        """
        Atualiza os labels de classe (sem alterar os bounding boxes) a cada 100 iterações.
        """
        
        if (runner.epoch + 1) >0 :  

            print(f"[EPOCH]- runner.epoch: {runner.epoch}")
            print(f"[DEBUG] Antes do hook - iter: {runner.iter}")
            dataloader = runner.train_loop.dataloader
            dataset = dataloader.dataset

            reload_dataset = self.reload_dataset
            relabel_conf = self.relabel_conf
            
            if reload_dataset:
                runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

                runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

            # Desencapsular RepeatDataset até o ConcatDataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset

            if not hasattr(dataset, 'datasets'):
                raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

            datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

            assigner = MaxIoUAssigner(
                pos_iou_thr=self.iou_assigner,
                neg_iou_thr=self.iou_assigner,
                min_pos_iou=self.iou_assigner,
                match_low_quality=self.low_quality
            )

            # Criar um dicionário para acesso rápido ao índice das imagens no dataset
            dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                            for sub_dataset_idx, sub_dataset in enumerate(datasets)
                            if hasattr(sub_dataset, 'data_list')
                            for data_idx, data_info in enumerate(sub_dataset.data_list)}
            
            # Dicionário para armazenar os scores médios por GT sem assumir sequência
            allbb_preds_map = defaultdict(dict)
            all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
            all_pred_instances_map = defaultdict(dict)
            # all_inputs_map = defaultdict(dict)

            num_classes = 20
            all_classes_preds_map=[]
            all_classes_gt_idx_map=[]
            for i in range(num_classes):
                all_classes_preds_map.append(defaultdict(list))
                all_classes_gt_idx_map.append(defaultdict(list))
            
            count_gts = 0
            sanity_count = 0
            global_index_counter = 0

            # Processar o batch de dados
            for batch_idx, data_batch in enumerate(dataloader):
                with torch.no_grad():
                    #data = runner.model.data_preprocessor(data_batch, False)
                    data = runner.model.data_preprocessor(data_batch, True)
                    
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                    
                    #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                    predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
                
                    

                for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
                # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                    img_path = data_sample.img_path
                    
                    if img_path not in dataset_img_map:
                        print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                        continue

                    # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                    pred_instances = predictions_pred[i].pred_instances

                    # Pegar bounding boxes preditas
                    pred_instances.priors = pred_instances.pop('bboxes')

                    # Obter os ground truths
                    gt_instances = data_sample.gt_instances
                    gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que os GTs estão no mesmo dispositivo das predições
                    gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                    gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que as predições também estão no mesmo dispositivo
                    pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                    pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                    pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                    pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                    all_pred_instances_map[img_path] = pred_instances
                    

                    assign_result = assigner.assign(pred_instances, gt_instances)

                    updated_labels = gt_labels.clone()

                    # Criar lista de GTs para esta imagem
                    all_gt_idx_map[img_path] = []
                    count_gts += assign_result.num_gts
                    
                    # **Vetorização do processo de associação**
                    for gt_idx in range(assign_result.num_gts):
                        associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                        sanity_count += 1
                        
                        if associated_preds.numel() == 0:
                            continue 

                        # Calcular a média dos scores das bounding boxes associadas a este `gt_idx`
                        # mean_score = pred_instances.scores[associated_preds].mean().item()
                        

                        # max_score = pred_instances.scores[associated_preds].max().item()
                        given_max_score = pred_instances.scores[associated_preds].max().item()
                        # max_logit = pred_instances.logits[associated_preds].max().item()
                        # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                        logits_associated = pred_instances.logits[associated_preds] 
                        myscores = torch.softmax(logits_associated ,dim=-1)

                        
                        # import pdb;pdb.set_trace()
                        # all_pred_instances_map[img_path]['priors']=pred_instances.priors[associated_preds]
                        # all_pred_instances_map[img_path]['scores']=pred_instances.scores[associated_preds]
                        # all_pred_instances_map[img_path]['labels']=pred_instances.labels[associated_preds]
                        # import pdb;pdb.set_trace()
                        # myscores_pred =  myscores.max(dim=1).values.item()

                        # quero pegar o logit_gt da amostra que tem maior logit_pred
                        # import pdb; pdb.set_trace()
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        max_score_val = myscores.max()
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()
                         
                        max_score_idx_local = torch.argmax(myscores_gt)
                        #----> remover depois do debug
                        #max_logit_idx_gt = associated_preds[max_score_idx_local]
                        max_logit_idx_gt = associated_preds[amostra_id]
                        logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()

                        #Loss Margin
                        # Índice da classe associada
                        gt_class = updated_labels[gt_idx].cpu().item()

                        # Cria uma máscara que é True para todas as classes, exceto a ground-truth
                        mask = torch.ones_like(logits_associated, dtype=bool)
                        mask[:, gt_class] = False  # zera a coluna da classe correta

                        # Máximo logit dentre as classes erradas
                        max_wrong_logits, _ = logits_associated.masked_select(mask).reshape(logits_associated.size(0), -1).max(dim=1)
                        # LM = logit da classe correta - máximo logit incorreto
                        LMs = mylogits_gt - max_wrong_logits  # shape: (N,)
                        max_LM = LMs.max().item()
                        idx_max_LM = torch.argmax(LMs).item()

                                
                        gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]

                        
                        # Pegar idx local do maior IoU
                        max_iou_values = assign_result.max_overlaps[associated_preds]
                        max_iou_idx_local = torch.argmax(max_iou_values)
                        max_iou_idx = associated_preds[max_iou_idx_local]

                        # Vetor contendo ambos (pode repetir se for o mesmo)
                        important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                        important_associated_ids = torch.unique(important_associated_ids)

                        # gt_logits = [gt_logit[updated_labels[gt_idx]] for gt_logit in logits_associated]
                        
                        # Armazenar `gt_idx` na ordem correta
                        all_gt_idx_map[img_path].append(gt_idx)

                        # Armazenar `mean_score` corretamente
                        #---->[original]
                        # allbb_preds_map[img_path][gt_idx] = {'pred':max_logit, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #---->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx] = {'pred':score_gt_max_pred, 'logit':logit_gt_max_pred , 'lm':max_LM, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 'filtered':False}
                        global_index_counter += 1



                        # import pdb; pdb.set_trace()

                        #-->original   
                        #confident_preds =  associated_preds[pred_instances.scores[associated_preds] > relabel_conf]
                        #-->temporario-filipe-debug - remover depois
                        # import pdb; pdb.set_trace()
                        confident_preds =  associated_preds[myscores.max(dim=1).values> relabel_conf]

                        
                        if (batch_idx <200):
                            
                            if confident_preds.numel() == 0:
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg'
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg')  
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                # pass
                            else:
                                
                                # for _, label_t, score_t, inter, logits_t in zip(pred_instances['priors'][important_associated_ids], pred_instances['labels'][important_associated_ids], pred_instances['scores'][important_associated_ids],assign_result.max_overlaps[important_associated_ids], pred_instances['logits'][important_associated_ids]):
                                #     label_temp = int(label_t)
                                #     score_temp = float(score_t)
                                #     gt_pred_temp = updated_labels[gt_idx].cpu().item()
                                #     myscores_temp = torch.softmax(logits_t ,dim=-1)
                                #     myscores_pred_temp =  myscores_temp.max(dim=0).values.item()
                                #     mylabel_pred_temp = myscores_temp.argmax(dim=0).item()
                                #     myscore_gt_temp = myscores_temp[gt_pred_temp].item()
                                #     if myscores_pred_temp <0.7:
                                #         import pdb; pdb.set_trace
                                    # if (label_temp == gt_pred_temp) and myscore_gt_temp!=myscores_pred_temp:
                                    #     import pdb; pdb.set_trace() 
                                # pass
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg')  
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg'
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                            


                        if confident_preds.numel() == 0:
                            continue  

                        
                        
                        # if (runner.epoch + 1 >= 0) and (i <100):
                        #     import pdb; pdb.set_trace()
                        #     img_path_real = img_path
                        #     # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/sub_{os.path.basename(img_path)}')  
                            # desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}.jpg')  
                        

                        #---> original
                        # pred_labels_confident = pred_instances.labels[confident_preds]
                        #---> temporario-filipe-debug - remover depois do debug
                        #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                        pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                        most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        updated_labels[gt_idx] = most_common_label
                        #atualiza também no mapeamento
                        allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                        #atualiza max logit-gt
                        gt_logits = logits_associated[:,most_common_label]
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                        max_logit = gt_logits.max().item()

                        ###-->debug -remover depois
                        # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        # max_score_val = myscores.max()
                        # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                        gt_class = updated_labels[gt_idx].cpu().item()

                        # Cria uma máscara que é True para todas as classes, exceto a ground-truth
                        mask = torch.ones_like(logits_associated, dtype=bool)
                        mask[:, gt_class] = False  # zera a coluna da classe correta

                        # Máximo logit dentre as classes erradas
                        max_wrong_logits, _ = logits_associated.masked_select(mask).reshape(logits_associated.size(0), -1).max(dim=1)
                        # LM = logit da classe correta - máximo logit incorreto
                        LMs = mylogits_gt - max_wrong_logits  # shape: (N,)
                        max_LM = LMs.max().item()
                        idx_max_LM = torch.argmax(LMs).item()
                        allbb_preds_map[img_path][gt_idx]['lm'] = max_LM

                        #
                        #-->[original]
                        #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                        #-->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                        allbb_preds_map[img_path][gt_idx]['filtered'] = True
                        
                        

                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                    valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                    # Atualizar os labels corretamente usando o mapeamento
                    for gt_idx, valid_idx in enumerate(valid_instance_indices):
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                        # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
            

            # só faz filter depois do warmup
            if (runner.epoch + 1) >= self.filter_warmup:

                # filtering
                #Unir todas as probabilidades das imagens e treinar o GMM**
                # all_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values()
                # ]).reshape(-1, 1)
                # import pdb; pdb.set_trace()
                # print("[DEBUG1]: Entrou no filtro")
                all_classes_low_confidence_scores_global_idx = []

                for c in range(num_classes):
                    # c_class_scores = np.array([
                    #     score['pred'] for img_scores in allbb_preds_map.values()
                    #     for score in img_scores.values() if score['gt_label'] == c
                    # ]).reshape(-1, 1)

                    # if runner.epoch + 1 >1:
                    #     import pdb; pdb.set_trace()

                    scores = np.array([])
                    # img_indexes = np.array([])
                    img_paths = np.array([], dtype=object)
                    c_global_indexes = np.array([])

                    for img_path, img_info in allbb_preds_map.items():
                        #for pred, gt_label, index  in img_info.values():
                        for temp_gt_idx, values in img_info.items():
                            #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                            #if gt_label== c:
                            if values['gt_label'] == c:
                                
                                if self.filter_type == 'pred':
                                    scores = np.append(scores, values['pred'])
                                elif self.filter_type == 'logit':
                                    scores = np.append(scores, values['logit'])
                                elif self.filter_type == 'lm':
                                    scores = np.append(scores, values['lm'])
                                # img_indexes = np.append(img_indexes, values['global_index_counter'])
                                c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                                if img_path not in img_paths:
                                    img_paths = np.append(img_paths, img_path)
                    # if runner.epoch + 1 >1:
                    #     import pdb; pdb.set_trace()
                    # print("[DEBUG1.5]: INICIANDO GMM")
                    c_class_scores = scores.reshape(-1, 1)
                                
                    
                    #gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    gmm = GaussianMixture(n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    gmm.fit(c_class_scores)

                    # Identificar o cluster com menor média (baixa confiança)
                    low_confidence_component = np.argmin(gmm.means_)
                    low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]
                    threshold = self.filter_conf
                
                    low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                    all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                    # Valor do ponto de corte nos scores (máximo dos scores das amostras de baixa confiança)
                    # Tratamento para array vazio:
                    if len(low_confidence_indices) > 0:
                        score_cutoff = np.max(c_class_scores[low_confidence_indices])
                    else:
                        print(f"Aviso: Nenhuma amostra com confiança > {threshold} encontrada.")
                        score_cutoff = np.min(c_class_scores)  # valor alternativo seguro

                    ## save import pickle
                    # import pickle
                    # with open(f"c{c}_scores_ep{runner.epoch + 1}.pkl", "wb") as f:
                    #     pickle.dump(c_class_scores, f)
                    #end saving

                    import matplotlib.pyplot as plt

                    # Dentro do seu for c in range(num_classes):
                    # import pdb; pdb.set_trace()
                    # ... após o fit do GMM
                    plt.figure()
                    plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                    # Adiciona linha vertical que separa as duas classes com base no GMM
                    plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                    # plt.legend()
                    plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                    plt.xlabel('Score')
                    plt.ylabel('Frequência')
                    plt.grid(True)
                    plt.tight_layout()
                    # Criar pasta, se necessário
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                    plt.savefig(save_path)
                    plt.close()
                   

                # print("[DEBUG2]: SAIU do GMM")
                # if len(all_scores) > 0:
                    

                # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
                # index = 0
                # import pdb; pdb.set_trace()
                my_counter = 0
                for img_path, gt_scores in allbb_preds_map.items():
                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    

                    valid_instance_indices = [
                        idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                        if inst['ignore_flag'] == 0
                    ]

                    # **Mapear GTs reais na ordem correta**
                    gt_idx_list = all_gt_idx_map[img_path]  

                    # import pdb; pdb.set_trace()

                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path)}')  
                    # save_path = f'debug_imgs/epoch{runner.epoch+1}_{os.path.basename(img_path)}'
                    # import pdb; pdb.set_trace()
                    

                    for gt_idx in gt_idx_list:
                    # for gt_idx in valid_instance_indices:

                        related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']  

                        

                        #if index in low_confidence_indices:
                        #if related_global_index in all_classes_low_confidence_scores_global_idx:
                        # if low confidence and not too high confident
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.relabel_conf):
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.5):
                        
                        # if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.3):
                        if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                            if my_counter<400:

                                my_counter+=1                                
                                import shutil

                                # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                                
                                base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                                possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg"]

                                for suffix in possible_suffixes:
                                    
                                    base_debug_path = base_prefix + suffix
                                    if os.path.exists(base_debug_path):
                                        filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                                        # if suffix == "_relabeled.jpg":
                                        #     import pdb; pdb.set_trace()
                                        shutil.copy(base_debug_path, filtered_debug_path)
                                        print(f"[INFO] Cópia criada: {filtered_debug_path}")
                                        break  # Para no primeiro que encontrar 

                                # desenhar_bboxesv3_filtered(all_inputs_map[img_path], sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_filtered.jpg')  
                            # Encontrar `valid_idx` correspondente ao `gt_idx`
                            # if gt_idx in gt_idx_list:
                            #[ME PARECE ERRADO]
                            # valid_idx = valid_instance_indices[gt_idx_list.index(gt_idx)]
                            #[TESTAR ESSE]
                            # import pdb; pdb.set_trace()
                            valid_idx = valid_instance_indices[gt_idx]

                            
                            # import pdb; pdb.set_trace()
                            sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                            # sub_dataset.data_list[dataset_data_idx]['instances'][gt_idx]['ignore_flag'] = 1
                                #print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, GT: {gt_idx}")
                                #bbox = sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox']
                                # print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, BBOX: {bbox} GT: {gt_idx}")
                        
                    
                                
                                
                            
                        # index += 1
                        

            print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")

@HOOKS.register_module()
class MyHookFilterKNN_Class_Relabel(Hook):

    def __init__(self, reload_dataset=False, relabel_conf=0.95, filter_conf=0.8, filter_warmup=0, iou_assigner=0.5, low_quality=False, filter_thr=0.7, numGMM=2, filter_type = 'pred',kvalue=100, double_thr=1.1):
        """
        Hook personalizado para relabeling de amostras.

        Args:
            reload_dataset (bool): Indica se o dataset deve ser recarregado a cada época.
            my_value (int): Um valor de exemplo para demonstrar parâmetros personalizados.
        """
        self.reload_dataset = reload_dataset
        self.relabel_conf = relabel_conf
        self.filter_conf = filter_conf
        self.filter_warmup = filter_warmup
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        self.filter_thr = filter_thr
        self.numGMM = numGMM
        self.filter_type = filter_type
        self.kvalue = kvalue
        self.double_thr = double_thr
        

    def before_train_epoch(self, runner, *args, **kwargs):
        
        """
        Atualiza os labels de classe (sem alterar os bounding boxes) a cada 100 iterações.
        """
        
        if (runner.epoch + 1) >0 :  

            print(f"[EPOCH]- runner.epoch: {runner.epoch}")
            print(f"[DEBUG] Antes do hook - iter: {runner.iter}")
            dataloader = runner.train_loop.dataloader
            dataset = dataloader.dataset

            reload_dataset = self.reload_dataset
            relabel_conf = self.relabel_conf
            
            if reload_dataset:
                runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

                runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

            # Desencapsular RepeatDataset até o ConcatDataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset

            if not hasattr(dataset, 'datasets'):
                raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

            datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

            assigner = MaxIoUAssigner(
                pos_iou_thr=self.iou_assigner,
                neg_iou_thr=self.iou_assigner,
                min_pos_iou=self.iou_assigner,
                match_low_quality=self.low_quality
            )

            # Criar um dicionário para acesso rápido ao índice das imagens no dataset
            dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                            for sub_dataset_idx, sub_dataset in enumerate(datasets)
                            if hasattr(sub_dataset, 'data_list')
                            for data_idx, data_info in enumerate(sub_dataset.data_list)}
            
            # Dicionário para armazenar os scores médios por GT sem assumir sequência
            allbb_preds_map = defaultdict(dict)
            all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
            all_pred_instances_map = defaultdict(dict)
            # all_inputs_map = defaultdict(dict)

            num_classes = 20
            all_classes_preds_map=[]
            all_classes_gt_idx_map=[]
            for i in range(num_classes):
                all_classes_preds_map.append(defaultdict(list))
                all_classes_gt_idx_map.append(defaultdict(list))
            
            count_gts = 0
            sanity_count = 0
            global_index_counter = 0

            # Processar o batch de dados
            for batch_idx, data_batch in enumerate(dataloader):
                with torch.no_grad():
                    #data = runner.model.data_preprocessor(data_batch, False)
                    data = runner.model.data_preprocessor(data_batch, True)
                    
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                    
                    #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                    # import pdb; pdb.set_trace()
                    #predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
                    predictions_pred = runner.model.my_get_features_logits(inputs, data_samples,all_logits=True)
                    
                    

                for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
                # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                    img_path = data_sample.img_path
                    
                    if img_path not in dataset_img_map:
                        print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                        continue

                    # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                    pred_instances = predictions_pred[i].pred_instances

                    # Pegar bounding boxes preditas
                    pred_instances.priors = pred_instances.pop('bboxes')

                    # Obter os ground truths
                    gt_instances = data_sample.gt_instances
                    gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que os GTs estão no mesmo dispositivo das predições
                    gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                    gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que as predições também estão no mesmo dispositivo
                    pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                    pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                    pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                    pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)
                    # import pdb; pdb.set_trace()
                    pred_instances.feats =  pred_instances.feat.to(pred_instances.priors.device)
                    
                    all_pred_instances_map[img_path] = pred_instances
                    

                    assign_result = assigner.assign(pred_instances, gt_instances)

                    updated_labels = gt_labels.clone()

                    # Criar lista de GTs para esta imagem
                    all_gt_idx_map[img_path] = []
                    count_gts += assign_result.num_gts
                    
                    # **Vetorização do processo de associação**
                    for gt_idx in range(assign_result.num_gts):
                        associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                        sanity_count += 1
                        
                        if associated_preds.numel() == 0:
                            continue 

                        # Calcular a média dos scores das bounding boxes associadas a este `gt_idx`
                        # mean_score = pred_instances.scores[associated_preds].mean().item()
                        

                        # max_score = pred_instances.scores[associated_preds].max().item()
                        given_max_score = pred_instances.scores[associated_preds].max().item()
                        # max_logit = pred_instances.logits[associated_preds].max().item()
                        # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                        logits_associated = pred_instances.logits[associated_preds] 
                        myscores = torch.softmax(logits_associated ,dim=-1)
                        feats_associated = pred_instances.feats[associated_preds]

                        
                        # import pdb;pdb.set_trace()
                        # all_pred_instances_map[img_path]['priors']=pred_instances.priors[associated_preds]
                        # all_pred_instances_map[img_path]['scores']=pred_instances.scores[associated_preds]
                        # all_pred_instances_map[img_path]['labels']=pred_instances.labels[associated_preds]
                        # import pdb;pdb.set_trace()
                        # myscores_pred =  myscores.max(dim=1).values.item()

                        # quero pegar o logit_gt da amostra que tem maior logit_pred
                        # import pdb; pdb.set_trace()
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        max_score_val = myscores.max()
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()
                        feat_max_score = feats_associated[amostra_id].cpu().numpy()
                         
                        max_score_idx_local = torch.argmax(myscores_gt)
                        #----> remover depois do debug
                        #max_logit_idx_gt = associated_preds[max_score_idx_local]
                        max_logit_idx_gt = associated_preds[amostra_id]
                        logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()

                        #Loss Margin
                        # Índice da classe associada
                        gt_class = updated_labels[gt_idx].cpu().item()

                        # Cria uma máscara que é True para todas as classes, exceto a ground-truth
                        mask = torch.ones_like(logits_associated, dtype=bool)
                        mask[:, gt_class] = False  # zera a coluna da classe correta

                        # Máximo logit dentre as classes erradas
                        max_wrong_logits, _ = logits_associated.masked_select(mask).reshape(logits_associated.size(0), -1).max(dim=1)
                        # LM = logit da classe correta - máximo logit incorreto
                        LMs = mylogits_gt - max_wrong_logits  # shape: (N,)
                        max_LM = LMs.max().item()
                        idx_max_LM = torch.argmax(LMs).item()

                                
                        gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]

                        
                        # Pegar idx local do maior IoU
                        max_iou_values = assign_result.max_overlaps[associated_preds]
                        max_iou_idx_local = torch.argmax(max_iou_values)
                        max_iou_idx = associated_preds[max_iou_idx_local]

                        # Vetor contendo ambos (pode repetir se for o mesmo)
                        important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                        important_associated_ids = torch.unique(important_associated_ids)

                        # gt_logits = [gt_logit[updated_labels[gt_idx]] for gt_logit in logits_associated]
                        
                        # Armazenar `gt_idx` na ordem correta
                        all_gt_idx_map[img_path].append(gt_idx)

                        # Armazenar `mean_score` corretamente
                        #---->[original]
                        # allbb_preds_map[img_path][gt_idx] = {'pred':max_logit, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #---->[temporario-filipe-debug] - remover depois do debug
                        # import pdb; pdb.set_trace()
                        allbb_preds_map[img_path][gt_idx] = {'pred':score_gt_max_pred, 'max_pred_model': max_score_val, 'logit':logit_gt_max_pred , 'lm':max_LM, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 'feat_max_score':feat_max_score,  'pred_label': classe_id, 'filtered':False}
                        global_index_counter += 1



                        # import pdb; pdb.set_trace()

                        #-->original   
                        #confident_preds =  associated_preds[pred_instances.scores[associated_preds] > relabel_conf]
                        #-->temporario-filipe-debug - remover depois
                        # import pdb; pdb.set_trace()
                        confident_preds =  associated_preds[myscores.max(dim=1).values> relabel_conf]

                        
                        if (batch_idx <200):
                            
                            if confident_preds.numel() == 0:
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg'
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg')  
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                # pass
                            else:
                                
                                # for _, label_t, score_t, inter, logits_t in zip(pred_instances['priors'][important_associated_ids], pred_instances['labels'][important_associated_ids], pred_instances['scores'][important_associated_ids],assign_result.max_overlaps[important_associated_ids], pred_instances['logits'][important_associated_ids]):
                                #     label_temp = int(label_t)
                                #     score_temp = float(score_t)
                                #     gt_pred_temp = updated_labels[gt_idx].cpu().item()
                                #     myscores_temp = torch.softmax(logits_t ,dim=-1)
                                #     myscores_pred_temp =  myscores_temp.max(dim=0).values.item()
                                #     mylabel_pred_temp = myscores_temp.argmax(dim=0).item()
                                #     myscore_gt_temp = myscores_temp[gt_pred_temp].item()
                                #     if myscores_pred_temp <0.7:
                                #         import pdb; pdb.set_trace
                                    # if (label_temp == gt_pred_temp) and myscore_gt_temp!=myscores_pred_temp:
                                    #     import pdb; pdb.set_trace() 
                                # pass
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg')  
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg'
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                            


                        if confident_preds.numel() == 0:
                            continue  

                        
                        
                        # if (runner.epoch + 1 >= 0) and (i <100):
                        #     import pdb; pdb.set_trace()
                        #     img_path_real = img_path
                        #     # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/sub_{os.path.basename(img_path)}')  
                            # desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}.jpg')  
                        

                        #---> original
                        # pred_labels_confident = pred_instances.labels[confident_preds]
                        #---> temporario-filipe-debug - remover depois do debug
                        #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                        pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                        most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        updated_labels[gt_idx] = most_common_label
                        #atualiza também no mapeamento
                        allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                        #atualiza max logit-gt
                        gt_logits = logits_associated[:,most_common_label]
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                        max_logit = gt_logits.max().item()

                        ###-->debug -remover depois
                        # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        # max_score_val = myscores.max()
                        # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                        gt_class = updated_labels[gt_idx].cpu().item()

                        # Cria uma máscara que é True para todas as classes, exceto a ground-truth
                        mask = torch.ones_like(logits_associated, dtype=bool)
                        mask[:, gt_class] = False  # zera a coluna da classe correta

                        # Máximo logit dentre as classes erradas
                        max_wrong_logits, _ = logits_associated.masked_select(mask).reshape(logits_associated.size(0), -1).max(dim=1)
                        # LM = logit da classe correta - máximo logit incorreto
                        LMs = mylogits_gt - max_wrong_logits  # shape: (N,)
                        max_LM = LMs.max().item()
                        idx_max_LM = torch.argmax(LMs).item()
                        allbb_preds_map[img_path][gt_idx]['lm'] = max_LM

                        #
                        #-->[original]
                        #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                        #-->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                        allbb_preds_map[img_path][gt_idx]['filtered'] = True
                        
                        

                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                    valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                    # Atualizar os labels corretamente usando o mapeamento
                    for gt_idx, valid_idx in enumerate(valid_instance_indices):
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                        # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
            

            # só faz filter depois do warmup
            if (runner.epoch + 1) >= self.filter_warmup:

                # filtering
                #Unir todas as probabilidades das imagens e treinar o GMM**
                # all_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values()
                # ]).reshape(-1, 1)
                # import pdb; pdb.set_trace()
                # print("[DEBUG1]: Entrou no filtro")
                all_classes_low_confidence_scores_global_idx = []
                global_indices = []
                # import pdb; pdb.set_trace()

                #calculate feature_banka
                feature_bank = []
                bank_labels = []

                for img_path, img_info in allbb_preds_map.items():
                    
                    for temp_gt_idx, values in img_info.items():
                        feature = values['feat_max_score']
                        temp_gt_class = values['gt_label']
                        # feature_bank.append(feature)
                        # feature_bank.append(torch.tensor(feature, device='cuda'))
                        feature_bank.append(torch.tensor(feature, device='cuda').unsqueeze(0))
                        bank_labels.append(temp_gt_class)
                        global_indices.append(values['global_index_counter'])
                
                feature_bank = F.normalize(torch.cat(feature_bank, dim=0), dim=1)
                # import pdb; pdb.set_trace()
                prediction_knn = weighted_knn(feature_bank, feature_bank, bank_labels, num_classes, self.kvalue, 10)  # temperature in weighted KNN
                # import pdb; pdb.set_trace()
                bank_labels = torch.tensor(bank_labels, device=prediction_knn.device)
                vote_y = torch.gather(prediction_knn, 1, bank_labels.view(-1, 1)).squeeze()
                vote_max = prediction_knn.max(dim=1)[0]
                right_score = vote_y / vote_max

                theta_s=1

                clean_id = torch.where(right_score >= theta_s)[0]
                noisy_id = torch.where(right_score < theta_s)[0]
                global_indices = torch.tensor(global_indices, device=noisy_id.device)
                global_indices_noisy = global_indices[noisy_id]

                # import pdb; pdb.set_trace()

                # for c in range(num_classes):
                #     # c_class_scores = np.array([
                #     #     score['pred'] for img_scores in allbb_preds_map.values()
                #     #     for score in img_scores.values() if score['gt_label'] == c
                #     # ]).reshape(-1, 1)

                #     # if runner.epoch + 1 >1:
                #     #     import pdb; pdb.set_trace()

                #     scores = np.array([])
                #     # img_indexes = np.array([])
                #     img_paths = np.array([], dtype=object)
                #     c_global_indexes = np.array([])

                #     for img_path, img_info in allbb_preds_map.items():
                #         #for pred, gt_label, index  in img_info.values():
                #         for temp_gt_idx, values in img_info.items():
                #             #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                #             #if gt_label== c:
                #             if values['gt_label'] == c:
                                
                #                 if self.filter_type == 'pred':
                #                     scores = np.append(scores, values['pred'])
                #                 elif self.filter_type == 'logit':
                #                     scores = np.append(scores, values['logit'])
                #                 elif self.filter_type == 'lm':
                #                     scores = np.append(scores, values['lm'])
                #                 # img_indexes = np.append(img_indexes, values['global_index_counter'])
                #                 c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                #                 if img_path not in img_paths:
                #                     img_paths = np.append(img_paths, img_path)
                #     # if runner.epoch + 1 >1:
                #     #     import pdb; pdb.set_trace()
                #     # print("[DEBUG1.5]: INICIANDO GMM")
                #     c_class_scores = scores.reshape(-1, 1)
                                
                    
                #     #gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                #     gmm = GaussianMixture(n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                #     gmm.fit(c_class_scores)

                #     # Identificar o cluster com menor média (baixa confiança)
                #     low_confidence_component = np.argmin(gmm.means_)
                #     low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]
                #     threshold = self.filter_conf
                
                #     low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                #     all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                #     # Valor do ponto de corte nos scores (máximo dos scores das amostras de baixa confiança)
                #     # Tratamento para array vazio:
                #     if len(low_confidence_indices) > 0:
                #         score_cutoff = np.max(c_class_scores[low_confidence_indices])
                #     else:
                #         print(f"Aviso: Nenhuma amostra com confiança > {threshold} encontrada.")
                #         score_cutoff = np.min(c_class_scores)  # valor alternativo seguro

                #     ## save import pickle
                #     # import pickle
                #     # with open(f"c{c}_scores_ep{runner.epoch + 1}.pkl", "wb") as f:
                #     #     pickle.dump(c_class_scores, f)
                #     #end saving

                #     import matplotlib.pyplot as plt

                #     # Dentro do seu for c in range(num_classes):
                #     # import pdb; pdb.set_trace()
                #     # ... após o fit do GMM
                #     plt.figure()
                #     plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                #     # Adiciona linha vertical que separa as duas classes com base no GMM
                #     plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                #     # plt.legend()
                #     plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                #     plt.xlabel('Score')
                #     plt.ylabel('Frequência')
                #     plt.grid(True)
                #     plt.tight_layout()
                #     # Criar pasta, se necessário
                #     save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
                #     #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                #     plt.savefig(save_path)
                #     plt.close()
                   

                
                # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
                # index = 0
                # import pdb; pdb.set_trace()
                my_counter = 0
                for img_path, gt_scores in allbb_preds_map.items():
                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    valid_instance_indices = [
                        idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                        if inst['ignore_flag'] == 0
                    ]

                    # **Mapear GTs reais na ordem correta**
                    gt_idx_list = all_gt_idx_map[img_path]  

                    # import pdb; pdb.set_trace()

                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path)}')  
                    # save_path = f'debug_imgs/epoch{runner.epoch+1}_{os.path.basename(img_path)}'
                    # import pdb; pdb.set_trace()
                    

                    for gt_idx in gt_idx_list:
                    # for gt_idx in valid_instance_indices:

                        related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']  

                        
                        #if index in low_confidence_indices:
                        #if related_global_index in all_classes_low_confidence_scores_global_idx:
                        # if low confidence and not too high confident
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.relabel_conf):
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.5):
                        
                        # if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.3):
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                        if (related_global_index in global_indices_noisy) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                            if my_counter<400:

                                my_counter+=1                                
                                import shutil

                                # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                                
                                base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                                possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg"]

                                for suffix in possible_suffixes:
                                    
                                    base_debug_path = base_prefix + suffix
                                    if os.path.exists(base_debug_path):
                                        filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                                        # if suffix == "_relabeled.jpg":
                                        #     import pdb; pdb.set_trace()
                                        shutil.copy(base_debug_path, filtered_debug_path)
                                        print(f"[INFO] Cópia criada: {filtered_debug_path}")
                                        break  # Para no primeiro que encontrar 

                                # desenhar_bboxesv3_filtered(all_inputs_map[img_path], sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_filtered.jpg')  
                            # Encontrar `valid_idx` correspondente ao `gt_idx`
                            # if gt_idx in gt_idx_list:
                            #[ME PARECE ERRADO]
                            # valid_idx = valid_instance_indices[gt_idx_list.index(gt_idx)]
                            #[TESTAR ESSE]
                            # import pdb; pdb.set_trace()
                            valid_idx = valid_instance_indices[gt_idx]

                            if allbb_preds_map[img_path][gt_idx]['max_pred_model'] >= self.double_thr:
                                #update
                                sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = allbb_preds_map[img_path][gt_idx]['pred_label']
                            else:    
                                #filtra
                                sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1

                            
                            # import pdb; pdb.set_trace()
                            # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                            # sub_dataset.data_list[dataset_data_idx]['instances'][gt_idx]['ignore_flag'] = 1
                                #print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, GT: {gt_idx}")
                                #bbox = sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox']
                                # print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, BBOX: {bbox} GT: {gt_idx}")
                        
                    
                                
                                
                            
                        # index += 1
                        

            print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")


# @HOOKS.register_module()
# class MyHookIter(Hook):

#     def before_train_iter(self, runner, batch_idx, data_batch):
#         # import pdb; pdb.set_trace()
#         #data_batch['data_samples'][0].gt_instances
#         #runner.train_dataloader.dataset.fully_initialized=False
#         #runner.train_dataloader.dataset.dataset._fully_initialized=False
#         #runner.train_dataloader.dataset.dataset.full_init()


# @HOOKS.register_module()
# class MyHookAfterEp(Hook):

#     def after_train_epoch(self, runner, batch_idx, data_batch, outputs):
#         # import pdb; pdb.set_trace()
        # (If you find: valid_instance_indices[gt_idx], replace with:
        # inst_all = sub_dataset.data_list[dataset_data_idx]['instances']
        # if gt_idx >= len(inst_all):
        #     print(f"[MAP][WARN] gt_idx={gt_idx} out of range for instances (len={len(inst_all)}) img={os.path.basename(img_path)}")
        #     continue
        # valid_idx = gt_idx
        # )
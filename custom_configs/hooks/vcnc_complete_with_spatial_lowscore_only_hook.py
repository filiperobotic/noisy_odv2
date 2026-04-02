"""
Estratégia B: Relabeling apenas para score_gt muito baixo + Spatial Refinement

Diferença em relação ao VCNC original:
- Relabela APENAS quando score_gt < 0.05 (ruído simétrico claro)
- Filtra quando score_gt entre 0.05-0.3 (pode ser assimétrico)
- Mantém quando score_gt > 0.3 (provavelmente correto)
- Spatial Refinement mantido igual ao original
"""

from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import torch
from mmdet.models.task_modules.assigners import MaxIoUAssigner
from collections import Counter, defaultdict
import numpy as np
from sklearn.mixture import GaussianMixture

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    from sklearn.cluster import KMeans

def unwrap_to_leaf_datasets(dataset):
    """
    Retorna uma lista com os datasets 'folha', independentemente de o
    dataset estar encapsulado em RepeatDataset, ConcatDataset etc.
    """
    datasets = [dataset]

    changed = True
    while changed:
        changed = False
        new_datasets = []

        for ds in datasets:
            if hasattr(ds, 'datasets'):   # ConcatDataset
                new_datasets.extend(ds.datasets)
                changed = True
            elif hasattr(ds, 'dataset'):  # RepeatDataset e wrappers similares
                new_datasets.append(ds.dataset)
                changed = True
            else:                         # dataset folha, ex: VOCDataset
                new_datasets.append(ds)

        datasets = new_datasets

    return datasets


def reload_leaf_datasets(dataset):
    """
    Recarrega todos os datasets folha.
    """
    leaf_datasets = unwrap_to_leaf_datasets(dataset)

    for ds in leaf_datasets:
        if hasattr(ds, '_fully_initialized'):
            ds._fully_initialized = False
        if hasattr(ds, 'full_init'):
            ds.full_init()


def compute_box_difficulty(box_i, all_boxes, box_i_idx=None):
    """Calcula dificuldade de um box baseado em contaminação espacial."""
    difficulty = 0.0
    area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
    
    if area_i <= 0:
        return 0.0
    
    for j, box_j in enumerate(all_boxes):
        if box_i_idx is not None and j == box_i_idx:
            continue
        
        x1_inter = max(box_i[0].item(), box_j[0].item())
        y1_inter = max(box_i[1].item(), box_j[1].item())
        x2_inter = min(box_i[2].item(), box_j[2].item())
        y2_inter = min(box_i[3].item(), box_j[3].item())
        
        inter_w = max(0, x2_inter - x1_inter)
        inter_h = max(0, y2_inter - y1_inter)
        inter_area = inter_w * inter_h
        
        if inter_area <= 0:
            continue
        
        area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
        
        if area_j <= 0:
            continue
        
        iou = inter_area / (area_i + area_j - inter_area)
        
        if iou > 0.0:
            if area_j > area_i:
                contamination = iou * min(area_j / area_i, 2.0)
                difficulty += min(contamination, 1.0)
            
            if iou > 0.5:
                difficulty += 0.5
    
    return min(difficulty, 1.0)


def spatial_aware_relabeling(boxes, pred_labels, pred_scores, difficulty_threshold=0.5):
    """Refinamento de labels para boxes com alta contaminação espacial."""
    refined_labels = pred_labels.clone()
    refinements = 0
    
    for i, box_i in enumerate(boxes):
        difficulty = compute_box_difficulty(box_i, boxes, box_i_idx=i)
        
        if difficulty < difficulty_threshold:
            continue
        
        contaminators = []
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        
        for j, box_j in enumerate(boxes):
            if i == j:
                continue
            
            x1_inter = max(box_i[0].item(), box_j[0].item())
            y1_inter = max(box_i[1].item(), box_j[1].item())
            x2_inter = min(box_i[2].item(), box_j[2].item())
            y2_inter = min(box_i[3].item(), box_j[3].item())
            
            inter_w = max(0, x2_inter - x1_inter)
            inter_h = max(0, y2_inter - y1_inter)
            inter_area = inter_w * inter_h
            
            if inter_area <= 0:
                continue
            
            area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
            iou = inter_area / (area_i + area_j - inter_area)
            
            if iou > 0.3 and area_j > area_i:
                influence = iou * (area_j / area_i)
                contaminators.append((j, iou, area_j / area_i, influence))
        
        if len(contaminators) == 0:
            continue
        
        contaminators.sort(key=lambda x: x[3], reverse=True)
        biggest_contaminator_idx = contaminators[0][0]
        biggest_contaminator_label = pred_labels[biggest_contaminator_idx]
        
        if pred_labels[i] == biggest_contaminator_label:
            top2_scores, top2_labels = pred_scores[i].topk(2)
            
            if len(top2_scores) >= 2 and top2_scores[1] > top2_scores[0] * 0.3:
                refined_labels[i] = top2_labels[1]
                refinements += 1
    
    return refined_labels, refinements


@HOOKS.register_module()
class VCNCRelabelLowScoreOnlyHook(Hook):
    """
    VCNC Estratégia B: Relabeling apenas para score_gt muito baixo
    
    Lógica:
    - score_gt < 0.05: Relabela (provavelmente ruído simétrico)
    - score_gt entre 0.05 e 0.3: Filtra (pode ser assimétrico)
    - score_gt > 0.3: Mantém (provavelmente correto)
    - Spatial Refinement: mantido igual ao original
    """
    
    def __init__(self,
                 warmup_epochs: int = 1,
                 num_classes: int = 20,
                 
                 # Thresholds de score_gt para decisão
                 relabel_max_score_gt: float = 0.05,  # Abaixo disso: relabela
                 filter_max_score_gt: float = 0.30,   # Entre relabel e filter: filtra
                 
                 # Clustering (igual ao original)
                 n_clusters: int = 30,
                 use_softmax_as_embedding: bool = True,
                 
                 # Critérios progressivos (igual ao original)
                 progressive_epochs: int = 4,
                 
                 # Conservador (épocas iniciais)
                 early_anchor_gmm_threshold: float = 0.15,
                 early_anchor_pred_agreement: float = 0.85,
                 early_anchor_confidence: float = 0.9,
                 early_suspect_gmm_threshold: float = 0.8,
                 early_similarity_threshold: float = 0.7,
                 early_cluster_consensus: float = 0.85,
                 
                 # Agressivo (épocas posteriores)
                 anchor_gmm_threshold: float = 0.4,
                 anchor_pred_agreement: float = 0.6,
                 anchor_confidence: float = 0.7,
                 suspect_gmm_threshold: float = 0.5,
                 similarity_threshold: float = 0.4,
                 cluster_consensus: float = 0.6,
                 
                 # Spatial Refinement (igual ao original)
                 enable_spatial_refinement: bool = True,
                 spatial_difficulty_threshold: float = 0.5,
                 
                 # Configuração
                 iou_assigner: float = 0.5,
                 reload_dataset: bool = True,
                 debug: bool = True):
        
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        
        self.relabel_max_score_gt = relabel_max_score_gt
        self.filter_max_score_gt = filter_max_score_gt
        
        self.n_clusters = n_clusters
        self.use_softmax_as_embedding = use_softmax_as_embedding
        
        self.progressive_epochs = progressive_epochs
        
        # Conservador
        self.early_anchor_gmm_threshold = early_anchor_gmm_threshold
        self.early_anchor_pred_agreement = early_anchor_pred_agreement
        self.early_anchor_confidence = early_anchor_confidence
        self.early_suspect_gmm_threshold = early_suspect_gmm_threshold
        self.early_similarity_threshold = early_similarity_threshold
        self.early_cluster_consensus = early_cluster_consensus
        
        # Agressivo
        self.anchor_gmm_threshold = anchor_gmm_threshold
        self.anchor_pred_agreement = anchor_pred_agreement
        self.anchor_confidence = anchor_confidence
        self.suspect_gmm_threshold = suspect_gmm_threshold
        self.similarity_threshold = similarity_threshold
        self.cluster_consensus = cluster_consensus
        
        # Spatial
        self.enable_spatial_refinement = enable_spatial_refinement
        self.spatial_difficulty_threshold = spatial_difficulty_threshold
        
        self.iou_assigner = iou_assigner
        self.reload_dataset = reload_dataset
        self.debug = debug
    
    def _get_current_criteria(self, epoch):
        if epoch <= self.progressive_epochs:
            return {
                'anchor_gmm_threshold': self.early_anchor_gmm_threshold,
                'anchor_pred_agreement': self.early_anchor_pred_agreement,
                'anchor_confidence': self.early_anchor_confidence,
                'suspect_gmm_threshold': self.early_suspect_gmm_threshold,
                'similarity_threshold': self.early_similarity_threshold,
                'cluster_consensus': self.early_cluster_consensus,
                'phase': 'CONSERVADOR'
            }
        else:
            return {
                'anchor_gmm_threshold': self.anchor_gmm_threshold,
                'anchor_pred_agreement': self.anchor_pred_agreement,
                'anchor_confidence': self.anchor_confidence,
                'suspect_gmm_threshold': self.suspect_gmm_threshold,
                'similarity_threshold': self.similarity_threshold,
                'cluster_consensus': self.cluster_consensus,
                'phase': 'AGRESSIVO'
            }
    
    def _fit_gmm_per_class(self, scores_by_class):
        gmm_dict = {}
        for cls_id, scores in scores_by_class.items():
            if len(scores) < 10:
                continue
            scores_np = np.array(scores).reshape(-1, 1)
            try:
                gmm = GaussianMixture(n_components=2, max_iter=100, random_state=42)
                gmm.fit(scores_np)
                gmm_dict[cls_id] = gmm
            except:
                pass
        return gmm_dict
    
    def _get_p_noise(self, score_gt, gt_label, gmm_dict):
        if gt_label not in gmm_dict:
            return 0.5
        gmm = gmm_dict[gt_label]
        try:
            probs = gmm.predict_proba(np.array([[score_gt]]))
            noisy_component = np.argmin(gmm.means_)
            return float(probs[0, noisy_component])
        except:
            return 0.5
    
    def _cluster_embeddings(self, embeddings, n_clusters):
        N, D = embeddings.shape
        n_clusters = min(n_clusters, N // 2)
        if n_clusters < 2:
            return np.zeros(N, dtype=np.int32)
        
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        embeddings_norm = embeddings_norm.astype(np.float32)
        
        if FAISS_AVAILABLE:
            kmeans = faiss.Kmeans(D, n_clusters, niter=20, verbose=False)
            kmeans.train(embeddings_norm)
            _, cluster_ids = kmeans.index.search(embeddings_norm, 1)
            cluster_ids = cluster_ids.flatten()
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_ids = kmeans.fit_predict(embeddings_norm)
        
        return cluster_ids
    
    def before_train_epoch(self, runner):
        epoch = runner.epoch + 1
        
        if epoch <= self.warmup_epochs:
            if self.debug:
                print(f"[RelabelLowScore] Época {epoch}: Warmup, pulando.")
            return
        
        if self.debug:
            print(f"\n[RelabelLowScore] ========== Época {epoch} ==========")
        
        # Obter dataset
        dataloader = runner.train_loop.dataloader
        # dataset = self._get_base_dataset(dataloader.dataset)    
        dataset = dataloader.dataset
        
        # Reload dataset
        if self.reload_dataset:
            # self._reload_datasets(runner)
            reload_leaf_datasets(dataset)

        
        # if not hasattr(dataset, 'datasets'):
        #     print("[VCNC-Spatial] ERRO: Esperado ConcatDataset")
        #     return
        
        # datasets = dataset.datasets
        datasets = unwrap_to_leaf_datasets(dataset)
        dataset_img_map = self._build_image_map(datasets)
        
        assigner = MaxIoUAssigner(
            pos_iou_thr=self.iou_assigner,
            neg_iou_thr=self.iou_assigner,
            min_pos_iou=self.iou_assigner,
            match_low_quality=False
        )
        
        # Coleta de dados
        all_box_data = []
        boxes_by_image = defaultdict(list)
        scores_by_class = defaultdict(list)
        
        for batch_idx, data_batch in enumerate(dataloader):
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                predictions = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
            
            for i, data_sample in enumerate(data_batch['data_samples']):
                img_path = data_sample.img_path
                if img_path not in dataset_img_map:
                    continue
                
                sub_idx, data_idx = dataset_img_map[img_path]
                pred_instances = predictions[i].pred_instances
                pred_instances.priors = pred_instances.pop('bboxes')
                device = pred_instances.priors.device
                
                gt_instances = data_sample.gt_instances
                gt_instances.bboxes = gt_instances.bboxes.to(device)
                gt_instances.labels = gt_instances.labels.to(device)
                pred_instances.priors = pred_instances.priors.to(device)
                pred_instances.labels = pred_instances.labels.to(device)
                pred_instances.scores = pred_instances.scores.to(device)
                pred_instances.logits = pred_instances.logits.to(device)
                
                gt_bboxes = gt_instances.bboxes
                
                assign_result = assigner.assign(pred_instances, gt_instances)
                
                for gt_idx in range(assign_result.num_gts):
                    associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if associated_preds.numel() == 0:
                        continue
                    
                    logits_associated = pred_instances.logits[associated_preds]
                    scores = torch.softmax(logits_associated, dim=-1)
                    best_pred_idx = scores.max(dim=1).values.argmax()
                    best_scores = scores[best_pred_idx]
                    best_logits = logits_associated[best_pred_idx]
                    
                    gt_label = gt_instances.labels[gt_idx].item()
                    
                    if hasattr(gt_bboxes, 'tensor'):
                        gt_bbox = gt_bboxes.tensor[gt_idx]
                    else:
                        gt_bbox = gt_bboxes[gt_idx]
                    
                    score_gt = best_scores[gt_label].item()
                    pred_label = best_scores.argmax().item()
                    pred_score = best_scores.max().item()
                    
                    if self.use_softmax_as_embedding:
                        embedding = best_scores.cpu().numpy()
                    else:
                        embedding = best_logits.cpu().numpy()
                    
                    box_data = {
                        'img_path': img_path,
                        'sub_idx': sub_idx,
                        'data_idx': data_idx,
                        'gt_idx': gt_idx,
                        'gt_label': gt_label,
                        'gt_bbox': gt_bbox.cpu(),
                        'score_gt': score_gt,
                        'pred_label': pred_label,
                        'pred_score': pred_score,
                        'embedding': embedding,
                        'scores': best_scores.cpu(),
                        'filtered': False,
                        'relabeled': False,
                    }
                    all_box_data.append(box_data)
                    boxes_by_image[img_path].append(box_data)
                    scores_by_class[gt_label].append(score_gt)
        
        if len(all_box_data) == 0:
            return
        
        if self.debug:
            print(f"[RelabelLowScore] Total boxes: {len(all_box_data)}")
        
        # Ajustar GMM
        gmm_dict = self._fit_gmm_per_class(scores_by_class)
        
        # Calcular p_noise
        for box in all_box_data:
            box['p_noise'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict)
        
        # Clustering
        embeddings = np.array([box['embedding'] for box in all_box_data])
        cluster_ids = self._cluster_embeddings(embeddings, self.n_clusters)
        for i, box in enumerate(all_box_data):
            box['cluster_id'] = cluster_ids[i]
        
        # Obter critérios da época
        criteria = self._get_current_criteria(epoch)
        
        if self.debug:
            print(f"[RelabelLowScore] Fase: {criteria['phase']}, Clusters: {len(set(cluster_ids))}")
        
        # Agrupar por cluster
        clusters = defaultdict(list)
        for box in all_box_data:
            clusters[box['cluster_id']].append(box)
        
        c_anchor_gmm = criteria['anchor_gmm_threshold']
        c_anchor_pred = criteria['anchor_pred_agreement']
        c_anchor_conf = criteria['anchor_confidence']
        c_suspect_gmm = criteria['suspect_gmm_threshold']
        c_similarity = criteria['similarity_threshold']
        c_consensus = criteria['cluster_consensus']
        
        total_anchors = 0
        total_suspects = 0
        relabel_count = 0
        filter_count = 0
        
        for cluster_id, cluster_boxes in clusters.items():
            if len(cluster_boxes) < 2:
                continue
            
            # Identificar âncoras
            anchors = []
            for box in cluster_boxes:
                low_noise = box['p_noise'] < c_anchor_gmm
                model_agrees = box['score_gt'] > c_anchor_pred
                high_confidence = box['pred_score'] > c_anchor_conf
                
                if low_noise and model_agrees and high_confidence:
                    anchors.append(box)
            
            total_anchors += len(anchors)
            
            if len(anchors) == 0:
                continue
            
            # Verificar consenso
            anchor_labels = [a['gt_label'] for a in anchors]
            label_counts = Counter(anchor_labels)
            dominant_label, count = label_counts.most_common(1)[0]
            consensus_ratio = count / len(anchors)
            
            if consensus_ratio < c_consensus:
                continue
            
            # Calcular embedding médio das âncoras
            anchor_embeddings = np.array([a['embedding'] for a in anchors if a['gt_label'] == dominant_label])
            if len(anchor_embeddings) == 0:
                continue
            
            anchor_mean = anchor_embeddings.mean(axis=0)
            anchor_mean_norm = anchor_mean / (np.linalg.norm(anchor_mean) + 1e-8)
            
            anchor_ids = set(id(a) for a in anchors)
            
            # Processar suspeitos
            for box in cluster_boxes:
                if id(box) in anchor_ids:
                    continue
                
                if box['p_noise'] < c_suspect_gmm:
                    continue
                
                total_suspects += 1
                
                # Se já é da classe dominante, não precisa fazer nada
                if box['gt_label'] == dominant_label:
                    continue
                
                # Verificar similaridade
                box_emb_norm = box['embedding'] / (np.linalg.norm(box['embedding']) + 1e-8)
                similarity = np.dot(box_emb_norm, anchor_mean_norm)
                
                if similarity > c_similarity:
                    # ============================================================
                    # DIFERENÇA: Decisão baseada em score_gt
                    # ============================================================
                    if box['score_gt'] < self.relabel_max_score_gt:
                        # score_gt muito baixo → provavelmente ruído simétrico → RELABELA
                        self._apply_relabel(datasets, box['sub_idx'], box['data_idx'], box['gt_idx'], dominant_label)
                        box['relabeled'] = True
                        relabel_count += 1
                    elif box['score_gt'] < self.filter_max_score_gt:
                        # score_gt moderado → pode ser assimétrico → FILTRA
                        self._apply_ignore_flag(datasets, box['sub_idx'], box['data_idx'], box['gt_idx'])
                        box['filtered'] = True
                        filter_count += 1
                    # else: score_gt alto → mantém
        
        if self.debug:
            print(f"[RelabelLowScore] Total âncoras: {total_anchors} ({total_anchors/len(all_box_data)*100:.2f}%)")
            print(f"[RelabelLowScore] Total suspeitos: {total_suspects} ({total_suspects/len(all_box_data)*100:.2f}%)")
            print(f"[RelabelLowScore] Relabelados (score_gt < {self.relabel_max_score_gt}): {relabel_count}")
            print(f"[RelabelLowScore] Filtrados (score_gt < {self.filter_max_score_gt}): {filter_count}")
        
        # ============================================================
        # SPATIAL REFINEMENT (igual ao original)
        # ============================================================
        spatial_relabel_count = 0
        
        if self.enable_spatial_refinement:
            if self.debug:
                print(f"[RelabelLowScore] Aplicando Spatial Refinement...")
            
            for img_path, img_boxes in boxes_by_image.items():
                if len(img_boxes) < 2:
                    continue
                
                # Pular boxes já filtrados
                active_boxes = [b for b in img_boxes if not b['filtered']]
                if len(active_boxes) < 2:
                    continue
                
                boxes_tensor = torch.stack([b['gt_bbox'] for b in active_boxes])
                pred_labels = torch.tensor([b['pred_label'] for b in active_boxes])
                pred_scores = torch.stack([b['scores'] for b in active_boxes])
                
                refined_labels, refinements = spatial_aware_relabeling(
                    boxes_tensor,
                    pred_labels,
                    pred_scores,
                    difficulty_threshold=self.spatial_difficulty_threshold
                )
                
                for idx, box in enumerate(active_boxes):
                    if refined_labels[idx] != pred_labels[idx]:
                        new_label = refined_labels[idx].item()
                        self._apply_relabel(datasets, box['sub_idx'], box['data_idx'], box['gt_idx'], new_label)
                        spatial_relabel_count += 1
            
            if self.debug:
                print(f"[RelabelLowScore] Relabelados por spatial: {spatial_relabel_count}")
        
        if self.debug:
            print(f"[RelabelLowScore] ==========================================\n")
    
    def _reload_datasets(self, runner):
        try:
            ds = runner.train_loop.dataloader.dataset.dataset
            for subds in ds.datasets:
                if hasattr(subds, '_fully_initialized'):
                    subds._fully_initialized = False
                if hasattr(subds, 'full_init'):
                    subds.full_init()
        except:
            pass
    
    def _get_base_dataset(self, dataset):
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        return dataset
    
    def _build_image_map(self, datasets):
        img_map = {}
        for sub_idx, subds in enumerate(datasets):
            if hasattr(subds, 'data_list'):
                for data_idx, data_info in enumerate(subds.data_list):
                    img_map[data_info['img_path']] = (sub_idx, data_idx)
        return img_map
    
    def _apply_relabel(self, datasets, sub_idx, data_idx, gt_idx, new_label):
        try:
            datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]['bbox_label'] = new_label
        except:
            pass
    
    def _apply_ignore_flag(self, datasets, sub_idx, data_idx, gt_idx):
        try:
            datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]['ignore_flag'] = 1
        except:
            pass
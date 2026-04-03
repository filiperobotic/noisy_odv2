"""
Estratégia C (Ajustada): Distance Based com Thresholds Relaxados

Problema anterior: "Longe de tudo: 0" - nenhum box foi filtrado por estar longe
Ajuste: 
- far_threshold: 0.3 → 0.5 (mais boxes serão considerados "longe")
- close_threshold: 0.7 → 0.5 (menos boxes serão considerados "perto da GT")
- Adiciona lógica de score_gt mais conservadora para decidir relabel vs filter
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
class VCNCDistanceBasedV2Hook(Hook):
    """
    VCNC Estratégia C v2: Distance Based com thresholds ajustados
    
    Mudanças:
    - far_threshold: 0.5 (era 0.3) - mais boxes longe de tudo
    - close_threshold: 0.5 (era 0.7) - menos boxes perto da GT
    - score_gt_threshold: 0.02 (era 0.05) - mais conservador para relabel
    """
    
    def __init__(self,
                 warmup_epochs: int = 1,
                 num_classes: int = 20,
                 
                 # Thresholds de distância (AJUSTADOS)
                 close_threshold: float = 0.5,  # Era 0.7 - similaridade para "perto da GT"
                 far_threshold: float = 0.5,    # Era 0.3 - similaridade máxima para "longe de tudo"
                 score_gt_threshold: float = 0.02,  # Era 0.05 - mais conservador
                 
                 # Clustering
                 n_clusters: int = 30,
                 use_softmax_as_embedding: bool = True,
                 
                 # Critérios progressivos
                 progressive_epochs: int = 4,
                 
                 # Conservador
                 early_anchor_gmm_threshold: float = 0.15,
                 early_anchor_pred_agreement: float = 0.85,
                 early_anchor_confidence: float = 0.9,
                 early_suspect_gmm_threshold: float = 0.8,
                 early_similarity_threshold: float = 0.7,
                 early_cluster_consensus: float = 0.85,
                 
                 # Agressivo
                 anchor_gmm_threshold: float = 0.4,
                 anchor_pred_agreement: float = 0.6,
                 anchor_confidence: float = 0.7,
                 suspect_gmm_threshold: float = 0.5,
                 similarity_threshold: float = 0.4,
                 cluster_consensus: float = 0.6,
                 
                 # Spatial Refinement
                 enable_spatial_refinement: bool = True,
                 spatial_difficulty_threshold: float = 0.5,
                 
                 # Configuração
                 iou_assigner: float = 0.5,
                 reload_dataset: bool = True,
                 debug: bool = True):
        
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        
        self.close_threshold = close_threshold
        self.far_threshold = far_threshold
        self.score_gt_threshold = score_gt_threshold
        
        self.n_clusters = n_clusters
        self.use_softmax_as_embedding = use_softmax_as_embedding
        
        self.progressive_epochs = progressive_epochs
        
        self.early_anchor_gmm_threshold = early_anchor_gmm_threshold
        self.early_anchor_pred_agreement = early_anchor_pred_agreement
        self.early_anchor_confidence = early_anchor_confidence
        self.early_suspect_gmm_threshold = early_suspect_gmm_threshold
        self.early_similarity_threshold = early_similarity_threshold
        self.early_cluster_consensus = early_cluster_consensus
        
        self.anchor_gmm_threshold = anchor_gmm_threshold
        self.anchor_pred_agreement = anchor_pred_agreement
        self.anchor_confidence = anchor_confidence
        self.suspect_gmm_threshold = suspect_gmm_threshold
        self.similarity_threshold = similarity_threshold
        self.cluster_consensus = cluster_consensus
        
        self.enable_spatial_refinement = enable_spatial_refinement
        self.spatial_difficulty_threshold = spatial_difficulty_threshold
        
        self.iou_assigner = iou_assigner
        self.reload_dataset = reload_dataset
        self.debug = debug
        
        # Centróides das classes (atualizados a cada época)
        self.class_centroids = {}
    
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
    
    def _compute_class_centroids(self, anchors_by_class):
        """Calcula centróide de cada classe baseado nas âncoras."""
        centroids = {}
        for cls_id, embeddings in anchors_by_class.items():
            if len(embeddings) > 0:
                emb_array = np.array(embeddings)
                centroid = emb_array.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                centroids[cls_id] = centroid
        return centroids
    
    def _get_similarity_to_class(self, embedding, class_id):
        """Retorna similaridade do embedding com o centróide da classe."""
        if class_id not in self.class_centroids:
            return 0.0
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        return float(np.dot(emb_norm, self.class_centroids[class_id]))
    
    def _get_max_similarity_to_any_class(self, embedding):
        """Retorna a maior similaridade com qualquer classe."""
        if len(self.class_centroids) == 0:
            return 0.0
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        max_sim = 0.0
        for centroid in self.class_centroids.values():
            sim = float(np.dot(emb_norm, centroid))
            max_sim = max(max_sim, sim)
        return max_sim
    
    def before_train_epoch(self, runner):
        epoch = runner.epoch + 1
        
        if epoch <= self.warmup_epochs:
            if self.debug:
                print(f"[DistanceBasedV2] Época {epoch}: Warmup, pulando.")
            return
        
        if self.debug:
            print(f"\n[DistanceBasedV2] ========== Época {epoch} ==========")
        
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
            print(f"[DistanceBasedV2] Total boxes: {len(all_box_data)}")
        
        # Ajustar GMM
        gmm_dict = self._fit_gmm_per_class(scores_by_class)
        
        for box in all_box_data:
            box['p_noise'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict)
        
        # Clustering
        embeddings = np.array([box['embedding'] for box in all_box_data])
        cluster_ids = self._cluster_embeddings(embeddings, self.n_clusters)
        for i, box in enumerate(all_box_data):
            box['cluster_id'] = cluster_ids[i]
        
        criteria = self._get_current_criteria(epoch)
        
        if self.debug:
            print(f"[DistanceBasedV2] Fase: {criteria['phase']}")
        
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
        
        # Primeiro passo: identificar âncoras e calcular centróides
        anchors_by_class = defaultdict(list)
        total_anchors = 0
        
        for cluster_id, cluster_boxes in clusters.items():
            if len(cluster_boxes) < 2:
                continue
            
            for box in cluster_boxes:
                low_noise = box['p_noise'] < c_anchor_gmm
                model_agrees = box['score_gt'] > c_anchor_pred
                high_confidence = box['pred_score'] > c_anchor_conf
                if low_noise and model_agrees and high_confidence:
                    anchors_by_class[box['gt_label']].append(box['embedding'])
                    total_anchors += 1
        
        # Calcular centróides
        self.class_centroids = self._compute_class_centroids(anchors_by_class)
        
        if self.debug:
            print(f"[DistanceBasedV2] Total âncoras: {total_anchors}")
            print(f"[DistanceBasedV2] Classes com centróides: {len(self.class_centroids)}")
        
        # Segundo passo: processar suspeitos
        total_suspects = 0
        stats = {
            'close_to_gt': 0,
            'far_from_all': 0,
            'relabel_symmetric': 0,
            'filter_asymmetric': 0,
        }
        
        for cluster_id, cluster_boxes in clusters.items():
            if len(cluster_boxes) < 2:
                continue
            
            # Re-identificar âncoras
            anchors = []
            for box in cluster_boxes:
                low_noise = box['p_noise'] < c_anchor_gmm
                model_agrees = box['score_gt'] > c_anchor_pred
                high_confidence = box['pred_score'] > c_anchor_conf
                if low_noise and model_agrees and high_confidence:
                    anchors.append(box)
            
            if len(anchors) == 0:
                continue
            
            anchor_labels = [a['gt_label'] for a in anchors]
            label_counts = Counter(anchor_labels)
            dominant_label, count = label_counts.most_common(1)[0]
            consensus_ratio = count / len(anchors)
            
            if consensus_ratio < c_consensus:
                continue
            
            anchor_embeddings = np.array([a['embedding'] for a in anchors if a['gt_label'] == dominant_label])
            if len(anchor_embeddings) == 0:
                continue
            
            anchor_mean = anchor_embeddings.mean(axis=0)
            anchor_mean_norm = anchor_mean / (np.linalg.norm(anchor_mean) + 1e-8)
            
            anchor_ids = set(id(a) for a in anchors)
            
            for box in cluster_boxes:
                if id(box) in anchor_ids:
                    continue
                if box['p_noise'] < c_suspect_gmm:
                    continue
                
                total_suspects += 1
                
                if box['gt_label'] == dominant_label:
                    continue
                
                box_emb_norm = box['embedding'] / (np.linalg.norm(box['embedding']) + 1e-8)
                similarity = np.dot(box_emb_norm, anchor_mean_norm)
                
                if similarity > c_similarity:
                    # ============================================================
                    # DECISÃO BASEADA EM DISTÂNCIA (V2)
                    # ============================================================
                    
                    # Calcular similaridade com a GT e com qualquer classe
                    sim_to_gt = self._get_similarity_to_class(box['embedding'], box['gt_label'])
                    max_sim_any = self._get_max_similarity_to_any_class(box['embedding'])
                    
                    # Caso 1: Perto da GT → manter (não fazer nada)
                    if sim_to_gt > self.close_threshold:
                        stats['close_to_gt'] += 1
                        continue
                    
                    # Caso 2: Longe de todas as classes → filtrar (outlier)
                    if max_sim_any < self.far_threshold:
                        self._apply_ignore_flag(datasets, box['sub_idx'], box['data_idx'], box['gt_idx'])
                        box['filtered'] = True
                        stats['far_from_all'] += 1
                        continue
                    
                    # Caso 3: Decidir entre relabel e filter baseado em score_gt
                    if box['score_gt'] < self.score_gt_threshold:
                        # Score muito baixo + longe da GT → provavelmente simétrico → RELABELA
                        self._apply_relabel(datasets, box['sub_idx'], box['data_idx'], box['gt_idx'], dominant_label)
                        box['relabeled'] = True
                        stats['relabel_symmetric'] += 1
                    else:
                        # Score não tão baixo + longe da GT → pode ser assimétrico → FILTRA
                        self._apply_ignore_flag(datasets, box['sub_idx'], box['data_idx'], box['gt_idx'])
                        box['filtered'] = True
                        stats['filter_asymmetric'] += 1
        
        if self.debug:
            print(f"[DistanceBasedV2] Total suspeitos: {total_suspects}")
            print(f"[DistanceBasedV2] Decisões:")
            print(f"  - Perto da GT (mantém): {stats['close_to_gt']}")
            print(f"  - Longe de tudo (filtra): {stats['far_from_all']}")
            print(f"  - Relabel (simétrico, score<{self.score_gt_threshold}): {stats['relabel_symmetric']}")
            print(f"  - Filtra (assimétrico): {stats['filter_asymmetric']}")
            print(f"[DistanceBasedV2] Total: Relabel={stats['relabel_symmetric']}, Filtra={stats['far_from_all'] + stats['filter_asymmetric']}")
        
        # Spatial Refinement
        spatial_relabel_count = 0
        if self.enable_spatial_refinement:
            for img_path, img_boxes in boxes_by_image.items():
                if len(img_boxes) < 2:
                    continue
                active_boxes = [b for b in img_boxes if not b['filtered']]
                if len(active_boxes) < 2:
                    continue
                boxes_tensor = torch.stack([b['gt_bbox'] for b in active_boxes])
                pred_labels = torch.tensor([b['pred_label'] for b in active_boxes])
                pred_scores = torch.stack([b['scores'] for b in active_boxes])
                refined_labels, refinements = spatial_aware_relabeling(
                    boxes_tensor, pred_labels, pred_scores,
                    difficulty_threshold=self.spatial_difficulty_threshold
                )
                for idx, box in enumerate(active_boxes):
                    if refined_labels[idx] != pred_labels[idx]:
                        new_label = refined_labels[idx].item()
                        self._apply_relabel(datasets, box['sub_idx'], box['data_idx'], box['gt_idx'], new_label)
                        spatial_relabel_count += 1
            if self.debug:
                print(f"[DistanceBasedV2] Relabelados por spatial: {spatial_relabel_count}")
        
        if self.debug:
            print(f"[DistanceBasedV2] ==========================================\n")
    
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
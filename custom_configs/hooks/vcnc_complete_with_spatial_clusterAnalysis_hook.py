"""
VCNC com Análise por Cluster

Insight: A proporção de âncoras por cluster indica o tipo de ruído:
- Simétrico: GMM detecta ruído → POUCAS âncoras no cluster (~50-60%)
- Assimétrico: GMM NÃO detecta → MUITAS âncoras no cluster (~80%+)
- Limpo: Tudo ok → MUITAS âncoras no cluster (~80%+)

Lógica por cluster:
- % âncoras BAIXO → simétrico → RELABELING (GMM já tratou)
- % âncoras ALTO + top-2 concentrado → assimétrico → FILTRAR
- % âncoras ALTO + top-2 variado → limpo → RELABELING
"""

from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import torch
import torch.nn.functional as F
from mmdet.models.task_modules.assigners import MaxIoUAssigner
from collections import Counter, defaultdict
import numpy as np
from sklearn.mixture import GaussianMixture
import os
import json

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    from sklearn.cluster import KMeans


def unwrap_to_leaf_datasets(dataset):
    datasets = [dataset]
    changed = True
    while changed:
        changed = False
        new_datasets = []
        for ds in datasets:
            if hasattr(ds, 'datasets'):
                new_datasets.extend(ds.datasets)
                changed = True
            elif hasattr(ds, 'dataset'):
                new_datasets.append(ds.dataset)
                changed = True
            else:
                new_datasets.append(ds)
        datasets = new_datasets
    return datasets


def reload_leaf_datasets(dataset):
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
class VCNCClusterAnalysisHook(Hook):
    """
    VCNC com Análise por Cluster.
    
    Detecta tipo de ruído analisando % de âncoras por cluster:
    - Baixo % âncoras → simétrico → relabeling ok
    - Alto % âncoras + top-2 concentrado → assimétrico → filtrar
    - Alto % âncoras + top-2 variado → limpo → relabeling ok
    """
    
    def __init__(self,
                 warmup_epochs: int = 1,
                 num_classes: int = 20,
                 
                 # === ANÁLISE POR CLUSTER ===
                 anchor_ratio_threshold: float = 0.7,  # Acima = alto, abaixo = baixo
                 top2_concentration_threshold: float = 0.3,  # Top-2 concentrado se > 30%
                 min_cluster_size: int = 5,  # Mínimo de amostras para analisar
                 
                 # === CLUSTERING ===
                 enable_clustering_relabel: bool = True,
                 n_clusters: int = 30,
                 use_softmax_as_embedding: bool = True,
                 
                 # === CRITÉRIOS PROGRESSIVOS ===
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
                 
                 # === CONFIANÇA (ETAPA 1) ===
                 enable_confidence_relabel: bool = True,
                 relabel_confidence_threshold: float = 0.9,
                 
                 # === SPATIAL REFINEMENT ===
                 enable_spatial_refinement: bool = True,
                 spatial_difficulty_threshold: float = 0.5,
                 
                 # === GMM FILTER ===
                 enable_gmm_filter: bool = True,
                 gmm_components: int = 4,
                 filter_gmm_threshold: float = 0.7,
                 
                 # === CONFIGURAÇÃO ===
                 iou_assigner: float = 0.5,
                 reload_dataset: bool = True,
                 debug: bool = True,
                 
                 # === DIAGNÓSTICO ===
                 enable_diagnostic: bool = True,
                 diagnostic_output_dir: str = './vcnc_diagnostics'):
        
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        
        # Análise por cluster
        self.anchor_ratio_threshold = anchor_ratio_threshold
        self.top2_concentration_threshold = top2_concentration_threshold
        self.min_cluster_size = min_cluster_size
        
        # Clustering
        self.enable_clustering_relabel = enable_clustering_relabel
        self.n_clusters = n_clusters
        self.use_softmax_as_embedding = use_softmax_as_embedding
        
        # Progressivo
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
        
        # Confiança
        self.enable_confidence_relabel = enable_confidence_relabel
        self.relabel_confidence_threshold = relabel_confidence_threshold
        
        # Spatial
        self.enable_spatial_refinement = enable_spatial_refinement
        self.spatial_difficulty_threshold = spatial_difficulty_threshold
        
        # GMM Filter
        self.enable_gmm_filter = enable_gmm_filter
        self.gmm_components = gmm_components
        self.filter_gmm_threshold = filter_gmm_threshold
        
        # Configuração
        self.iou_assigner = iou_assigner
        self.reload_dataset = reload_dataset
        self.debug = debug
        
        # Diagnóstico
        self.enable_diagnostic = enable_diagnostic
        self.diagnostic_output_dir = diagnostic_output_dir
        self.all_diagnostic_stats = []
        
        if self.enable_diagnostic:
            os.makedirs(diagnostic_output_dir, exist_ok=True)
    
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
    
    def _analyze_cluster(self, cluster_boxes, anchors):
        """
        Analisa um cluster e determina o tipo de ruído.
        
        Returns:
            cluster_type: 'symmetric', 'asymmetric', ou 'clean'
            anchor_ratio: % de âncoras no cluster
            top2_concentration: concentração do top-2 nas âncoras
        """
        if len(cluster_boxes) < self.min_cluster_size:
            return 'skip', 0, 0, None
        
        # Calcular % de âncoras
        anchor_ratio = len(anchors) / len(cluster_boxes)
        
        if anchor_ratio < self.anchor_ratio_threshold:
            # Poucas âncoras → GMM detectou ruído → simétrico
            return 'symmetric', anchor_ratio, 0, None
        
        # Muitas âncoras → verificar top-2
        if len(anchors) == 0:
            return 'skip', anchor_ratio, 0, None
        
        # Contar distribuição de top-2 nas âncoras
        top2_counts = Counter()
        for anchor in anchors:
            top2_label = anchor['top2_label']
            gt_label = anchor['gt_label']
            if top2_label != gt_label:
                top2_counts[top2_label] += 1
        
        if len(top2_counts) == 0:
            return 'clean', anchor_ratio, 0, None
        
        # Verificar concentração
        most_common_top2, count = top2_counts.most_common(1)[0]
        concentration = count / len(anchors)
        
        if concentration >= self.top2_concentration_threshold:
            # Top-2 concentrado → assimétrico (ou similaridade natural, mas tratamos igual)
            return 'asymmetric', anchor_ratio, concentration, most_common_top2
        else:
            # Top-2 variado → limpo
            return 'clean', anchor_ratio, concentration, None
    
    def before_train_epoch(self, runner):
        epoch = runner.epoch + 1
        
        if epoch <= self.warmup_epochs:
            if self.debug:
                print(f"[VCNC-Cluster] Época {epoch}: Warmup, pulando.")
            return
        
        if self.debug:
            print(f"\n[VCNC-Cluster] ========== Época {epoch} ==========")
        
        if self.reload_dataset:
            self._reload_datasets(runner)
        
        dataloader = runner.train_loop.dataloader
        dataset = self._get_base_dataset(dataloader.dataset)
        
        if not hasattr(dataset, 'datasets'):
            return
        
        datasets = dataset.datasets
        
        assigner = MaxIoUAssigner(
            pos_iou_thr=self.iou_assigner,
            neg_iou_thr=self.iou_assigner,
            min_pos_iou=self.iou_assigner,
            match_low_quality=False
        )
        
        # ============================================================
        # COLETA DE DADOS
        # ============================================================
        all_box_data = []
        boxes_by_image = defaultdict(list)
        scores_by_class = defaultdict(list)
        
        model = runner.model
        model.eval()
        
        with torch.no_grad():
            for sub_idx, subds in enumerate(datasets):
                if not hasattr(subds, 'data_list'):
                    continue
                    
                for data_idx, data_info in enumerate(subds.data_list):
                    img_path = data_info.get('img_path', '')
                    instances = data_info.get('instances', [])
                    
                    if not instances:
                        continue
                    
                    try:
                        data_batch = subds[data_idx]
                        if 'inputs' not in data_batch or 'data_samples' not in data_batch:
                            continue
                        
                        batch = {
                            'inputs': data_batch['inputs'].unsqueeze(0).cuda(),
                            'data_samples': [data_batch['data_samples']]
                        }
                        
                        features = model.extract_feat(batch['inputs'])
                        bbox_head = model.bbox_head
                        
                        cls_scores, bbox_preds = bbox_head(features)
                        
                        ds_sample = batch['data_samples'][0]
                        if hasattr(ds_sample, 'gt_instances'):
                            gt_bboxes = ds_sample.gt_instances.bboxes
                            gt_labels = ds_sample.gt_instances.labels
                        else:
                            continue
                        
                        all_anchors = bbox_head.prior_generator.grid_priors(
                            [f.shape[-2:] for f in cls_scores],
                            device=cls_scores[0].device
                        )
                        flat_anchors = torch.cat(all_anchors, dim=0)
                        flat_cls_scores = torch.cat([
                            s.permute(0, 2, 3, 1).reshape(-1, s.size(1))
                            for s in cls_scores
                        ], dim=0)
                        
                        assign_result = assigner.assign(
                            flat_anchors, gt_bboxes, gt_labels=gt_labels
                        )
                        
                        for gt_idx in range(len(gt_labels)):
                            assigned_mask = (assign_result.gt_inds == (gt_idx + 1))
                            if not assigned_mask.any():
                                continue
                            
                            assigned_scores = flat_cls_scores[assigned_mask]
                            probs = F.softmax(assigned_scores, dim=1)
                            
                            mean_probs = probs.mean(dim=0)
                            best_scores = mean_probs
                            best_logits = assigned_scores.mean(dim=0)
                            
                            gt_label = gt_labels[gt_idx].item()
                            gt_bbox = gt_bboxes[gt_idx]
                            
                            score_gt = best_scores[gt_label].item()
                            
                            top2_scores, top2_indices = best_scores.topk(2)
                            pred_label = top2_indices[0].item()
                            pred_score = top2_scores[0].item()
                            top2_label = top2_indices[1].item()
                            top2_score = top2_scores[1].item()
                            margin = (top2_scores[0] - top2_scores[1]).item()
                            
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
                                'top2_label': top2_label,
                                'top2_score': top2_score,
                                'margin': margin,
                                'embedding': embedding,
                                'scores': best_scores.cpu(),
                                'relabeled_by': None,
                                'filtered': False
                            }
                            all_box_data.append(box_data)
                            scores_by_class[gt_label].append(score_gt)
                            boxes_by_image[img_path].append(box_data)
                    except Exception as e:
                        continue
        
        if len(all_box_data) == 0:
            print("[VCNC-Cluster] Nenhum box coletado!")
            return
        
        if self.debug:
            print(f"[VCNC-Cluster] Coletados {len(all_box_data)} boxes")
        
        # ============================================================
        # ETAPA 1: RELABEL POR CONFIANÇA ALTA
        # ============================================================
        confidence_relabel_count = 0
        
        if self.enable_confidence_relabel:
            for box in all_box_data:
                if box['pred_score'] > self.relabel_confidence_threshold:
                    if box['pred_label'] != box['gt_label']:
                        self._apply_relabel(
                            datasets,
                            box['sub_idx'],
                            box['data_idx'],
                            box['gt_idx'],
                            box['pred_label']
                        )
                        box['gt_label'] = box['pred_label']
                        box['score_gt'] = box['pred_score']
                        box['relabeled_by'] = 'confidence'
                        confidence_relabel_count += 1
            
            if self.debug:
                print(f"[VCNC-Cluster] Relabelados por confiança: {confidence_relabel_count}")
        
        # ============================================================
        # ETAPA 2: ANÁLISE POR CLUSTER
        # ============================================================
        clustering_relabel_count = 0
        asymmetric_filter_count = 0
        
        # Estatísticas por tipo de cluster
        cluster_stats = {
            'symmetric': 0,
            'asymmetric': 0,
            'clean': 0,
            'skip': 0
        }
        
        if self.enable_clustering_relabel:
            # GMM
            scores_by_class_updated = defaultdict(list)
            for box in all_box_data:
                scores_by_class_updated[box['gt_label']].append(box['score_gt'])
            
            gmm_dict = self._fit_gmm_per_class(scores_by_class_updated)
            
            for box in all_box_data:
                box['p_noise'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict)
            
            # Clustering
            embeddings = np.array([box['embedding'] for box in all_box_data])
            cluster_ids = self._cluster_embeddings(embeddings, self.n_clusters)
            
            for i, box in enumerate(all_box_data):
                box['cluster_id'] = cluster_ids[i]
            
            criteria = self._get_current_criteria(epoch)
            
            clusters = defaultdict(list)
            for box in all_box_data:
                clusters[box['cluster_id']].append(box)
            
            c_anchor_gmm = criteria['anchor_gmm_threshold']
            c_anchor_pred = criteria['anchor_pred_agreement']
            c_anchor_conf = criteria['anchor_confidence']
            c_suspect_gmm = criteria['suspect_gmm_threshold']
            c_similarity = criteria['similarity_threshold']
            c_consensus = criteria['cluster_consensus']
            
            if self.debug:
                print(f"[VCNC-Cluster] Fase: {criteria['phase']}, Clusters: {len(clusters)}")
            
            # Processar cada cluster
            for cluster_id, cluster_boxes in clusters.items():
                if len(cluster_boxes) < 2:
                    continue
                
                # Identificar âncoras do cluster
                anchors = []
                for box in cluster_boxes:
                    is_clean = box['p_noise'] < c_anchor_gmm
                    model_agrees = box['score_gt'] > c_anchor_pred
                    high_confidence = box['pred_score'] > c_anchor_conf
                    
                    if is_clean and model_agrees and high_confidence:
                        box['is_anchor'] = True
                        anchors.append(box)
                    else:
                        box['is_anchor'] = False
                
                # Analisar tipo do cluster
                cluster_type, anchor_ratio, top2_conc, concentrated_class = self._analyze_cluster(
                    cluster_boxes, anchors
                )
                
                cluster_stats[cluster_type] += 1
                
                if cluster_type == 'skip':
                    continue
                
                if len(anchors) == 0:
                    continue
                
                # Determinar classe dominante
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
                
                # Processar suspeitos
                for box in cluster_boxes:
                    if id(box) in anchor_ids:
                        continue
                    
                    if box['relabeled_by'] == 'confidence':
                        continue
                    
                    if box['p_noise'] < c_suspect_gmm:
                        continue
                    
                    if box['gt_label'] == dominant_label:
                        continue
                    
                    box_emb_norm = box['embedding'] / (np.linalg.norm(box['embedding']) + 1e-8)
                    similarity = np.dot(box_emb_norm, anchor_mean_norm)
                    
                    if similarity > c_similarity:
                        # DECISÃO BASEADA NO TIPO DO CLUSTER
                        if cluster_type == 'asymmetric':
                            # Cluster assimétrico → FILTRAR
                            self._apply_ignore_flag(
                                datasets,
                                box['sub_idx'],
                                box['data_idx'],
                                box['gt_idx']
                            )
                            box['filtered'] = True
                            box['filtered_reason'] = 'asymmetric_cluster'
                            asymmetric_filter_count += 1
                        else:
                            # Cluster simétrico ou limpo → RELABELING
                            self._apply_relabel(
                                datasets,
                                box['sub_idx'],
                                box['data_idx'],
                                box['gt_idx'],
                                dominant_label
                            )
                            box['gt_label'] = dominant_label
                            box['score_gt'] = box['scores'][dominant_label].item()
                            box['relabeled_by'] = 'clustering'
                            clustering_relabel_count += 1
            
            if self.debug:
                print(f"[VCNC-Cluster] Tipos de cluster: simétrico={cluster_stats['symmetric']}, "
                      f"assimétrico={cluster_stats['asymmetric']}, limpo={cluster_stats['clean']}")
                print(f"[VCNC-Cluster] Relabelados (sim/limpo): {clustering_relabel_count}")
                print(f"[VCNC-Cluster] FILTRADOS (assimétrico): {asymmetric_filter_count}")
        
        # ============================================================
        # ETAPA 3: SPATIAL REFINEMENT
        # ============================================================
        spatial_relabel_count = 0
        
        if self.enable_spatial_refinement:
            for img_path, img_boxes in boxes_by_image.items():
                if len(img_boxes) < 2:
                    continue
                
                boxes_tensor = torch.stack([b['gt_bbox'] for b in img_boxes])
                pred_labels = torch.tensor([b['pred_label'] for b in img_boxes])
                pred_scores = torch.stack([b['scores'] for b in img_boxes])
                
                refined_labels, refinements = spatial_aware_relabeling(
                    boxes_tensor, pred_labels, pred_scores, 
                    self.spatial_difficulty_threshold
                )
                
                for i, box in enumerate(img_boxes):
                    if refined_labels[i] != pred_labels[i]:
                        if box['relabeled_by'] is None and not box['filtered']:
                            new_label = refined_labels[i].item()
                            self._apply_relabel(
                                datasets,
                                box['sub_idx'],
                                box['data_idx'],
                                box['gt_idx'],
                                new_label
                            )
                            box['gt_label'] = new_label
                            box['relabeled_by'] = 'spatial'
                            spatial_relabel_count += 1
            
            if self.debug:
                print(f"[VCNC-Cluster] Relabelados por spatial: {spatial_relabel_count}")
        
        # ============================================================
        # ETAPA 4: FILTRAGEM GMM
        # ============================================================
        gmm_filter_count = 0
        
        if self.enable_gmm_filter:
            scores_by_class_final = defaultdict(list)
            box_indices_by_class = defaultdict(list)
            
            for i, box in enumerate(all_box_data):
                if not box['filtered']:
                    scores_by_class_final[box['gt_label']].append(box['score_gt'])
                    box_indices_by_class[box['gt_label']].append(i)
            
            for cls_id, scores in scores_by_class_final.items():
                if len(scores) < 10:
                    continue
                
                scores_np = np.array(scores).reshape(-1, 1)
                
                try:
                    n_comp = min(self.gmm_components, len(scores) // 5)
                    if n_comp < 2:
                        continue
                    
                    gmm = GaussianMixture(n_components=n_comp, random_state=42)
                    gmm.fit(scores_np)
                    
                    probs = gmm.predict_proba(scores_np)
                    noisy_component = np.argmin(gmm.means_)
                    
                    for local_idx, global_idx in enumerate(box_indices_by_class[cls_id]):
                        box = all_box_data[global_idx]
                        if box['filtered'] or box['relabeled_by'] is not None:
                            continue
                        
                        p_noise = probs[local_idx, noisy_component]
                        
                        if p_noise > self.filter_gmm_threshold:
                            self._apply_ignore_flag(
                                datasets,
                                box['sub_idx'],
                                box['data_idx'],
                                box['gt_idx']
                            )
                            box['filtered'] = True
                            gmm_filter_count += 1
                except Exception as e:
                    continue
            
            if self.debug:
                print(f"[VCNC-Cluster] Filtrados por GMM: {gmm_filter_count}")
        
        # ============================================================
        # DIAGNÓSTICO
        # ============================================================
        if self.enable_diagnostic:
            stats = {
                'epoch': epoch,
                'total_boxes': len(all_box_data),
                'cluster_types': cluster_stats,
                'confidence_relabel': confidence_relabel_count,
                'clustering_relabel': clustering_relabel_count,
                'asymmetric_filter': asymmetric_filter_count,
                'spatial_relabel': spatial_relabel_count,
                'gmm_filter': gmm_filter_count,
            }
            
            self.all_diagnostic_stats.append(stats)
            
            output_file = os.path.join(self.diagnostic_output_dir, f'diagnostic_epoch_{epoch}.json')
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"\n[VCNC-Cluster] ========== RESUMO ==========")
            print(f"[VCNC-Cluster] Clusters simétricos: {cluster_stats['symmetric']} → relabeling")
            print(f"[VCNC-Cluster] Clusters assimétricos: {cluster_stats['asymmetric']} → filtragem")
            print(f"[VCNC-Cluster] Clusters limpos: {cluster_stats['clean']} → relabeling")
            print(f"[VCNC-Cluster] ===============================\n")
        
        model.train()
    
    def after_train(self, runner):
        if self.enable_diagnostic and self.all_diagnostic_stats:
            output_file = os.path.join(self.diagnostic_output_dir, 'diagnostic_all_epochs.json')
            with open(output_file, 'w') as f:
                json.dump(self.all_diagnostic_stats, f, indent=2)
    
    def _reload_datasets(self, runner):
        dataloader = runner.train_loop.dataloader
        dataset = self._get_base_dataset(dataloader.dataset)
        reload_leaf_datasets(dataset)
    
    def _get_base_dataset(self, dataset):
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        return dataset
    
    def _apply_relabel(self, datasets, sub_idx, data_idx, gt_idx, new_label):
        try:
            instance = datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]
            instance['bbox_label'] = new_label
        except Exception as e:
            if self.debug:
                print(f"[VCNC-Cluster] Erro relabel: {e}")
    
    def _apply_ignore_flag(self, datasets, sub_idx, data_idx, gt_idx):
        try:
            instance = datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]
            instance['ignore_flag'] = 1
        except Exception as e:
            if self.debug:
                print(f"[VCNC-Cluster] Erro ignore_flag: {e}")
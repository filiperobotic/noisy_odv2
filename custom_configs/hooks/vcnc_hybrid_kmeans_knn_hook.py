"""
Visual Clustering Noise Correction Hook - VERSÃO HÍBRIDA K-MEANS + KNN

Combina K-Means e KNN:
1. K-Means agrupa boxes em clusters (visão global)
2. Para cada box suspeito, KNN vota apenas entre vizinhos DO MESMO CLUSTER

Vantagens:
- K-Means garante que vizinhos são globalmente similares (mesmo cluster)
- KNN garante que são localmente similares (mais próximos)
- Mais robusto que usar apenas um dos métodos
- Computacionalmente mais eficiente que KNN global

Pipeline:
1. Relabel por confiança > 0.9 (baseline)
2. Relabel por K-Means + KNN híbrido
3. Spatial Refinement para boxes contaminados espacialmente
4. Filtragem seletiva (opcional)
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

# FAISS para clustering e busca eficiente
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] FAISS não disponível. Usando sklearn como fallback.")
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors


# ============================================================
# FUNÇÕES DE SPATIAL REFINEMENT
# ============================================================

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
    
    stats = {
        'total_boxes': len(boxes),
        'high_contamination': 0,
        'refinements_applied': 0,
    }
    
    for i, box_i in enumerate(boxes):
        difficulty = compute_box_difficulty(box_i, boxes, box_i_idx=i)
        
        if difficulty < difficulty_threshold:
            continue
        
        stats['high_contamination'] += 1
        
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
                stats['refinements_applied'] += 1
    
    return refined_labels, stats


@HOOKS.register_module()
class VCNCHybridKMeansKNNHook(Hook):
    """
    VCNC Híbrido: K-Means + KNN
    
    Fluxo do método híbrido:
    1. K-Means agrupa todos os boxes em clusters
    2. Para cada cluster:
       a. Identifica âncoras e suspeitos
       b. Para cada suspeito, usa KNN para encontrar os K vizinhos mais próximos
          DENTRO DO MESMO CLUSTER
       c. Vota baseado nos labels das âncoras vizinhas
    
    Isso combina:
    - Visão global do K-Means (boxes no mesmo cluster são similares)
    - Precisão local do KNN (vizinhos mais próximos têm mais peso)
    """
    
    def __init__(self,
                 # Configuração geral
                 warmup_epochs: int = 1,
                 num_classes: int = 20,
                 
                 # === ETAPA 1: Relabel por confiança (Baseline) ===
                 enable_confidence_relabel: bool = True,
                 relabel_confidence_threshold: float = 0.9,
                 
                 # === ETAPA 2: Híbrido K-Means + KNN ===
                 enable_hybrid_relabel: bool = True,
                 use_softmax_as_embedding: bool = True,
                 
                 # Parâmetros do K-Means
                 n_clusters: int = 150,
                 
                 # Parâmetros do KNN (dentro do cluster)
                 knn_k: int = 10,                     # K vizinhos dentro do cluster
                 knn_min_anchors: int = 3,            # Mínimo de âncoras vizinhas
                 knn_consensus_threshold: float = 0.6, # Consenso mínimo
                 knn_distance_weighted: bool = True,   # Ponderar por similaridade
                 
                 # Critérios progressivos
                 progressive_epochs: int = 4,
                 
                 # Conservador (épocas iniciais)
                 early_anchor_gmm_threshold: float = 0.15,
                 early_anchor_pred_agreement: float = 0.85,
                 early_anchor_confidence: float = 0.9,
                 early_suspect_gmm_threshold: float = 0.8,
                 
                 # Agressivo (épocas posteriores)
                 anchor_gmm_threshold: float = 0.4,
                 anchor_pred_agreement: float = 0.6,
                 anchor_confidence: float = 0.7,
                 suspect_gmm_threshold: float = 0.5,
                 
                 # === ETAPA 3: Spatial Refinement ===
                 enable_spatial_refinement: bool = True,
                 spatial_difficulty_threshold: float = 0.5,
                 
                 # === ETAPA 4: Filtragem Seletiva (opcional) ===
                 enable_selective_filtering: bool = False,
                 selective_filter_gmm_threshold: float = 0.5,
                 selective_filter_confidence_threshold: float = 0.7,
                 
                 # === ETAPA 5: Filtragem GMM (Baseline) ===
                 enable_gmm_filter: bool = False,
                 gmm_components: int = 4,
                 filter_gmm_threshold: float = 0.7,
                 
                 # Configuração do assigner
                 iou_assigner: float = 0.5,
                 
                 # Reload dataset
                 reload_dataset: bool = True,
                 
                 # Debug
                 debug: bool = True):
        
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        
        # Etapa 1
        self.enable_confidence_relabel = enable_confidence_relabel
        self.relabel_confidence_threshold = relabel_confidence_threshold
        
        # Etapa 2 - Híbrido
        self.enable_hybrid_relabel = enable_hybrid_relabel
        self.use_softmax_as_embedding = use_softmax_as_embedding
        self.n_clusters = n_clusters
        self.knn_k = knn_k
        self.knn_min_anchors = knn_min_anchors
        self.knn_consensus_threshold = knn_consensus_threshold
        self.knn_distance_weighted = knn_distance_weighted
        
        self.progressive_epochs = progressive_epochs
        
        # Conservador
        self.early_anchor_gmm_threshold = early_anchor_gmm_threshold
        self.early_anchor_pred_agreement = early_anchor_pred_agreement
        self.early_anchor_confidence = early_anchor_confidence
        self.early_suspect_gmm_threshold = early_suspect_gmm_threshold
        
        # Agressivo
        self.anchor_gmm_threshold = anchor_gmm_threshold
        self.anchor_pred_agreement = anchor_pred_agreement
        self.anchor_confidence = anchor_confidence
        self.suspect_gmm_threshold = suspect_gmm_threshold
        
        # Etapa 3
        self.enable_spatial_refinement = enable_spatial_refinement
        self.spatial_difficulty_threshold = spatial_difficulty_threshold
        
        # Etapa 4
        self.enable_selective_filtering = enable_selective_filtering
        self.selective_filter_gmm_threshold = selective_filter_gmm_threshold
        self.selective_filter_confidence_threshold = selective_filter_confidence_threshold
        
        # Etapa 5
        self.enable_gmm_filter = enable_gmm_filter
        self.gmm_components = gmm_components
        self.filter_gmm_threshold = filter_gmm_threshold
        
        # Assigner
        self.iou_assigner = iou_assigner
        
        # Reload
        self.reload_dataset = reload_dataset
        
        # Debug
        self.debug = debug
    
    def _get_current_criteria(self, epoch):
        """Retorna critérios baseado na época."""
        if epoch <= self.progressive_epochs:
            return {
                'anchor_gmm_threshold': self.early_anchor_gmm_threshold,
                'anchor_pred_agreement': self.early_anchor_pred_agreement,
                'anchor_confidence': self.early_anchor_confidence,
                'suspect_gmm_threshold': self.early_suspect_gmm_threshold,
                'phase': 'CONSERVADOR'
            }
        else:
            return {
                'anchor_gmm_threshold': self.anchor_gmm_threshold,
                'anchor_pred_agreement': self.anchor_pred_agreement,
                'anchor_confidence': self.anchor_confidence,
                'suspect_gmm_threshold': self.suspect_gmm_threshold,
                'phase': 'AGRESSIVO'
            }
    
    def _fit_gmm_per_class(self, scores_by_class):
        """Ajusta GMM para cada classe."""
        gmm_dict = {}
        
        for cls_id, scores in scores_by_class.items():
            if len(scores) < 10:
                continue
            
            scores_np = np.array(scores).reshape(-1, 1)
            
            try:
                n_comp = min(self.gmm_components, len(scores) // 5)
                if n_comp < 2:
                    n_comp = 2
                    
                gmm = GaussianMixture(
                    n_components=n_comp,
                    max_iter=100,
                    tol=1e-3,
                    reg_covar=1e-4,
                    random_state=42
                )
                gmm.fit(scores_np)
                
                low_conf_component = np.argmin(gmm.means_)
                gmm_dict[cls_id] = (gmm, low_conf_component)
                
            except Exception as e:
                if self.debug:
                    print(f"[VCNC-Hybrid] Erro GMM classe {cls_id}: {e}")
        
        return gmm_dict
    
    def _get_p_noise(self, score, cls_id, gmm_dict):
        """Calcula p_noise para um box."""
        if cls_id not in gmm_dict:
            return 0.5
        
        gmm, low_conf_comp = gmm_dict[cls_id]
        score_np = np.array([[score]])
        
        try:
            probs = gmm.predict_proba(score_np)
            return float(probs[0, low_conf_comp])
        except:
            return 0.5
    
    def _cluster_embeddings(self, embeddings, n_clusters):
        """Agrupa embeddings usando K-Means."""
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
    
    def _knn_vote_within_cluster(self, suspect_box, cluster_boxes, anchor_indices_in_cluster,
                                  embeddings_norm, k, distance_weighted=True):
        """
        Realiza votação KNN dentro de um cluster.
        
        Args:
            suspect_box: dict com dados do box suspeito
            cluster_boxes: lista de boxes no cluster
            anchor_indices_in_cluster: índices (no cluster) dos boxes que são âncoras
            embeddings_norm: embeddings normalizados de todos os boxes do cluster
            k: número de vizinhos
            distance_weighted: se True, pondera votos pela similaridade
        
        Returns:
            (label_sugerido, confiança, n_anchors) ou (None, 0, 0) se não há consenso
        """
        if len(anchor_indices_in_cluster) == 0:
            return None, 0.0, 0
        
        # Encontrar índice do suspeito no cluster
        suspect_idx_in_cluster = None
        for i, box in enumerate(cluster_boxes):
            if id(box) == id(suspect_box):
                suspect_idx_in_cluster = i
                break
        
        if suspect_idx_in_cluster is None:
            return None, 0.0, 0
        
        # Calcular similaridades com todas as âncoras do cluster
        suspect_emb = embeddings_norm[suspect_idx_in_cluster]
        
        anchor_similarities = []
        for anchor_idx in anchor_indices_in_cluster:
            anchor_emb = embeddings_norm[anchor_idx]
            sim = np.dot(suspect_emb, anchor_emb)
            anchor_similarities.append((anchor_idx, sim))
        
        # Ordenar por similaridade (maior primeiro)
        anchor_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Pegar os K vizinhos mais próximos (ou todos se < K)
        k_neighbors = anchor_similarities[:min(k, len(anchor_similarities))]
        
        if len(k_neighbors) < self.knn_min_anchors:
            return None, 0.0, len(k_neighbors)
        
        # Votar
        if distance_weighted:
            label_weights = defaultdict(float)
            for anchor_idx, sim in k_neighbors:
                label = cluster_boxes[anchor_idx]['gt_label']
                label_weights[label] += max(sim, 0)  # Garantir que peso é positivo
            
            if sum(label_weights.values()) == 0:
                return None, 0.0, len(k_neighbors)
            
            total_weight = sum(label_weights.values())
            best_label = max(label_weights, key=label_weights.get)
            confidence = label_weights[best_label] / total_weight
        else:
            labels = [cluster_boxes[idx]['gt_label'] for idx, _ in k_neighbors]
            label_counts = Counter(labels)
            best_label, count = label_counts.most_common(1)[0]
            confidence = count / len(labels)
        
        return best_label, confidence, len(k_neighbors)
    
    def before_train_epoch(self, runner):
        """Executa o pipeline completo antes de cada época."""
        epoch = runner.epoch + 1
        
        if epoch <= self.warmup_epochs:
            if self.debug:
                print(f"[VCNC-Hybrid] Época {epoch}: Warmup, pulando.")
            return
        
        if self.debug:
            print(f"\n[VCNC-Hybrid] ========== Época {epoch} ==========")
        
        # Reload dataset
        if self.reload_dataset:
            self._reload_datasets(runner)
        
        # Obter dataset
        dataloader = runner.train_loop.dataloader
        dataset = self._get_base_dataset(dataloader.dataset)
        
        if not hasattr(dataset, 'datasets'):
            print("[VCNC-Hybrid] ERRO: Esperado ConcatDataset")
            return
        
        datasets = dataset.datasets
        dataset_img_map = self._build_image_map(datasets)
        
        assigner = MaxIoUAssigner(
            pos_iou_thr=self.iou_assigner,
            neg_iou_thr=self.iou_assigner,
            min_pos_iou=self.iou_assigner,
            match_low_quality=False
        )
        
        # ============================================================
        # COLETA DE DADOS
        # ============================================================
        if self.debug:
            print("[VCNC-Hybrid] Coletando dados...")
        
        all_box_data = []
        scores_by_class = defaultdict(list)
        boxes_by_image = defaultdict(list)
        
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
                
                gt_labels = gt_instances.labels
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
                    
                    gt_label = gt_labels[gt_idx].item()
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
                        'relabeled_by': None,
                        'filtered': False,
                        'was_relabeled': False,
                    }
                    all_box_data.append(box_data)
                    scores_by_class[gt_label].append(score_gt)
                    boxes_by_image[img_path].append(box_data)
        
        if len(all_box_data) == 0:
            print("[VCNC-Hybrid] Nenhum box coletado!")
            return
        
        if self.debug:
            print(f"[VCNC-Hybrid] Coletados {len(all_box_data)} boxes em {len(boxes_by_image)} imagens")
        
        # ============================================================
        # ETAPA 1: RELABEL POR CONFIANÇA ALTA
        # ============================================================
        confidence_relabel_count = 0
        
        if self.enable_confidence_relabel:
            if self.debug:
                print(f"\n[VCNC-Hybrid] ETAPA 1: Relabel por confiança > {self.relabel_confidence_threshold}")
            
            for box in all_box_data:
                if (box['pred_score'] > self.relabel_confidence_threshold and 
                    box['pred_label'] != box['gt_label']):
                    
                    new_label = box['pred_label']
                    
                    self._apply_relabel(
                        datasets,
                        box['sub_idx'],
                        box['data_idx'],
                        box['gt_idx'],
                        new_label
                    )
                    
                    box['gt_label'] = new_label
                    box['score_gt'] = box['scores'][new_label].item()
                    box['relabeled_by'] = 'confidence'
                    box['was_relabeled'] = True
                    confidence_relabel_count += 1
            
            if self.debug:
                print(f"[VCNC-Hybrid] Relabelados por confiança: {confidence_relabel_count} "
                      f"({confidence_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 2: RELABEL POR K-MEANS + KNN HÍBRIDO
        # ============================================================
        hybrid_relabel_count = 0
        
        if self.enable_hybrid_relabel:
            if self.debug:
                print(f"\n[VCNC-Hybrid] ETAPA 2: Relabel por K-Means + KNN híbrido")
                print(f"[VCNC-Hybrid] Clusters: {self.n_clusters}, KNN K: {self.knn_k}")
            
            # Calcular GMM
            scores_by_class_updated = defaultdict(list)
            for box in all_box_data:
                scores_by_class_updated[box['gt_label']].append(box['score_gt'])
            
            gmm_dict = self._fit_gmm_per_class(scores_by_class_updated)
            
            for box in all_box_data:
                box['p_noise'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict)
            
            # Clustering com K-Means
            embeddings = np.array([box['embedding'] for box in all_box_data])
            cluster_ids = self._cluster_embeddings(embeddings, self.n_clusters)
            
            for i, box in enumerate(all_box_data):
                box['cluster_id'] = cluster_ids[i]
            
            criteria = self._get_current_criteria(epoch)
            
            if self.debug:
                print(f"[VCNC-Hybrid] Fase: {criteria['phase']}, Clusters únicos: {len(set(cluster_ids))}")
            
            # Organizar por cluster
            clusters = defaultdict(list)
            for box in all_box_data:
                clusters[box['cluster_id']].append(box)
            
            c_anchor_gmm = criteria['anchor_gmm_threshold']
            c_anchor_pred = criteria['anchor_pred_agreement']
            c_anchor_conf = criteria['anchor_confidence']
            c_suspect_gmm = criteria['suspect_gmm_threshold']
            
            # Estatísticas
            hybrid_stats = {
                'clusters_processed': 0,
                'clusters_skipped_small': 0,
                'clusters_skipped_no_anchors': 0,
                'suspects_processed': 0,
                'few_anchor_neighbors': 0,
                'no_consensus': 0,
                'same_label': 0,
                'relabeled': 0
            }
            
            # Processar cada cluster
            for cluster_id, cluster_boxes in clusters.items():
                if len(cluster_boxes) < 2:
                    hybrid_stats['clusters_skipped_small'] += 1
                    continue
                
                hybrid_stats['clusters_processed'] += 1
                
                # Identificar âncoras e suspeitos dentro do cluster
                anchor_indices = []
                suspect_boxes = []
                
                for i, box in enumerate(cluster_boxes):
                    is_clean = box['p_noise'] < c_anchor_gmm
                    model_agrees = box['score_gt'] > c_anchor_pred
                    high_confidence = box['pred_score'] > c_anchor_conf
                    
                    if is_clean and model_agrees and high_confidence:
                        anchor_indices.append(i)
                    elif box['p_noise'] >= c_suspect_gmm and box['relabeled_by'] is None:
                        suspect_boxes.append(box)
                
                if len(anchor_indices) == 0:
                    hybrid_stats['clusters_skipped_no_anchors'] += 1
                    continue
                
                # Normalizar embeddings do cluster para KNN
                cluster_embeddings = np.array([box['embedding'] for box in cluster_boxes])
                cluster_embeddings_norm = cluster_embeddings / (
                    np.linalg.norm(cluster_embeddings, axis=1, keepdims=True) + 1e-8
                )
                
                # Processar cada suspeito com KNN
                for suspect_box in suspect_boxes:
                    hybrid_stats['suspects_processed'] += 1
                    
                    # Votação KNN dentro do cluster
                    suggested_label, confidence, n_anchor_neighbors = self._knn_vote_within_cluster(
                        suspect_box,
                        cluster_boxes,
                        anchor_indices,
                        cluster_embeddings_norm,
                        k=self.knn_k,
                        distance_weighted=self.knn_distance_weighted
                    )
                    
                    if n_anchor_neighbors < self.knn_min_anchors:
                        hybrid_stats['few_anchor_neighbors'] += 1
                        continue
                    
                    if suggested_label is None or confidence < self.knn_consensus_threshold:
                        hybrid_stats['no_consensus'] += 1
                        continue
                    
                    if suggested_label == suspect_box['gt_label']:
                        hybrid_stats['same_label'] += 1
                        continue
                    
                    # Aplicar relabeling
                    self._apply_relabel(
                        datasets,
                        suspect_box['sub_idx'],
                        suspect_box['data_idx'],
                        suspect_box['gt_idx'],
                        suggested_label
                    )
                    
                    suspect_box['gt_label'] = suggested_label
                    suspect_box['score_gt'] = suspect_box['scores'][suggested_label].item()
                    suspect_box['relabeled_by'] = 'hybrid'
                    suspect_box['was_relabeled'] = True
                    hybrid_relabel_count += 1
                    hybrid_stats['relabeled'] += 1
            
            if self.debug:
                print(f"[VCNC-Hybrid] Estatísticas:")
                print(f"[VCNC-Hybrid]   - Clusters processados: {hybrid_stats['clusters_processed']}")
                print(f"[VCNC-Hybrid]   - Clusters pulados (pequenos): {hybrid_stats['clusters_skipped_small']}")
                print(f"[VCNC-Hybrid]   - Clusters pulados (sem âncoras): {hybrid_stats['clusters_skipped_no_anchors']}")
                print(f"[VCNC-Hybrid]   - Suspeitos processados: {hybrid_stats['suspects_processed']}")
                print(f"[VCNC-Hybrid]   - Poucas âncoras vizinhas: {hybrid_stats['few_anchor_neighbors']}")
                print(f"[VCNC-Hybrid]   - Sem consenso: {hybrid_stats['no_consensus']}")
                print(f"[VCNC-Hybrid]   - Mesmo label: {hybrid_stats['same_label']}")
                print(f"[VCNC-Hybrid]   - Relabelados: {hybrid_stats['relabeled']}")
                print(f"[VCNC-Hybrid] Total relabelados por híbrido: {hybrid_relabel_count} "
                      f"({hybrid_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 3: SPATIAL REFINEMENT
        # ============================================================
        spatial_relabel_count = 0
        spatial_stats = {'total_boxes': 0, 'high_contamination': 0, 'refinements_applied': 0}
        
        if self.enable_spatial_refinement:
            if self.debug:
                print(f"\n[VCNC-Hybrid] ETAPA 3: Spatial Refinement (threshold={self.spatial_difficulty_threshold})")
            
            for img_path, img_boxes in boxes_by_image.items():
                if len(img_boxes) < 2:
                    continue
                
                boxes_tensor = torch.stack([b['gt_bbox'] for b in img_boxes])
                pred_labels = torch.tensor([b['pred_label'] for b in img_boxes])
                pred_scores = torch.stack([b['scores'] for b in img_boxes])
                
                refined_labels, stats = spatial_aware_relabeling(
                    boxes_tensor,
                    pred_labels,
                    pred_scores,
                    difficulty_threshold=self.spatial_difficulty_threshold
                )
                
                spatial_stats['total_boxes'] += stats['total_boxes']
                spatial_stats['high_contamination'] += stats['high_contamination']
                spatial_stats['refinements_applied'] += stats['refinements_applied']
                
                for idx, box in enumerate(img_boxes):
                    if refined_labels[idx] != pred_labels[idx]:
                        if box['relabeled_by'] is None:
                            new_label = refined_labels[idx].item()
                            
                            self._apply_relabel(
                                datasets,
                                box['sub_idx'],
                                box['data_idx'],
                                box['gt_idx'],
                                new_label
                            )
                            
                            box['gt_label'] = new_label
                            box['score_gt'] = box['scores'][new_label].item()
                            box['relabeled_by'] = 'spatial'
                            box['was_relabeled'] = True
                            spatial_relabel_count += 1
            
            if self.debug:
                print(f"[VCNC-Hybrid] Boxes com alta contaminação: {spatial_stats['high_contamination']}")
                print(f"[VCNC-Hybrid] Relabelados por spatial: {spatial_relabel_count} "
                      f"({spatial_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 4: FILTRAGEM SELETIVA (OPCIONAL)
        # ============================================================
        selective_filter_count = 0
        
        if self.enable_selective_filtering:
            if self.debug:
                print(f"\n[VCNC-Hybrid] ETAPA 4: Filtragem Seletiva")
            
            scores_by_class_final = defaultdict(list)
            for box in all_box_data:
                scores_by_class_final[box['gt_label']].append(box['score_gt'])
            
            gmm_dict_final = self._fit_gmm_per_class(scores_by_class_final)
            
            for box in all_box_data:
                box['p_noise_final'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict_final)
            
            for box in all_box_data:
                is_suspect = box['p_noise_final'] >= self.selective_filter_gmm_threshold
                
                if not is_suspect:
                    continue
                
                if box['was_relabeled']:
                    continue
                
                if box['pred_score'] >= self.selective_filter_confidence_threshold:
                    continue
                
                self._apply_ignore_flag(
                    datasets,
                    box['sub_idx'],
                    box['data_idx'],
                    box['gt_idx']
                )
                box['filtered'] = True
                selective_filter_count += 1
            
            if self.debug:
                print(f"[VCNC-Hybrid] Filtrados: {selective_filter_count}")
        
        # ============================================================
        # ETAPA 5: FILTRAGEM GMM (OPCIONAL)
        # ============================================================
        gmm_filter_count = 0
        
        if self.enable_gmm_filter:
            if self.debug:
                print(f"\n[VCNC-Hybrid] ETAPA 5: Filtragem GMM")
            
            if not self.enable_selective_filtering:
                scores_by_class_final = defaultdict(list)
                for box in all_box_data:
                    scores_by_class_final[box['gt_label']].append(box['score_gt'])
                
                gmm_dict_final = self._fit_gmm_per_class(scores_by_class_final)
                
                for box in all_box_data:
                    box['p_noise_final'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict_final)
            
            for box in all_box_data:
                if box['filtered']:
                    continue
                    
                if box['p_noise_final'] > self.filter_gmm_threshold:
                    self._apply_ignore_flag(
                        datasets,
                        box['sub_idx'],
                        box['data_idx'],
                        box['gt_idx']
                    )
                    box['filtered'] = True
                    gmm_filter_count += 1
            
            if self.debug:
                print(f"[VCNC-Hybrid] Filtrados por GMM: {gmm_filter_count}")
        
        # ============================================================
        # ESTATÍSTICAS FINAIS
        # ============================================================
        total_filtered = selective_filter_count + gmm_filter_count
        
        if self.debug:
            total_relabels = confidence_relabel_count + hybrid_relabel_count + spatial_relabel_count
            print(f"\n[VCNC-Hybrid] ===== Resumo Época {epoch} =====")
            print(f"[VCNC-Hybrid] Total de boxes: {len(all_box_data)}")
            print(f"[VCNC-Hybrid] Relabel confiança: {confidence_relabel_count} ({confidence_relabel_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-Hybrid] Relabel híbrido: {hybrid_relabel_count} ({hybrid_relabel_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-Hybrid] Relabel spatial: {spatial_relabel_count} ({spatial_relabel_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-Hybrid] Total relabels: {total_relabels} ({total_relabels/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-Hybrid] Total filtrados: {total_filtered} ({total_filtered/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-Hybrid] ==========================================\n")
    
    def _reload_datasets(self, runner):
        """Recarrega os datasets."""
        try:
            ds = runner.train_loop.dataloader.dataset.dataset
            for i, subds in enumerate(ds.datasets):
                if hasattr(subds, '_fully_initialized'):
                    subds._fully_initialized = False
                if hasattr(subds, 'full_init'):
                    subds.full_init()
        except Exception as e:
            if self.debug:
                print(f"[VCNC-Hybrid] Erro ao recarregar: {e}")
    
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
            instance = datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]
            instance['bbox_label'] = new_label
        except Exception as e:
            if self.debug:
                print(f"[VCNC-Hybrid] Erro relabel: {e}")
    
    def _apply_ignore_flag(self, datasets, sub_idx, data_idx, gt_idx):
        try:
            instance = datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]
            instance['ignore_flag'] = 1
        except Exception as e:
            if self.debug:
                print(f"[VCNC-Hybrid] Erro ignore_flag: {e}")

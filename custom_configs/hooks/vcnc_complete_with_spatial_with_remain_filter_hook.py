"""
Visual Clustering Noise Correction Hook - VERSÃO COMPLETA COM SPATIAL REFINEMENT + FILTRAGEM SELETIVA

Combina:
1. Relabel por confiança > 0.9 (baseline)
2. Relabel por clustering visual (VCNC)
3. Spatial Refinement para boxes contaminados espacialmente
4. **NOVO** Filtragem seletiva de boxes suspeitos não corrigidos pelo VCNC
5. Filtragem GMM com ignore_flag (baseline) - opcional

ORDEM DE EXECUÇÃO:
1. Relabel por confiança alta
2. Relabel por clustering visual
3. Spatial Refinement (corrige boxes contaminados por vizinhos maiores)
4. Filtragem seletiva (remove boxes que VCNC identificou como ruidosos mas não corrigiu)
5. Filtragem GMM (opcional)
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

# FAISS para clustering eficiente
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] FAISS não disponível. Usando sklearn KMeans como fallback.")
    from sklearn.cluster import KMeans


# ============================================================
# FUNÇÕES DE SPATIAL REFINEMENT
# ============================================================

def compute_box_difficulty(box_i, all_boxes, box_i_idx=None):
    """
    Calcula dificuldade de um box baseado em contaminação espacial.
    
    Quando um box pequeno está sobreposto a um box maior, as features
    do box pequeno ficam "contaminadas" pelas features do maior.
    
    Args:
        box_i: tensor [4] (x1, y1, x2, y2) do box alvo
        all_boxes: tensor [N, 4] com todos os boxes da imagem
        box_i_idx: int, índice do box_i em all_boxes (para pular ele mesmo)
    
    Returns:
        float: difficulty score [0, 1]
    """
    difficulty = 0.0
    
    # Área do box_i
    area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
    
    if area_i <= 0:
        return 0.0
    
    for j, box_j in enumerate(all_boxes):
        if box_i_idx is not None and j == box_i_idx:
            continue
        
        # Calcula IoU
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
            # Se box_j é maior e tem overlap → contamina box_i
            if area_j > area_i:
                contamination = iou * min(area_j / area_i, 2.0)
                difficulty += min(contamination, 1.0)
            
            # Overlap muito alto sempre aumenta dificuldade
            if iou > 0.5:
                difficulty += 0.5
    
    return min(difficulty, 1.0)


def spatial_aware_relabeling(boxes, pred_labels, pred_scores, difficulty_threshold=0.5):
    """
    Refinamento de labels para boxes com alta contaminação espacial.
    
    Se um box pequeno está "contaminado" por um box maior e o modelo
    prediz a mesma classe do contaminador, usa a segunda melhor predição.
    
    Args:
        boxes: tensor [N, 4] - bounding boxes em xyxy
        pred_labels: tensor [N] - labels preditas
        pred_scores: tensor [N, num_classes] - scores de probabilidade (após softmax)
        difficulty_threshold: float - só refina se difficulty > threshold
    
    Returns:
        refined_labels: tensor [N] - labels refinadas
        refinement_stats: dict - estatísticas do refinamento
    """
    refined_labels = pred_labels.clone()
    
    stats = {
        'total_boxes': len(boxes),
        'high_contamination': 0,
        'refinements_applied': 0,
    }
    
    for i, box_i in enumerate(boxes):
        # Calcula contaminação deste box
        difficulty = compute_box_difficulty(box_i, boxes, box_i_idx=i)
        
        # Só refina se dificuldade > threshold
        if difficulty < difficulty_threshold:
            continue
        
        stats['high_contamination'] += 1
        
        # Encontra boxes que contaminam este
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
        
        # Ordena por influência
        contaminators.sort(key=lambda x: x[3], reverse=True)
        biggest_contaminator_idx = contaminators[0][0]
        biggest_contaminator_label = pred_labels[biggest_contaminator_idx]
        
        # Se prediz a MESMA classe do contaminador → provavelmente contaminado
        if pred_labels[i] == biggest_contaminator_label:
            # Pega top-2 predições
            top2_scores, top2_labels = pred_scores[i].topk(2)
            
            # Se segunda opção tem score razoável (>30% da primeira)
            if len(top2_scores) >= 2 and top2_scores[1] > top2_scores[0] * 0.3:
                refined_labels[i] = top2_labels[1]
                stats['refinements_applied'] += 1
    
    return refined_labels, stats


@HOOKS.register_module()
class VCNCCompleteWithSpatialRemainFilterHook(Hook):
    """
    VCNC Completo com Spatial Refinement + Filtragem Seletiva
    
    Fluxo:
    1. Relabel por confiança alta (baseline)
    2. Relabel por clustering visual (VCNC)
    3. Spatial Refinement (corrige contaminação espacial)
    4. **NOVO** Filtragem seletiva (remove boxes suspeitos não corrigidos)
    5. Filtragem GMM (baseline) - opcional
    """
    
    def __init__(self,
                 # Configuração geral
                 warmup_epochs: int = 1,
                 num_classes: int = 20,
                 
                 # === ETAPA 1: Relabel por confiança (Baseline) ===
                 enable_confidence_relabel: bool = True,
                 relabel_confidence_threshold: float = 0.9,
                 
                 # === ETAPA 2: Clustering Visual (VCNC) ===
                 enable_clustering_relabel: bool = True,
                 n_clusters: int = 150,
                 use_softmax_as_embedding: bool = True,
                 
                 # Critérios progressivos
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
                 
                 # === ETAPA 3: Spatial Refinement ===
                 enable_spatial_refinement: bool = True,
                 spatial_difficulty_threshold: float = 0.5,
                 
                 # === ETAPA 4: Filtragem Seletiva (NOVO) ===
                 enable_selective_filtering: bool = True,
                 selective_filter_gmm_threshold: float = 0.5,  # p_noise > threshold = suspeito
                 
                 # === ETAPA 5: Filtragem GMM (Baseline) ===
                 enable_gmm_filter: bool = False,  # Desabilitado por padrão (redundante com seletiva)
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
        
        # Etapa 2
        self.enable_clustering_relabel = enable_clustering_relabel
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
        
        # Etapa 3
        self.enable_spatial_refinement = enable_spatial_refinement
        self.spatial_difficulty_threshold = spatial_difficulty_threshold
        
        # Etapa 4 - NOVO
        self.enable_selective_filtering = enable_selective_filtering
        self.selective_filter_gmm_threshold = selective_filter_gmm_threshold
        
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
                    print(f"[VCNC-Spatial-V2] Erro GMM classe {cls_id}: {e}")
        
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
        """Agrupa embeddings usando FAISS ou KMeans."""
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
        """Executa o pipeline completo antes de cada época."""
        epoch = runner.epoch + 1
        
        if epoch <= self.warmup_epochs:
            if self.debug:
                print(f"[VCNC-Spatial-V2] Época {epoch}: Warmup, pulando.")
            return
        
        if self.debug:
            print(f"\n[VCNC-Spatial-V2] ========== Época {epoch} ==========")
        
        # Reload dataset
        if self.reload_dataset:
            self._reload_datasets(runner)
        
        # Obter dataset
        dataloader = runner.train_loop.dataloader
        dataset = self._get_base_dataset(dataloader.dataset)
        
        if not hasattr(dataset, 'datasets'):
            print("[VCNC-Spatial-V2] ERRO: Esperado ConcatDataset")
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
            print("[VCNC-Spatial-V2] Coletando dados...")
        
        all_box_data = []
        scores_by_class = defaultdict(list)
        
        # Armazenar dados por imagem para Spatial Refinement
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
                    # gt_bboxes pode ser HorizontalBoxes, então extraímos o tensor
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
                        'is_suspect': False,  # NOVO: marca se foi identificado como suspeito
                        'was_relabeled': False,  # NOVO: marca se foi relabelado (por qualquer método)
                    }
                    all_box_data.append(box_data)
                    scores_by_class[gt_label].append(score_gt)
                    boxes_by_image[img_path].append(box_data)
        
        if len(all_box_data) == 0:
            print("[VCNC-Spatial-V2] Nenhum box coletado!")
            return
        
        if self.debug:
            print(f"[VCNC-Spatial-V2] Coletados {len(all_box_data)} boxes em {len(boxes_by_image)} imagens")
        
        # ============================================================
        # ETAPA 1: RELABEL POR CONFIANÇA ALTA
        # ============================================================
        confidence_relabel_count = 0
        
        if self.enable_confidence_relabel:
            if self.debug:
                print(f"\n[VCNC-Spatial-V2] ETAPA 1: Relabel por confiança > {self.relabel_confidence_threshold}")
            
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
                print(f"[VCNC-Spatial-V2] Relabelados por confiança: {confidence_relabel_count} "
                      f"({confidence_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 2: RELABEL POR CLUSTERING VISUAL
        # ============================================================
        clustering_relabel_count = 0
        
        if self.enable_clustering_relabel:
            if self.debug:
                print(f"\n[VCNC-Spatial-V2] ETAPA 2: Relabel por clustering visual")
            
            # Recalcular GMM
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
            
            if self.debug:
                print(f"[VCNC-Spatial-V2] Fase: {criteria['phase']}, Clusters: {len(set(cluster_ids))}")
            
            clusters = defaultdict(list)
            for box in all_box_data:
                clusters[box['cluster_id']].append(box)
            
            c_anchor_gmm = criteria['anchor_gmm_threshold']
            c_anchor_pred = criteria['anchor_pred_agreement']
            c_anchor_conf = criteria['anchor_confidence']
            c_suspect_gmm = criteria['suspect_gmm_threshold']
            c_similarity = criteria['similarity_threshold']
            c_consensus = criteria['cluster_consensus']
            
            # Marcar suspeitos ANTES do relabeling
            for box in all_box_data:
                if box['p_noise'] >= c_suspect_gmm:
                    box['is_suspect'] = True
            
            for cluster_id, cluster_boxes in clusters.items():
                if len(cluster_boxes) < 2:
                    continue
                
                anchors = []
                for box in cluster_boxes:
                    is_clean = box['p_noise'] < c_anchor_gmm
                    model_agrees = box['score_gt'] > c_anchor_pred
                    high_confidence = box['pred_score'] > c_anchor_conf
                    
                    if is_clean and model_agrees and high_confidence:
                        anchors.append(box)
                
                if len(anchors) == 0:
                    continue
                
                anchor_labels = [a['gt_label'] for a in anchors]
                label_counts = Counter(anchor_labels)
                dominant_label, count = label_counts.most_common(1)[0]
                consensus_ratio = count / len(anchors)
                
                if consensus_ratio < c_consensus:
                    continue
                
                anchor_embeddings = np.array([a['embedding'] for a in anchors])
                anchor_mean = anchor_embeddings.mean(axis=0)
                anchor_mean_norm = anchor_mean / (np.linalg.norm(anchor_mean) + 1e-8)
                
                anchor_ids = set(id(a) for a in anchors)
                
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
                        box['was_relabeled'] = True
                        clustering_relabel_count += 1
            
            if self.debug:
                print(f"[VCNC-Spatial-V2] Relabelados por clustering: {clustering_relabel_count} "
                      f"({clustering_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 3: SPATIAL REFINEMENT
        # ============================================================
        spatial_relabel_count = 0
        spatial_stats = {'total_boxes': 0, 'high_contamination': 0, 'refinements_applied': 0}
        
        if self.enable_spatial_refinement:
            if self.debug:
                print(f"\n[VCNC-Spatial-V2] ETAPA 3: Spatial Refinement (threshold={self.spatial_difficulty_threshold})")
            
            # Processar por imagem
            for img_path, img_boxes in boxes_by_image.items():
                if len(img_boxes) < 2:
                    continue
                
                # Pegar boxes e scores
                boxes_tensor = torch.stack([b['gt_bbox'] for b in img_boxes])
                pred_labels = torch.tensor([b['pred_label'] for b in img_boxes])
                pred_scores = torch.stack([b['scores'] for b in img_boxes])
                
                # Aplicar spatial refinement
                refined_labels, stats = spatial_aware_relabeling(
                    boxes_tensor,
                    pred_labels,
                    pred_scores,
                    difficulty_threshold=self.spatial_difficulty_threshold
                )
                
                spatial_stats['total_boxes'] += stats['total_boxes']
                spatial_stats['high_contamination'] += stats['high_contamination']
                spatial_stats['refinements_applied'] += stats['refinements_applied']
                
                # Aplicar refinamentos
                for idx, box in enumerate(img_boxes):
                    if refined_labels[idx] != pred_labels[idx]:
                        # Só aplica se ainda não foi relabelado
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
                print(f"[VCNC-Spatial-V2] Boxes com alta contaminação: {spatial_stats['high_contamination']}")
                print(f"[VCNC-Spatial-V2] Relabelados por spatial: {spatial_relabel_count} "
                      f"({spatial_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 4: FILTRAGEM SELETIVA (NOVO)
        # Filtra apenas boxes que:
        # 1. Foram identificados como suspeitos (p_noise alto)
        # 2. NÃO foram corrigidos por nenhum método (confidence, clustering, spatial)
        # ============================================================
        selective_filter_count = 0
        
        if self.enable_selective_filtering:
            if self.debug:
                print(f"\n[VCNC-Spatial-V2] ETAPA 4: Filtragem Seletiva (suspeitos não corrigidos)")
            
            # Recalcular GMM com labels atualizados (pós-relabeling)
            scores_by_class_final = defaultdict(list)
            for box in all_box_data:
                scores_by_class_final[box['gt_label']].append(box['score_gt'])
            
            gmm_dict_final = self._fit_gmm_per_class(scores_by_class_final)
            
            # Atualizar p_noise com GMM final
            for box in all_box_data:
                box['p_noise_final'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict_final)
            
            # Contar estatísticas
            total_suspects = 0
            suspects_relabeled = 0
            suspects_not_relabeled = 0
            
            for box in all_box_data:
                # Verifica se é suspeito pelo GMM atualizado
                is_suspect_now = box['p_noise_final'] >= self.selective_filter_gmm_threshold
                
                # Ou se foi marcado como suspeito antes do relabeling
                was_suspect_before = box.get('is_suspect', False)
                
                if is_suspect_now or was_suspect_before:
                    total_suspects += 1
                    
                    if box['was_relabeled']:
                        suspects_relabeled += 1
                    else:
                        suspects_not_relabeled += 1
                        
                        # FILTRA: box suspeito que não foi corrigido
                        self._apply_ignore_flag(
                            datasets,
                            box['sub_idx'],
                            box['data_idx'],
                            box['gt_idx']
                        )
                        box['filtered'] = True
                        selective_filter_count += 1
            
            if self.debug:
                print(f"[VCNC-Spatial-V2] Total suspeitos identificados: {total_suspects} "
                      f"({total_suspects/len(all_box_data)*100:.2f}%)")
                print(f"[VCNC-Spatial-V2]   - Corrigidos (relabelados): {suspects_relabeled} "
                      f"({suspects_relabeled/max(total_suspects,1)*100:.2f}% dos suspeitos)")
                print(f"[VCNC-Spatial-V2]   - Não corrigidos (FILTRADOS): {suspects_not_relabeled} "
                      f"({suspects_not_relabeled/max(total_suspects,1)*100:.2f}% dos suspeitos)")
                print(f"[VCNC-Spatial-V2] Filtrados (ignore_flag=1): {selective_filter_count} "
                      f"({selective_filter_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 5: FILTRAGEM GMM (Baseline) - OPCIONAL
        # Mantida para compatibilidade, mas desabilitada por padrão
        # ============================================================
        gmm_filter_count = 0
        
        if self.enable_gmm_filter:
            if self.debug:
                print(f"\n[VCNC-Spatial-V2] ETAPA 5: Filtragem GMM adicional (threshold={self.filter_gmm_threshold})")
            
            # Recalcular GMM se não foi calculado na etapa anterior
            if not self.enable_selective_filtering:
                scores_by_class_final = defaultdict(list)
                for box in all_box_data:
                    scores_by_class_final[box['gt_label']].append(box['score_gt'])
                
                gmm_dict_final = self._fit_gmm_per_class(scores_by_class_final)
                
                for box in all_box_data:
                    box['p_noise_final'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict_final)
            
            for box in all_box_data:
                # Só filtra se ainda não foi filtrado
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
                print(f"[VCNC-Spatial-V2] Filtrados por GMM adicional: {gmm_filter_count} "
                      f"({gmm_filter_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ESTATÍSTICAS FINAIS
        # ============================================================
        total_filtered = selective_filter_count + gmm_filter_count
        
        if self.debug:
            total_relabels = confidence_relabel_count + clustering_relabel_count + spatial_relabel_count
            print(f"\n[VCNC-Spatial-V2] ===== Resumo Época {epoch} =====")
            print(f"[VCNC-Spatial-V2] Total de boxes: {len(all_box_data)}")
            print(f"[VCNC-Spatial-V2] Relabel confiança: {confidence_relabel_count} ({confidence_relabel_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-Spatial-V2] Relabel clustering: {clustering_relabel_count} ({clustering_relabel_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-Spatial-V2] Relabel spatial: {spatial_relabel_count} ({spatial_relabel_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-Spatial-V2] Total relabels: {total_relabels} ({total_relabels/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-Spatial-V2] Filtrados (seletiva): {selective_filter_count} ({selective_filter_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-Spatial-V2] Filtrados (GMM): {gmm_filter_count} ({gmm_filter_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-Spatial-V2] Total filtrados: {total_filtered} ({total_filtered/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-Spatial-V2] ==========================================\n")
    
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
                print(f"[VCNC-Spatial-V2] Erro ao recarregar: {e}")
    
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
                print(f"[VCNC-Spatial-V2] Erro relabel: {e}")
    
    def _apply_ignore_flag(self, datasets, sub_idx, data_idx, gt_idx):
        try:
            instance = datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]
            instance['ignore_flag'] = 1
        except Exception as e:
            if self.debug:
                print(f"[VCNC-Spatial-V2] Erro ignore_flag: {e}")
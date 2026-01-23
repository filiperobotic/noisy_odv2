"""
Visual Clustering Noise Correction Hook - VERSÃO KNN COM SPATIAL REFINEMENT

Diferença principal em relação à versão K-Means:
- Em vez de criar clusters globais, usa KNN para encontrar os K vizinhos 
  mais próximos de cada box suspeito
- Decisão de relabeling é baseada na votação dos vizinhos âncoras
- Mais local e flexível que K-Means

Combina:
1. Relabel por confiança > 0.9 (baseline)
2. Relabel por KNN visual (VCNC-KNN)
3. Spatial Refinement para boxes contaminados espacialmente
4. Filtragem seletiva refinada (opcional)
5. Filtragem GMM com ignore_flag (baseline) - opcional
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

# FAISS para busca eficiente de vizinhos
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] FAISS não disponível. Usando sklearn NearestNeighbors como fallback.")
    from sklearn.neighbors import NearestNeighbors


# ============================================================
# FUNÇÕES DE SPATIAL REFINEMENT
# ============================================================

def compute_box_difficulty(box_i, all_boxes, box_i_idx=None):
    """
    Calcula dificuldade de um box baseado em contaminação espacial.
    """
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
    """
    Refinamento de labels para boxes com alta contaminação espacial.
    """
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
class VCNCKNNSpatialHook(Hook):
    """
    VCNC com KNN + Spatial Refinement
    
    Diferença do K-Means:
    - K-Means: agrupa todos os boxes em clusters, depois vota dentro de cada cluster
    - KNN: para cada box suspeito, encontra os K vizinhos mais próximos e vota
    
    Vantagens do KNN:
    - Decisões mais locais (não depende de estrutura global de clusters)
    - Mais flexível com distribuições não-esféricas
    - Pode ponderar vizinhos pela distância
    """
    
    def __init__(self,
                 # Configuração geral
                 warmup_epochs: int = 1,
                 num_classes: int = 20,
                 
                 # === ETAPA 1: Relabel por confiança (Baseline) ===
                 enable_confidence_relabel: bool = True,
                 relabel_confidence_threshold: float = 0.9,
                 
                 # === ETAPA 2: KNN Visual (VCNC-KNN) ===
                 enable_knn_relabel: bool = True,
                 use_softmax_as_embedding: bool = True,
                 
                 # Parâmetros do KNN
                 knn_k: int = 15,                    # Número de vizinhos a considerar
                 knn_min_anchors: int = 5,           # Mínimo de âncoras entre os vizinhos
                 knn_consensus_threshold: float = 0.6,  # Fração mínima de âncoras concordando
                 knn_distance_weighted: bool = True,    # Ponderar voto pela distância
                 
                 # Critérios progressivos para definir âncoras/suspeitos
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
        
        # Etapa 2 - KNN
        self.enable_knn_relabel = enable_knn_relabel
        self.use_softmax_as_embedding = use_softmax_as_embedding
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
                    print(f"[VCNC-KNN] Erro GMM classe {cls_id}: {e}")
        
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
    
    def _build_knn_index(self, embeddings):
        """
        Constrói índice para busca de vizinhos mais próximos.
        Usa FAISS se disponível, senão sklearn.
        """
        N, D = embeddings.shape
        
        # Normalizar embeddings para usar similaridade de cosseno
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        embeddings_norm = embeddings_norm.astype(np.float32)
        
        if FAISS_AVAILABLE:
            # Usar produto interno (equivalente a cosseno com vetores normalizados)
            index = faiss.IndexFlatIP(D)
            index.add(embeddings_norm)
            return index, embeddings_norm
        else:
            # Fallback para sklearn
            nn = NearestNeighbors(n_neighbors=self.knn_k + 1, metric='cosine')
            nn.fit(embeddings_norm)
            return nn, embeddings_norm
    
    def _find_knn(self, index, embeddings_norm, query_idx, k):
        """
        Encontra os K vizinhos mais próximos de um ponto.
        Retorna índices e similaridades (distâncias convertidas).
        """
        query = embeddings_norm[query_idx:query_idx+1]
        
        if FAISS_AVAILABLE:
            # FAISS retorna similaridades (produto interno)
            similarities, indices = index.search(query, k + 1)
            similarities = similarities[0]
            indices = indices[0]
            
            # Remover o próprio ponto (sempre é o mais similar a si mesmo)
            mask = indices != query_idx
            indices = indices[mask][:k]
            similarities = similarities[mask][:k]
            
        else:
            # sklearn retorna distâncias de cosseno
            distances, indices = index.kneighbors(query, n_neighbors=k + 1)
            distances = distances[0]
            indices = indices[0]
            
            # Remover o próprio ponto
            mask = indices != query_idx
            indices = indices[mask][:k]
            distances = distances[mask][:k]
            
            # Converter distância de cosseno para similaridade
            similarities = 1 - distances
        
        return indices, similarities
    
    def _knn_vote(self, neighbor_indices, neighbor_similarities, all_box_data, 
                  anchor_mask, distance_weighted=True):
        """
        Realiza votação baseada nos vizinhos âncoras.
        
        Args:
            neighbor_indices: índices dos K vizinhos
            neighbor_similarities: similaridades com cada vizinho
            all_box_data: lista com dados de todos os boxes
            anchor_mask: máscara booleana indicando quais boxes são âncoras
            distance_weighted: se True, pondera votos pela similaridade
        
        Returns:
            (label_sugerido, confiança, n_anchors) ou (None, 0, 0) se não há consenso
        """
        # Filtrar apenas vizinhos que são âncoras
        anchor_neighbors = []
        anchor_similarities = []
        
        for idx, sim in zip(neighbor_indices, neighbor_similarities):
            if anchor_mask[idx]:
                anchor_neighbors.append(idx)
                anchor_similarities.append(sim)
        
        if len(anchor_neighbors) < self.knn_min_anchors:
            return None, 0.0, len(anchor_neighbors)
        
        # Coletar labels das âncoras vizinhas
        anchor_labels = [all_box_data[idx]['gt_label'] for idx in anchor_neighbors]
        
        if distance_weighted:
            # Votação ponderada pela similaridade
            label_weights = defaultdict(float)
            for label, sim in zip(anchor_labels, anchor_similarities):
                label_weights[label] += sim
            
            total_weight = sum(label_weights.values())
            best_label = max(label_weights, key=label_weights.get)
            confidence = label_weights[best_label] / total_weight
        else:
            # Votação simples (maioria)
            label_counts = Counter(anchor_labels)
            best_label, count = label_counts.most_common(1)[0]
            confidence = count / len(anchor_labels)
        
        return best_label, confidence, len(anchor_neighbors)
    
    def before_train_epoch(self, runner):
        """Executa o pipeline completo antes de cada época."""
        epoch = runner.epoch + 1
        
        if epoch <= self.warmup_epochs:
            if self.debug:
                print(f"[VCNC-KNN] Época {epoch}: Warmup, pulando.")
            return
        
        if self.debug:
            print(f"\n[VCNC-KNN] ========== Época {epoch} ==========")
        
        # Reload dataset
        if self.reload_dataset:
            self._reload_datasets(runner)
        
        # Obter dataset
        dataloader = runner.train_loop.dataloader
        dataset = self._get_base_dataset(dataloader.dataset)
        
        if not hasattr(dataset, 'datasets'):
            print("[VCNC-KNN] ERRO: Esperado ConcatDataset")
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
            print("[VCNC-KNN] Coletando dados...")
        
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
            print("[VCNC-KNN] Nenhum box coletado!")
            return
        
        if self.debug:
            print(f"[VCNC-KNN] Coletados {len(all_box_data)} boxes em {len(boxes_by_image)} imagens")
        
        # ============================================================
        # ETAPA 1: RELABEL POR CONFIANÇA ALTA
        # ============================================================
        confidence_relabel_count = 0
        
        if self.enable_confidence_relabel:
            if self.debug:
                print(f"\n[VCNC-KNN] ETAPA 1: Relabel por confiança > {self.relabel_confidence_threshold}")
            
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
                print(f"[VCNC-KNN] Relabelados por confiança: {confidence_relabel_count} "
                      f"({confidence_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 2: RELABEL POR KNN VISUAL
        # ============================================================
        knn_relabel_count = 0
        
        if self.enable_knn_relabel:
            if self.debug:
                print(f"\n[VCNC-KNN] ETAPA 2: Relabel por KNN visual (K={self.knn_k})")
            
            # Calcular GMM para identificar âncoras e suspeitos
            scores_by_class_updated = defaultdict(list)
            for box in all_box_data:
                scores_by_class_updated[box['gt_label']].append(box['score_gt'])
            
            gmm_dict = self._fit_gmm_per_class(scores_by_class_updated)
            
            for box in all_box_data:
                box['p_noise'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict)
            
            # Obter critérios para a época atual
            criteria = self._get_current_criteria(epoch)
            
            if self.debug:
                print(f"[VCNC-KNN] Fase: {criteria['phase']}")
            
            c_anchor_gmm = criteria['anchor_gmm_threshold']
            c_anchor_pred = criteria['anchor_pred_agreement']
            c_anchor_conf = criteria['anchor_confidence']
            c_suspect_gmm = criteria['suspect_gmm_threshold']
            
            # Identificar âncoras e suspeitos
            anchor_mask = np.zeros(len(all_box_data), dtype=bool)
            suspect_indices = []
            
            for i, box in enumerate(all_box_data):
                is_clean = box['p_noise'] < c_anchor_gmm
                model_agrees = box['score_gt'] > c_anchor_pred
                high_confidence = box['pred_score'] > c_anchor_conf
                
                if is_clean and model_agrees and high_confidence:
                    anchor_mask[i] = True
                elif box['p_noise'] >= c_suspect_gmm and box['relabeled_by'] is None:
                    suspect_indices.append(i)
            
            n_anchors = anchor_mask.sum()
            n_suspects = len(suspect_indices)
            
            if self.debug:
                print(f"[VCNC-KNN] Âncoras: {n_anchors} ({n_anchors/len(all_box_data)*100:.2f}%)")
                print(f"[VCNC-KNN] Suspeitos: {n_suspects} ({n_suspects/len(all_box_data)*100:.2f}%)")
            
            if n_anchors < self.knn_min_anchors:
                if self.debug:
                    print(f"[VCNC-KNN] Poucas âncoras, pulando relabeling por KNN")
            else:
                # Construir índice KNN
                embeddings = np.array([box['embedding'] for box in all_box_data])
                knn_index, embeddings_norm = self._build_knn_index(embeddings)
                
                # Processar cada box suspeito
                knn_stats = {
                    'processed': 0,
                    'few_anchor_neighbors': 0,
                    'no_consensus': 0,
                    'same_label': 0,
                    'relabeled': 0
                }
                
                for suspect_idx in suspect_indices:
                    box = all_box_data[suspect_idx]
                    knn_stats['processed'] += 1
                    
                    # Encontrar K vizinhos mais próximos
                    neighbor_indices, neighbor_similarities = self._find_knn(
                        knn_index, embeddings_norm, suspect_idx, self.knn_k
                    )
                    
                    # Votar baseado nos vizinhos âncoras
                    suggested_label, confidence, n_anchor_neighbors = self._knn_vote(
                        neighbor_indices, 
                        neighbor_similarities,
                        all_box_data,
                        anchor_mask,
                        distance_weighted=self.knn_distance_weighted
                    )
                    
                    # Verificar se há consenso suficiente
                    if n_anchor_neighbors < self.knn_min_anchors:
                        knn_stats['few_anchor_neighbors'] += 1
                        continue
                    
                    if confidence < self.knn_consensus_threshold:
                        knn_stats['no_consensus'] += 1
                        continue
                    
                    if suggested_label == box['gt_label']:
                        knn_stats['same_label'] += 1
                        continue
                    
                    # Aplicar relabeling
                    self._apply_relabel(
                        datasets,
                        box['sub_idx'],
                        box['data_idx'],
                        box['gt_idx'],
                        suggested_label
                    )
                    
                    box['gt_label'] = suggested_label
                    box['score_gt'] = box['scores'][suggested_label].item()
                    box['relabeled_by'] = 'knn'
                    box['was_relabeled'] = True
                    knn_relabel_count += 1
                    knn_stats['relabeled'] += 1
                
                if self.debug:
                    print(f"[VCNC-KNN] Estatísticas KNN:")
                    print(f"[VCNC-KNN]   - Processados: {knn_stats['processed']}")
                    print(f"[VCNC-KNN]   - Poucas âncoras vizinhas: {knn_stats['few_anchor_neighbors']}")
                    print(f"[VCNC-KNN]   - Sem consenso: {knn_stats['no_consensus']}")
                    print(f"[VCNC-KNN]   - Mesmo label: {knn_stats['same_label']}")
                    print(f"[VCNC-KNN]   - Relabelados: {knn_stats['relabeled']}")
            
            if self.debug:
                print(f"[VCNC-KNN] Relabelados por KNN: {knn_relabel_count} "
                      f"({knn_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 3: SPATIAL REFINEMENT
        # ============================================================
        spatial_relabel_count = 0
        spatial_stats = {'total_boxes': 0, 'high_contamination': 0, 'refinements_applied': 0}
        
        if self.enable_spatial_refinement:
            if self.debug:
                print(f"\n[VCNC-KNN] ETAPA 3: Spatial Refinement (threshold={self.spatial_difficulty_threshold})")
            
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
                print(f"[VCNC-KNN] Boxes com alta contaminação: {spatial_stats['high_contamination']}")
                print(f"[VCNC-KNN] Relabelados por spatial: {spatial_relabel_count} "
                      f"({spatial_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 4: FILTRAGEM SELETIVA (OPCIONAL)
        # ============================================================
        selective_filter_count = 0
        
        if self.enable_selective_filtering:
            if self.debug:
                print(f"\n[VCNC-KNN] ETAPA 4: Filtragem Seletiva")
                print(f"[VCNC-KNN] Critérios: p_noise >= {self.selective_filter_gmm_threshold} AND "
                      f"não relabelado AND pred_score < {self.selective_filter_confidence_threshold}")
            
            # Recalcular GMM
            scores_by_class_final = defaultdict(list)
            for box in all_box_data:
                scores_by_class_final[box['gt_label']].append(box['score_gt'])
            
            gmm_dict_final = self._fit_gmm_per_class(scores_by_class_final)
            
            for box in all_box_data:
                box['p_noise_final'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict_final)
            
            total_suspects = 0
            suspects_relabeled = 0
            suspects_confident = 0
            
            for box in all_box_data:
                is_suspect = box['p_noise_final'] >= self.selective_filter_gmm_threshold
                
                if not is_suspect:
                    continue
                
                total_suspects += 1
                
                if box['was_relabeled']:
                    suspects_relabeled += 1
                    continue
                
                if box['pred_score'] >= self.selective_filter_confidence_threshold:
                    suspects_confident += 1
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
                print(f"[VCNC-KNN] Total suspeitos: {total_suspects}")
                print(f"[VCNC-KNN]   - Corrigidos: {suspects_relabeled}")
                print(f"[VCNC-KNN]   - Confiantes: {suspects_confident}")
                print(f"[VCNC-KNN]   - Filtrados: {selective_filter_count}")
        
        # ============================================================
        # ETAPA 5: FILTRAGEM GMM (OPCIONAL)
        # ============================================================
        gmm_filter_count = 0
        
        if self.enable_gmm_filter:
            if self.debug:
                print(f"\n[VCNC-KNN] ETAPA 5: Filtragem GMM (threshold={self.filter_gmm_threshold})")
            
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
                print(f"[VCNC-KNN] Filtrados por GMM: {gmm_filter_count}")
        
        # ============================================================
        # ESTATÍSTICAS FINAIS
        # ============================================================
        total_filtered = selective_filter_count + gmm_filter_count
        
        if self.debug:
            total_relabels = confidence_relabel_count + knn_relabel_count + spatial_relabel_count
            print(f"\n[VCNC-KNN] ===== Resumo Época {epoch} =====")
            print(f"[VCNC-KNN] Total de boxes: {len(all_box_data)}")
            print(f"[VCNC-KNN] Relabel confiança: {confidence_relabel_count} ({confidence_relabel_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-KNN] Relabel KNN: {knn_relabel_count} ({knn_relabel_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-KNN] Relabel spatial: {spatial_relabel_count} ({spatial_relabel_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-KNN] Total relabels: {total_relabels} ({total_relabels/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-KNN] Total filtrados: {total_filtered} ({total_filtered/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-KNN] ==========================================\n")
    
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
                print(f"[VCNC-KNN] Erro ao recarregar: {e}")
    
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
                print(f"[VCNC-KNN] Erro relabel: {e}")
    
    def _apply_ignore_flag(self, datasets, sub_idx, data_idx, gt_idx):
        try:
            instance = datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]
            instance['ignore_flag'] = 1
        except Exception as e:
            if self.debug:
                print(f"[VCNC-KNN] Erro ignore_flag: {e}")

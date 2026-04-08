"""
Estratégia L: Propagação de Ruído (Noise Propagation)

Ideia: Quando detectamos uma amostra ruidosa com ALTA confiança,
buscamos outras amostras muito parecidas no espaço de embeddings.
Essas amostras vizinhas provavelmente também são ruído do mesmo tipo.

Lógica:
1. Detecta amostras ruidosas com critérios normais
2. Para cada amostra ruidosa com alta confiança:
   - Busca K vizinhos mais próximos no espaço de embeddings
   - Se vizinho é muito similar (> threshold) → também marca como ruído
3. Aplica ação (filter ou random) nas amostras detectadas + propagadas

Benefício: Aumenta o "recall" da detecção de ruído assimétrico.
Se gato→cachorro é ruído, outros gatos muito parecidos provavelmente também são.
"""

from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import torch
from mmdet.models.task_modules.assigners import MaxIoUAssigner
from collections import Counter, defaultdict
import numpy as np
import random
from sklearn.mixture import GaussianMixture

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

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
class VCNCNoisePropagationHook(Hook):
    """
    VCNC Estratégia L: Propagação de Ruído
    
    Quando detecta ruído com alta confiança, busca vizinhos similares
    e também os marca como ruído.
    
    Combina com Random Relabeling:
    - Ruído detectado (original ou propagado) com margem pequena → Random Label
    - Ruído detectado com margem grande → Relabel para dominante
    """
    
    def __init__(self,
                 warmup_epochs: int = 1,
                 num_classes: int = 20,
                 
                 # Threshold de margem
                 margin_threshold: float = 0.2,
                 
                 # Propagação de ruído
                 enable_propagation: bool = True,
                 propagation_k: int = 5,  # Quantos vizinhos buscar
                 propagation_similarity_threshold: float = 0.9,  # Similaridade mínima para propagar
                 propagation_confidence_threshold: float = 0.8,  # p_noise mínimo para propagar a partir dessa amostra
                 
                 # Ação para ruído assimétrico (margem pequena)
                 use_random_label: bool = True,  # Se False, filtra ao invés de random
                 exclude_gt_and_pred: bool = True,
                 exclude_dominant: bool = True,
                 
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
        
        self.margin_threshold = margin_threshold
        
        self.enable_propagation = enable_propagation
        self.propagation_k = propagation_k
        self.propagation_similarity_threshold = propagation_similarity_threshold
        self.propagation_confidence_threshold = propagation_confidence_threshold
        
        self.use_random_label = use_random_label
        self.exclude_gt_and_pred = exclude_gt_and_pred
        self.exclude_dominant = exclude_dominant
        
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
    
    def _find_similar_neighbors(self, embeddings, query_indices, k, similarity_threshold):
        """
        Encontra vizinhos similares para as amostras em query_indices.
        Retorna set de índices de vizinhos que passam o threshold de similaridade.
        """
        if len(query_indices) == 0:
            return set()
        
        N, D = embeddings.shape
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        embeddings_norm = embeddings_norm.astype(np.float32)
        
        # Buscar k+1 vizinhos (inclui a própria amostra)
        k_search = min(k + 1, N)
        
        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(D)  # Inner product = cosine similarity para vetores normalizados
            index.add(embeddings_norm)
            
            query_embeddings = embeddings_norm[query_indices]
            similarities, neighbor_indices = index.search(query_embeddings, k_search)
        else:
            nn = NearestNeighbors(n_neighbors=k_search, metric='cosine')
            nn.fit(embeddings_norm)
            distances, neighbor_indices = nn.kneighbors(embeddings_norm[query_indices])
            similarities = 1 - distances  # Converter distância coseno para similaridade
        
        propagated = set()
        query_set = set(query_indices)
        
        for i, query_idx in enumerate(query_indices):
            for j in range(k_search):
                neighbor_idx = neighbor_indices[i, j]
                sim = similarities[i, j]
                
                # Pular a própria amostra e amostras já no query
                if neighbor_idx == query_idx or neighbor_idx in query_set:
                    continue
                
                # Verificar threshold de similaridade
                if sim >= similarity_threshold:
                    propagated.add(neighbor_idx)
        
        return propagated
    
    def _get_random_label(self, gt_label, pred_label, dominant_label):
        excluded = set()
        
        if self.exclude_gt_and_pred:
            excluded.add(gt_label)
            excluded.add(pred_label)
        
        if self.exclude_dominant:
            excluded.add(dominant_label)
        
        available = [c for c in range(self.num_classes) if c not in excluded]
        
        if len(available) == 0:
            available = [c for c in range(self.num_classes) if c != gt_label]
        
        return random.choice(available)
    
    def before_train_epoch(self, runner):
        epoch = runner.epoch + 1
        
        if epoch <= self.warmup_epochs:
            if self.debug:
                print(f"[NoiseProp] Época {epoch}: Warmup, pulando.")
            return
        
        if self.debug:
            print(f"\n[NoiseProp] ========== Época {epoch} ==========")
        
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
                    
                    top2_scores, _ = best_scores.topk(2)
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
                        'embedding': embedding,
                        'scores': best_scores.cpu(),
                        'margin': margin,
                        'action': None,
                        'is_propagated': False,
                    }
                    all_box_data.append(box_data)
                    boxes_by_image[img_path].append(box_data)
                    scores_by_class[gt_label].append(score_gt)
        
        if len(all_box_data) == 0:
            return
        
        if self.debug:
            print(f"[NoiseProp] Total boxes: {len(all_box_data)}")
        
        gmm_dict = self._fit_gmm_per_class(scores_by_class)
        
        for box in all_box_data:
            box['p_noise'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict)
        
        embeddings = np.array([box['embedding'] for box in all_box_data])
        cluster_ids = self._cluster_embeddings(embeddings, self.n_clusters)
        for i, box in enumerate(all_box_data):
            box['cluster_id'] = cluster_ids[i]
            box['global_idx'] = i  # Para referência
        
        criteria = self._get_current_criteria(epoch)
        
        if self.debug:
            print(f"[NoiseProp] Fase: {criteria['phase']}")
        
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
        
        # Primeira passada: identificar suspeitos originais
        original_suspects = []  # Lista de (box, dominant_label)
        high_confidence_suspects = []  # Para propagação
        
        for cluster_id, cluster_boxes in clusters.items():
            if len(cluster_boxes) < 2:
                continue
            
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
                
                if box['gt_label'] == dominant_label:
                    continue
                
                box_emb_norm = box['embedding'] / (np.linalg.norm(box['embedding']) + 1e-8)
                similarity = np.dot(box_emb_norm, anchor_mean_norm)
                
                if similarity > c_similarity:
                    original_suspects.append((box, dominant_label))
                    
                    # Se alta confiança de ruído, usar para propagação
                    if box['p_noise'] >= self.propagation_confidence_threshold:
                        high_confidence_suspects.append(box['global_idx'])
        
        if self.debug:
            print(f"[NoiseProp] Total âncoras: {total_anchors}")
            print(f"[NoiseProp] Suspeitos originais: {len(original_suspects)}")
            print(f"[NoiseProp] Alta confiança (para propagação): {len(high_confidence_suspects)}")
        
        # Propagação de ruído
        propagated_indices = set()
        if self.enable_propagation and len(high_confidence_suspects) > 0:
            propagated_indices = self._find_similar_neighbors(
                embeddings,
                high_confidence_suspects,
                self.propagation_k,
                self.propagation_similarity_threshold
            )
            
            # Marcar boxes propagados
            for idx in propagated_indices:
                all_box_data[idx]['is_propagated'] = True
            
            if self.debug:
                print(f"[NoiseProp] Propagados: {len(propagated_indices)}")
        
        # Criar mapeamento de suspeitos para dominant_label
        suspect_to_dominant = {}
        for box, dominant_label in original_suspects:
            suspect_to_dominant[box['global_idx']] = dominant_label
        
        # Para propagados, usar a mesma lógica de cluster
        # (simplificação: usar o dominant_label do cluster mais próximo)
        for idx in propagated_indices:
            if idx not in suspect_to_dominant:
                box = all_box_data[idx]
                cluster_id = box['cluster_id']
                # Pegar dominant_label do cluster
                cluster_boxes = clusters.get(cluster_id, [])
                if len(cluster_boxes) > 0:
                    labels = [b['gt_label'] for b in cluster_boxes if b.get('p_noise', 1) < c_anchor_gmm]
                    if labels:
                        dominant_label = Counter(labels).most_common(1)[0][0]
                        suspect_to_dominant[idx] = dominant_label
        
        # Aplicar ações
        stats = {
            'original_relabel': 0,
            'original_random': 0,
            'original_filter': 0,
            'propagated_relabel': 0,
            'propagated_random': 0,
            'propagated_filter': 0,
        }
        
        all_suspect_indices = set(box['global_idx'] for box, _ in original_suspects) | propagated_indices
        
        for idx in all_suspect_indices:
            box = all_box_data[idx]
            dominant_label = suspect_to_dominant.get(idx)
            
            if dominant_label is None:
                continue
            
            if box['gt_label'] == dominant_label:
                continue
            
            is_propagated = box['is_propagated']
            prefix = 'propagated' if is_propagated else 'original'
            
            if box['margin'] > self.margin_threshold:
                # Margem grande → RELABELA para dominante
                self._apply_relabel(datasets, box['sub_idx'], box['data_idx'], box['gt_idx'], dominant_label)
                box['action'] = 'relabel'
                stats[f'{prefix}_relabel'] += 1
            else:
                # Margem pequena → possível assimétrico
                if self.use_random_label:
                    # Random label
                    random_label = self._get_random_label(box['gt_label'], box['pred_label'], dominant_label)
                    self._apply_relabel(datasets, box['sub_idx'], box['data_idx'], box['gt_idx'], random_label)
                    box['action'] = 'random'
                    stats[f'{prefix}_random'] += 1
                else:
                    # Filtrar
                    self._apply_ignore_flag(datasets, box['sub_idx'], box['data_idx'], box['gt_idx'])
                    box['action'] = 'filter'
                    stats[f'{prefix}_filter'] += 1
        
        if self.debug:
            print(f"[NoiseProp] Ações originais: relabel={stats['original_relabel']}, random={stats['original_random']}, filter={stats['original_filter']}")
            print(f"[NoiseProp] Ações propagadas: relabel={stats['propagated_relabel']}, random={stats['propagated_random']}, filter={stats['propagated_filter']}")
        
        # Spatial Refinement
        spatial_relabel_count = 0
        if self.enable_spatial_refinement:
            for img_path, img_boxes in boxes_by_image.items():
                if len(img_boxes) < 2:
                    continue
                active_boxes = [b for b in img_boxes if b['action'] not in ['random', 'filter']]
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
                print(f"[NoiseProp] Relabelados por spatial: {spatial_relabel_count}")
        
        if self.debug:
            print(f"[NoiseProp] ==========================================\n")
    
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
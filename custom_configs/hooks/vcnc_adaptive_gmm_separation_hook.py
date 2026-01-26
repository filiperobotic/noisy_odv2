"""
VCNC com Thresholds Adaptativos - Baseado na Separação do GMM

A ideia é simples:
- Classes onde o GMM separa bem os componentes clean/noisy → podemos ser mais agressivos
- Classes onde o GMM tem componentes sobrepostos → devemos ser mais conservadores

A separação é medida pela distância entre as médias dos dois componentes do GMM.
"""

from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import torch
import torch.nn.functional as F
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
class VCNCAdaptiveGMMSeparationHook(Hook):
    """
    VCNC com Thresholds Adaptativos baseado na Separação do GMM
    
    Lógica:
    - Para cada classe, o GMM tem 2 componentes (clean e noisy)
    - A separação = |mean_clean - mean_noisy|
    - Alta separação → classes bem definidas → podemos ser mais agressivos
    - Baixa separação → classes ambíguas → devemos ser mais conservadores
    
    O threshold é calculado como:
    - anchor_threshold = base + separation_norm * range
    - suspect_threshold = base - separation_norm * range
    """
    
    def __init__(self,
                 # Configuração geral
                 warmup_epochs: int = 1,
                 num_classes: int = 20,
                 
                 # === Clustering ===
                 enable_clustering_relabel: bool = True,
                 use_softmax_as_embedding: bool = True,
                 n_clusters: int = 50,
                 
                 # === Thresholds Adaptativos baseados em GMM Separation ===
                 # Range para anchor_threshold (p_noise < threshold para ser âncora)
                 # Quanto MAIOR o threshold, MAIS âncoras (mais permissivo)
                 anchor_threshold_min: float = 0.15,  # Para classes com baixa separação
                 anchor_threshold_max: float = 0.45,  # Para classes com alta separação
                 
                 # Range para suspect_threshold (p_noise > threshold para ser suspeito)
                 # Quanto MENOR o threshold, MAIS suspeitos
                 suspect_threshold_min: float = 0.45,  # Para classes com alta separação
                 suspect_threshold_max: float = 0.75,  # Para classes com baixa separação
                 
                 # Critérios fixos
                 anchor_pred_agreement: float = 0.6,
                 anchor_confidence: float = 0.7,
                 similarity_threshold: float = 0.4,
                 cluster_consensus: float = 0.6,
                 
                 # Critérios progressivos
                 progressive_epochs: int = 4,
                 early_anchor_pred_agreement: float = 0.85,
                 early_anchor_confidence: float = 0.9,
                 early_similarity_threshold: float = 0.7,
                 early_cluster_consensus: float = 0.85,
                 
                 # === Spatial Refinement ===
                 enable_spatial_refinement: bool = True,
                 spatial_difficulty_threshold: float = 0.5,
                 
                 # === Filtragem (opcional) ===
                 enable_gmm_filter: bool = False,
                 filter_gmm_threshold: float = 0.9,
                 
                 # Configuração do assigner
                 iou_assigner: float = 0.5,
                 
                 # Reload dataset
                 reload_dataset: bool = True,
                 
                 # Debug
                 debug: bool = True):
        
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        
        self.enable_clustering_relabel = enable_clustering_relabel
        self.use_softmax_as_embedding = use_softmax_as_embedding
        self.n_clusters = n_clusters
        
        # Ranges adaptativos
        self.anchor_threshold_min = anchor_threshold_min
        self.anchor_threshold_max = anchor_threshold_max
        self.suspect_threshold_min = suspect_threshold_min
        self.suspect_threshold_max = suspect_threshold_max
        
        # Critérios fixos
        self.anchor_pred_agreement = anchor_pred_agreement
        self.anchor_confidence = anchor_confidence
        self.similarity_threshold = similarity_threshold
        self.cluster_consensus = cluster_consensus
        
        # Progressivos
        self.progressive_epochs = progressive_epochs
        self.early_anchor_pred_agreement = early_anchor_pred_agreement
        self.early_anchor_confidence = early_anchor_confidence
        self.early_similarity_threshold = early_similarity_threshold
        self.early_cluster_consensus = early_cluster_consensus
        
        # Spatial
        self.enable_spatial_refinement = enable_spatial_refinement
        self.spatial_difficulty_threshold = spatial_difficulty_threshold
        
        # Filtragem
        self.enable_gmm_filter = enable_gmm_filter
        self.filter_gmm_threshold = filter_gmm_threshold
        
        # Assigner
        self.iou_assigner = iou_assigner
        
        # Reload
        self.reload_dataset = reload_dataset
        
        # Debug
        self.debug = debug
    
    def _fit_gmm_per_class(self, scores_by_class):
        """Ajusta GMM para cada classe e retorna GMM + separação."""
        gmm_dict = {}
        separations = {}
        
        for cls_id, scores in scores_by_class.items():
            if len(scores) < 10:
                continue
            
            scores_np = np.array(scores).reshape(-1, 1)
            
            try:
                gmm = GaussianMixture(n_components=2, max_iter=100, random_state=42)
                gmm.fit(scores_np)
                gmm_dict[cls_id] = gmm
                
                # Calcular separação entre componentes
                means = gmm.means_.flatten()
                separation = abs(means[0] - means[1])
                separations[cls_id] = separation
                
            except Exception as e:
                if self.debug:
                    print(f"[VCNC-GMM-Sep] Erro GMM classe {cls_id}: {e}")
        
        return gmm_dict, separations
    
    def _compute_adaptive_thresholds(self, separations):
        """
        Calcula thresholds adaptativos baseado na separação do GMM.
        
        Normaliza as separações para [0, 1] e usa para interpolar os thresholds.
        """
        if len(separations) == 0:
            return {}
        
        # Normalizar separações
        sep_values = list(separations.values())
        sep_min, sep_max = min(sep_values), max(sep_values)
        
        thresholds = {}
        
        for cls_id, sep in separations.items():
            # Normalizar para [0, 1]
            if sep_max > sep_min:
                sep_norm = (sep - sep_min) / (sep_max - sep_min)
            else:
                sep_norm = 0.5
            
            # Interpolar thresholds
            # Alta separação (sep_norm ≈ 1) → mais agressivo
            # - anchor_threshold MAIOR (mais âncoras)
            # - suspect_threshold MENOR (mais suspeitos)
            
            anchor_thr = self.anchor_threshold_min + sep_norm * (self.anchor_threshold_max - self.anchor_threshold_min)
            suspect_thr = self.suspect_threshold_max - sep_norm * (self.suspect_threshold_max - self.suspect_threshold_min)
            
            thresholds[cls_id] = {
                'anchor_gmm_threshold': anchor_thr,
                'suspect_gmm_threshold': suspect_thr,
                'separation': sep,
                'separation_norm': sep_norm
            }
        
        return thresholds
    
    def _get_p_noise(self, score_gt, gt_label, gmm_dict):
        """Calcula probabilidade de ser ruído."""
        if gt_label not in gmm_dict:
            return 0.5
        
        gmm = gmm_dict[gt_label]
        score_np = np.array([[score_gt]])
        
        try:
            probs = gmm.predict_proba(score_np)
            noisy_component = np.argmin(gmm.means_)
            return float(probs[0, noisy_component])
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
    
    def _get_current_criteria(self, epoch):
        """Retorna critérios não-adaptativos baseado na época."""
        if epoch <= self.progressive_epochs:
            return {
                'anchor_pred_agreement': self.early_anchor_pred_agreement,
                'anchor_confidence': self.early_anchor_confidence,
                'similarity_threshold': self.early_similarity_threshold,
                'cluster_consensus': self.early_cluster_consensus,
                'phase': 'CONSERVADOR'
            }
        else:
            return {
                'anchor_pred_agreement': self.anchor_pred_agreement,
                'anchor_confidence': self.anchor_confidence,
                'similarity_threshold': self.similarity_threshold,
                'cluster_consensus': self.cluster_consensus,
                'phase': 'AGRESSIVO'
            }
    
    def before_train_epoch(self, runner):
        """Executa o pipeline completo antes de cada época."""
        epoch = runner.epoch + 1
        
        if epoch <= self.warmup_epochs:
            if self.debug:
                print(f"[VCNC-GMM-Sep] Época {epoch}: Warmup, pulando.")
            return
        
        if self.debug:
            print(f"\n[VCNC-GMM-Sep] ========== Época {epoch} ==========")
        
        # Reload dataset
        if self.reload_dataset:
            self._reload_datasets(runner)
        
        # Obter dataset
        dataloader = runner.train_loop.dataloader
        dataset = self._get_base_dataset(dataloader.dataset)
        
        if not hasattr(dataset, 'datasets'):
            print("[VCNC-GMM-Sep] ERRO: Esperado ConcatDataset")
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
            print("[VCNC-GMM-Sep] Coletando dados...")
        
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
                    boxes_by_image[img_path].append(box_data)
                    scores_by_class[gt_label].append(score_gt)
        
        if len(all_box_data) == 0:
            print("[VCNC-GMM-Sep] Nenhum box coletado!")
            return
        
        if self.debug:
            print(f"[VCNC-GMM-Sep] Coletados {len(all_box_data)} boxes em {len(boxes_by_image)} imagens")
        
        # ============================================================
        # ETAPA 2: CLUSTERING COM THRESHOLDS ADAPTATIVOS
        # ============================================================
        clustering_relabel_count = 0
        
        if self.enable_clustering_relabel:
            if self.debug:
                print(f"\n[VCNC-GMM-Sep] ETAPA 2: Clustering com Thresholds Adaptativos (GMM Separation)")
            
            # Ajustar GMM por classe e obter separações
            gmm_dict, separations = self._fit_gmm_per_class(scores_by_class)
            
            # Calcular thresholds adaptativos
            class_thresholds = self._compute_adaptive_thresholds(separations)
            
            if self.debug:
                print(f"\n[VCNC-GMM-Sep] Thresholds adaptativos por classe (baseado em GMM Separation):")
                print(f"{'Classe':<8} | {'Separação':<12} | {'Sep. Norm':<12} | {'Anchor Thr':<12} | {'Suspect Thr':<12}")
                print("-" * 70)
                
                for cls_id in sorted(class_thresholds.keys()):
                    t = class_thresholds[cls_id]
                    print(f"{cls_id:<8} | {t['separation']:<12.4f} | {t['separation_norm']:<12.4f} | "
                          f"{t['anchor_gmm_threshold']:<12.4f} | {t['suspect_gmm_threshold']:<12.4f}")
            
            # Calcular p_noise para cada box
            for box in all_box_data:
                box['p_noise'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict)
            
            # Clustering
            embeddings = np.array([box['embedding'] for box in all_box_data])
            cluster_ids = self._cluster_embeddings(embeddings, self.n_clusters)
            
            for i, box in enumerate(all_box_data):
                box['cluster_id'] = cluster_ids[i]
            
            # Critérios não-adaptativos
            criteria = self._get_current_criteria(epoch)
            
            if self.debug:
                print(f"\n[VCNC-GMM-Sep] Fase: {criteria['phase']}, Clusters: {len(set(cluster_ids))}")
            
            # Processar clusters
            clusters = defaultdict(list)
            for box in all_box_data:
                clusters[box['cluster_id']].append(box)
            
            total_anchors = 0
            total_suspects = 0
            
            for cluster_id, cluster_boxes in clusters.items():
                if len(cluster_boxes) < 2:
                    continue
                
                # Identificar âncoras (threshold adaptativo por classe)
                anchors = []
                for box in cluster_boxes:
                    cls_id = box['gt_label']
                    
                    # Obter threshold da classe
                    if cls_id in class_thresholds:
                        anchor_thr = class_thresholds[cls_id]['anchor_gmm_threshold']
                    else:
                        anchor_thr = self.anchor_threshold_min  # Conservador
                    
                    low_noise = box['p_noise'] < anchor_thr
                    model_agrees = box['score_gt'] > criteria['anchor_pred_agreement']
                    high_confidence = box['pred_score'] > criteria['anchor_confidence']
                    
                    if low_noise and model_agrees and high_confidence:
                        anchors.append(box)
                
                total_anchors += len(anchors)
                
                if len(anchors) == 0:
                    continue
                
                anchor_labels = [a['gt_label'] for a in anchors]
                label_counts = Counter(anchor_labels)
                dominant_label, count = label_counts.most_common(1)[0]
                consensus_ratio = count / len(anchors)
                
                if consensus_ratio < criteria['cluster_consensus']:
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
                    
                    cls_id = box['gt_label']
                    
                    # Obter threshold de suspeito da classe
                    if cls_id in class_thresholds:
                        suspect_thr = class_thresholds[cls_id]['suspect_gmm_threshold']
                    else:
                        suspect_thr = self.suspect_threshold_max  # Conservador
                    
                    if box['p_noise'] < suspect_thr:
                        continue
                    
                    total_suspects += 1
                    
                    if box['gt_label'] == dominant_label:
                        continue
                    
                    box_emb_norm = box['embedding'] / (np.linalg.norm(box['embedding']) + 1e-8)
                    similarity = np.dot(box_emb_norm, anchor_mean_norm)
                    
                    if similarity > criteria['similarity_threshold']:
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
                print(f"\n[VCNC-GMM-Sep] Total âncoras: {total_anchors} ({total_anchors/len(all_box_data)*100:.2f}%)")
                print(f"[VCNC-GMM-Sep] Total suspeitos: {total_suspects} ({total_suspects/len(all_box_data)*100:.2f}%)")
                print(f"[VCNC-GMM-Sep] Relabelados por clustering: {clustering_relabel_count} "
                      f"({clustering_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 3: SPATIAL REFINEMENT
        # ============================================================
        spatial_relabel_count = 0
        
        if self.enable_spatial_refinement:
            if self.debug:
                print(f"\n[VCNC-GMM-Sep] ETAPA 3: Spatial Refinement")
            
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
                print(f"[VCNC-GMM-Sep] Relabelados por spatial: {spatial_relabel_count}")
        
        # ============================================================
        # ESTATÍSTICAS FINAIS
        # ============================================================
        if self.debug:
            total_relabels = clustering_relabel_count + spatial_relabel_count
            print(f"\n[VCNC-GMM-Sep] ===== Resumo Época {epoch} =====")
            print(f"[VCNC-GMM-Sep] Total de boxes: {len(all_box_data)}")
            print(f"[VCNC-GMM-Sep] Total relabels: {total_relabels} ({total_relabels/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-GMM-Sep] ==========================================\n")
    
    def _reload_datasets(self, runner):
        try:
            ds = runner.train_loop.dataloader.dataset.dataset
            for i, subds in enumerate(ds.datasets):
                if hasattr(subds, '_fully_initialized'):
                    subds._fully_initialized = False
                if hasattr(subds, 'full_init'):
                    subds.full_init()
        except Exception as e:
            if self.debug:
                print(f"[VCNC-GMM-Sep] Erro ao recarregar: {e}")
    
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
                print(f"[VCNC-GMM-Sep] Erro relabel: {e}")
    
    def _apply_ignore_flag(self, datasets, sub_idx, data_idx, gt_idx):
        try:
            instance = datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]
            instance['ignore_flag'] = 1
        except Exception as e:
            if self.debug:
                print(f"[VCNC-GMM-Sep] Erro ignore_flag: {e}")

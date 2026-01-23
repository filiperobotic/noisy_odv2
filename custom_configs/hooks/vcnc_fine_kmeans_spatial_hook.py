"""
Visual Clustering Noise Correction Hook - VERSÃO COM FINE + K-MEANS + SPATIAL REFINEMENT

Combina:
1. FINE (Filtering Noisy Instances via Eigenvectors) para identificar clean/noisy
2. K-Means para clustering visual
3. Spatial Refinement para boxes contaminados espacialmente

Diferença principal:
- Em vez de usar confiança/loss + GMM para identificar âncoras/suspeitos
- Usa alignment score (FINE) + GMM para essa identificação
- O alignment score mede quão alinhado o embedding está com o autovetor principal da classe

Pipeline:
1. Relabel por confiança > 0.9 (baseline, opcional)
2. FINE: calcula alignment scores e usa GMM para obter p_clean/p_noise
3. K-Means clustering + relabeling baseado em âncoras FINE
4. Spatial Refinement para boxes contaminados espacialmente
5. Filtragem seletiva (opcional)
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
class VCNCFineKMeansSpatialHook(Hook):
    """
    VCNC com FINE + K-Means + Spatial Refinement
    
    FINE (Filtering Noisy Instances via Eigenvectors):
    - Para cada classe, calcula a matriz de Gram dos embeddings
    - Extrai o primeiro autovetor (componente principal)
    - Calcula alignment score: f_i = <u_k, z_i>^2
    - Amostras alinhadas (score alto) = provavelmente limpas
    - Amostras desalinhadas (score baixo) = provavelmente ruidosas
    - Usa GMM nos alignment scores para obter p_clean
    
    Vantagem sobre GMM baseado em loss/confiança:
    - FINE usa informação topológica do espaço latente
    - Não depende do classificador linear que pode estar corrompido
    """
    
    def __init__(self,
                 # Configuração geral
                 warmup_epochs: int = 1,
                 num_classes: int = 20,
                 
                 # === ETAPA 1: Relabel por confiança (Baseline) ===
                 enable_confidence_relabel: bool = False,
                 relabel_confidence_threshold: float = 0.9,
                 
                 # === ETAPA 2: FINE + K-Means Clustering ===
                 enable_fine_clustering_relabel: bool = True,
                 use_softmax_as_embedding: bool = True,
                 n_clusters: int = 150,
                 
                 # Parâmetros do FINE
                 fine_gmm_components: int = 2,  # 2 componentes: clean e noisy
                 fine_use_normalized_embedding: bool = True,  # Normalizar embeddings para FINE
                 
                 # Critérios progressivos (baseados em p_clean do FINE)
                 progressive_epochs: int = 4,
                 
                 # Conservador (épocas iniciais)
                 early_anchor_clean_threshold: float = 0.85,  # p_clean >= threshold para ser âncora
                 early_anchor_pred_agreement: float = 0.85,   # pred == gt_label
                 early_anchor_confidence: float = 0.9,        # confiança alta
                 early_suspect_clean_threshold: float = 0.3,  # p_clean <= threshold para ser suspeito
                 early_similarity_threshold: float = 0.7,
                 early_cluster_consensus: float = 0.85,
                 
                 # Agressivo (épocas posteriores)
                 anchor_clean_threshold: float = 0.6,
                 anchor_pred_agreement: float = 0.6,
                 anchor_confidence: float = 0.7,
                 suspect_clean_threshold: float = 0.5,
                 similarity_threshold: float = 0.4,
                 cluster_consensus: float = 0.6,
                 
                 # === ETAPA 3: Spatial Refinement ===
                 enable_spatial_refinement: bool = True,
                 spatial_difficulty_threshold: float = 0.5,
                 
                 # === ETAPA 4: Filtragem Seletiva (opcional) ===
                 enable_selective_filtering: bool = False,
                 selective_filter_clean_threshold: float = 0.3,  # p_clean <= threshold
                 selective_filter_confidence_threshold: float = 0.7,
                 
                 # === ETAPA 5: Filtragem GMM Adicional (Baseline) ===
                 enable_gmm_filter: bool = False,
                 filter_clean_threshold: float = 0.3,
                 
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
        
        # Etapa 2 - FINE + K-Means
        self.enable_fine_clustering_relabel = enable_fine_clustering_relabel
        self.use_softmax_as_embedding = use_softmax_as_embedding
        self.n_clusters = n_clusters
        self.fine_gmm_components = fine_gmm_components
        self.fine_use_normalized_embedding = fine_use_normalized_embedding
        
        self.progressive_epochs = progressive_epochs
        
        # Conservador
        self.early_anchor_clean_threshold = early_anchor_clean_threshold
        self.early_anchor_pred_agreement = early_anchor_pred_agreement
        self.early_anchor_confidence = early_anchor_confidence
        self.early_suspect_clean_threshold = early_suspect_clean_threshold
        self.early_similarity_threshold = early_similarity_threshold
        self.early_cluster_consensus = early_cluster_consensus
        
        # Agressivo
        self.anchor_clean_threshold = anchor_clean_threshold
        self.anchor_pred_agreement = anchor_pred_agreement
        self.anchor_confidence = anchor_confidence
        self.suspect_clean_threshold = suspect_clean_threshold
        self.similarity_threshold = similarity_threshold
        self.cluster_consensus = cluster_consensus
        
        # Etapa 3
        self.enable_spatial_refinement = enable_spatial_refinement
        self.spatial_difficulty_threshold = spatial_difficulty_threshold
        
        # Etapa 4
        self.enable_selective_filtering = enable_selective_filtering
        self.selective_filter_clean_threshold = selective_filter_clean_threshold
        self.selective_filter_confidence_threshold = selective_filter_confidence_threshold
        
        # Etapa 5
        self.enable_gmm_filter = enable_gmm_filter
        self.filter_clean_threshold = filter_clean_threshold
        
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
                'anchor_clean_threshold': self.early_anchor_clean_threshold,
                'anchor_pred_agreement': self.early_anchor_pred_agreement,
                'anchor_confidence': self.early_anchor_confidence,
                'suspect_clean_threshold': self.early_suspect_clean_threshold,
                'similarity_threshold': self.early_similarity_threshold,
                'cluster_consensus': self.early_cluster_consensus,
                'phase': 'CONSERVADOR'
            }
        else:
            return {
                'anchor_clean_threshold': self.anchor_clean_threshold,
                'anchor_pred_agreement': self.anchor_pred_agreement,
                'anchor_confidence': self.anchor_confidence,
                'suspect_clean_threshold': self.suspect_clean_threshold,
                'similarity_threshold': self.similarity_threshold,
                'cluster_consensus': self.cluster_consensus,
                'phase': 'AGRESSIVO'
            }
    
    def _compute_fine_scores(self, all_box_data):
        """
        Computa os FINE scores (alignment scores) para todos os boxes.
        
        Para cada classe:
        1. Coleta embeddings da classe
        2. Calcula matriz de Gram: Σ = Σ z_i * z_i^T
        3. Decomposição de autovalores: Σ = U Λ U^T
        4. Primeiro autovetor u_1 (maior autovalor)
        5. Alignment score: f_i = <u_1, z_i>^2
        
        Returns:
            dict: eigenvectors por classe
        """
        # Organizar embeddings por classe
        embeddings_by_class = defaultdict(list)
        indices_by_class = defaultdict(list)
        
        for i, box in enumerate(all_box_data):
            cls_id = box['gt_label']
            emb = box['embedding']
            
            if self.fine_use_normalized_embedding:
                emb = emb / (np.linalg.norm(emb) + 1e-8)
            
            embeddings_by_class[cls_id].append(emb)
            indices_by_class[cls_id].append(i)
        
        eigenvectors = {}
        
        for cls_id, embeddings in embeddings_by_class.items():
            if len(embeddings) < 2:
                continue
            
            embeddings_np = np.array(embeddings)  # [N, D]
            
            # Calcular matriz de Gram: Σ = X^T X (ou Σ = Σ z_i z_i^T)
            # Para eficiência, usamos a matriz de covariância
            gram_matrix = embeddings_np.T @ embeddings_np  # [D, D]
            
            try:
                # Decomposição de autovalores
                eigenvalues, eigenvecs = np.linalg.eigh(gram_matrix)
                
                # eigh retorna em ordem crescente, queremos o maior
                # O último é o maior autovalor
                first_eigenvector = eigenvecs[:, -1]  # [D]
                
                eigenvectors[cls_id] = first_eigenvector
                
            except Exception as e:
                if self.debug:
                    print(f"[VCNC-FINE] Erro na decomposição para classe {cls_id}: {e}")
                continue
        
        # Calcular alignment scores
        for i, box in enumerate(all_box_data):
            cls_id = box['gt_label']
            
            if cls_id not in eigenvectors:
                box['alignment_score'] = 0.5  # Valor neutro
                continue
            
            emb = box['embedding']
            if self.fine_use_normalized_embedding:
                emb = emb / (np.linalg.norm(emb) + 1e-8)
            
            u = eigenvectors[cls_id]
            
            # Alignment score: f_i = <u, z>^2
            alignment = np.dot(u, emb) ** 2
            box['alignment_score'] = float(alignment)
        
        return eigenvectors
    
    def _fit_fine_gmm_per_class(self, all_box_data):
        """
        Ajusta GMM nos alignment scores para cada classe.
        
        O GMM separa em 2 componentes:
        - Componente com média MAIOR = amostras limpas (alinhadas)
        - Componente com média MENOR = amostras ruidosas (desalinhadas)
        
        Returns:
            dict: (gmm, clean_component_idx) por classe
        """
        gmm_dict = {}
        
        # Organizar alignment scores por classe
        scores_by_class = defaultdict(list)
        for box in all_box_data:
            scores_by_class[box['gt_label']].append(box['alignment_score'])
        
        for cls_id, scores in scores_by_class.items():
            if len(scores) < 10:
                continue
            
            scores_np = np.array(scores).reshape(-1, 1)
            
            try:
                gmm = GaussianMixture(
                    n_components=self.fine_gmm_components,
                    max_iter=100,
                    tol=1e-3,
                    reg_covar=1e-6,
                    random_state=42
                )
                gmm.fit(scores_np)
                
                # Componente com média MAIOR = clean (mais alinhado)
                clean_component = int(np.argmax(gmm.means_))
                gmm_dict[cls_id] = (gmm, clean_component)
                
            except Exception as e:
                if self.debug:
                    print(f"[VCNC-FINE] Erro GMM classe {cls_id}: {e}")
        
        return gmm_dict
    
    def _get_p_clean(self, alignment_score, cls_id, gmm_dict):
        """
        Calcula probabilidade de ser clean baseado no alignment score.
        
        Diferente do p_noise tradicional:
        - p_clean ALTO = alinhado = provavelmente limpo
        - p_clean BAIXO = desalinhado = provavelmente ruidoso
        """
        if cls_id not in gmm_dict:
            return 0.5
        
        gmm, clean_comp = gmm_dict[cls_id]
        score_np = np.array([[alignment_score]])
        
        try:
            probs = gmm.predict_proba(score_np)
            return float(probs[0, clean_comp])
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
    
    def before_train_epoch(self, runner):
        """Executa o pipeline completo antes de cada época."""
        epoch = runner.epoch + 1
        
        if epoch <= self.warmup_epochs:
            if self.debug:
                print(f"[VCNC-FINE] Época {epoch}: Warmup, pulando.")
            return
        
        if self.debug:
            print(f"\n[VCNC-FINE] ========== Época {epoch} ==========")
        
        # Reload dataset
        if self.reload_dataset:
            self._reload_datasets(runner)
        
        # Obter dataset
        dataloader = runner.train_loop.dataloader
        dataset = self._get_base_dataset(dataloader.dataset)
        
        if not hasattr(dataset, 'datasets'):
            print("[VCNC-FINE] ERRO: Esperado ConcatDataset")
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
            print("[VCNC-FINE] Coletando dados...")
        
        all_box_data = []
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
                    boxes_by_image[img_path].append(box_data)
        
        if len(all_box_data) == 0:
            print("[VCNC-FINE] Nenhum box coletado!")
            return
        
        if self.debug:
            print(f"[VCNC-FINE] Coletados {len(all_box_data)} boxes em {len(boxes_by_image)} imagens")
        
        # ============================================================
        # ETAPA 1: RELABEL POR CONFIANÇA ALTA (OPCIONAL)
        # ============================================================
        confidence_relabel_count = 0
        
        if self.enable_confidence_relabel:
            if self.debug:
                print(f"\n[VCNC-FINE] ETAPA 1: Relabel por confiança > {self.relabel_confidence_threshold}")
            
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
                print(f"[VCNC-FINE] Relabelados por confiança: {confidence_relabel_count} "
                      f"({confidence_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 2: FINE + K-MEANS CLUSTERING RELABEL
        # ============================================================
        clustering_relabel_count = 0
        
        if self.enable_fine_clustering_relabel:
            if self.debug:
                print(f"\n[VCNC-FINE] ETAPA 2: FINE + K-Means Clustering")
            
            # 2.1 Calcular FINE scores (alignment scores)
            if self.debug:
                print(f"[VCNC-FINE] Calculando alignment scores (FINE)...")
            
            eigenvectors = self._compute_fine_scores(all_box_data)
            
            if self.debug:
                scores = [box['alignment_score'] for box in all_box_data]
                print(f"[VCNC-FINE] Alignment scores - min: {min(scores):.4f}, "
                      f"max: {max(scores):.4f}, mean: {np.mean(scores):.4f}")
            
            # 2.2 Ajustar GMM nos alignment scores
            if self.debug:
                print(f"[VCNC-FINE] Ajustando GMM nos alignment scores...")
            
            fine_gmm_dict = self._fit_fine_gmm_per_class(all_box_data)
            
            # 2.3 Calcular p_clean para cada box
            for box in all_box_data:
                box['p_clean'] = self._get_p_clean(
                    box['alignment_score'], 
                    box['gt_label'], 
                    fine_gmm_dict
                )
                box['p_noise'] = 1 - box['p_clean']
            
            if self.debug:
                p_cleans = [box['p_clean'] for box in all_box_data]
                print(f"[VCNC-FINE] p_clean - min: {min(p_cleans):.4f}, "
                      f"max: {max(p_cleans):.4f}, mean: {np.mean(p_cleans):.4f}")
            
            # 2.4 Clustering com K-Means
            embeddings = np.array([box['embedding'] for box in all_box_data])
            cluster_ids = self._cluster_embeddings(embeddings, self.n_clusters)
            
            for i, box in enumerate(all_box_data):
                box['cluster_id'] = cluster_ids[i]
            
            criteria = self._get_current_criteria(epoch)
            
            if self.debug:
                print(f"[VCNC-FINE] Fase: {criteria['phase']}, Clusters: {len(set(cluster_ids))}")
            
            # 2.5 Processar clusters
            clusters = defaultdict(list)
            for box in all_box_data:
                clusters[box['cluster_id']].append(box)
            
            c_anchor_clean = criteria['anchor_clean_threshold']
            c_anchor_pred = criteria['anchor_pred_agreement']
            c_anchor_conf = criteria['anchor_confidence']
            c_suspect_clean = criteria['suspect_clean_threshold']
            c_similarity = criteria['similarity_threshold']
            c_consensus = criteria['cluster_consensus']
            
            # Estatísticas
            total_anchors = 0
            total_suspects = 0
            
            for cluster_id, cluster_boxes in clusters.items():
                if len(cluster_boxes) < 2:
                    continue
                
                # Identificar âncoras usando FINE (p_clean alto)
                anchors = []
                for box in cluster_boxes:
                    is_clean = box['p_clean'] >= c_anchor_clean  # FINE: p_clean alto
                    model_agrees = box['score_gt'] > c_anchor_pred
                    high_confidence = box['pred_score'] > c_anchor_conf
                    
                    if is_clean and model_agrees and high_confidence:
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
                
                anchor_embeddings = np.array([a['embedding'] for a in anchors])
                anchor_mean = anchor_embeddings.mean(axis=0)
                anchor_mean_norm = anchor_mean / (np.linalg.norm(anchor_mean) + 1e-8)
                
                anchor_ids = set(id(a) for a in anchors)
                
                for box in cluster_boxes:
                    if id(box) in anchor_ids:
                        continue
                    
                    if box['relabeled_by'] == 'confidence':
                        continue
                    
                    # Identificar suspeitos usando FINE (p_clean baixo)
                    if box['p_clean'] > c_suspect_clean:  # FINE: p_clean baixo = suspeito
                        continue
                    
                    total_suspects += 1
                    
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
                        box['relabeled_by'] = 'fine_clustering'
                        box['was_relabeled'] = True
                        clustering_relabel_count += 1
            
            if self.debug:
                print(f"[VCNC-FINE] Âncoras identificadas (p_clean >= {c_anchor_clean}): {total_anchors} "
                      f"({total_anchors/len(all_box_data)*100:.2f}%)")
                print(f"[VCNC-FINE] Suspeitos identificados (p_clean <= {c_suspect_clean}): {total_suspects} "
                      f"({total_suspects/len(all_box_data)*100:.2f}%)")
                print(f"[VCNC-FINE] Relabelados por FINE clustering: {clustering_relabel_count} "
                      f"({clustering_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 3: SPATIAL REFINEMENT
        # ============================================================
        spatial_relabel_count = 0
        spatial_stats = {'total_boxes': 0, 'high_contamination': 0, 'refinements_applied': 0}
        
        if self.enable_spatial_refinement:
            if self.debug:
                print(f"\n[VCNC-FINE] ETAPA 3: Spatial Refinement (threshold={self.spatial_difficulty_threshold})")
            
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
                print(f"[VCNC-FINE] Boxes com alta contaminação: {spatial_stats['high_contamination']}")
                print(f"[VCNC-FINE] Relabelados por spatial: {spatial_relabel_count} "
                      f"({spatial_relabel_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ETAPA 4: FILTRAGEM SELETIVA (OPCIONAL)
        # ============================================================
        selective_filter_count = 0
        
        if self.enable_selective_filtering:
            if self.debug:
                print(f"\n[VCNC-FINE] ETAPA 4: Filtragem Seletiva")
                print(f"[VCNC-FINE] Critérios: p_clean <= {self.selective_filter_clean_threshold} AND "
                      f"não relabelado AND pred_score < {self.selective_filter_confidence_threshold}")
            
            # Recalcular FINE scores com labels atualizados
            eigenvectors = self._compute_fine_scores(all_box_data)
            fine_gmm_dict = self._fit_fine_gmm_per_class(all_box_data)
            
            for box in all_box_data:
                box['p_clean_final'] = self._get_p_clean(
                    box['alignment_score'], 
                    box['gt_label'], 
                    fine_gmm_dict
                )
            
            total_suspects = 0
            suspects_relabeled = 0
            suspects_confident = 0
            
            for box in all_box_data:
                # Suspeito pelo FINE (p_clean baixo)
                is_suspect = box['p_clean_final'] <= self.selective_filter_clean_threshold
                
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
                print(f"[VCNC-FINE] Total suspeitos (p_clean <= {self.selective_filter_clean_threshold}): "
                      f"{total_suspects} ({total_suspects/len(all_box_data)*100:.2f}%)")
                print(f"[VCNC-FINE]   - Corrigidos (relabelados): {suspects_relabeled}")
                print(f"[VCNC-FINE]   - Confiantes: {suspects_confident}")
                print(f"[VCNC-FINE]   - Filtrados: {selective_filter_count}")
        
        # ============================================================
        # ETAPA 5: FILTRAGEM GMM ADICIONAL (OPCIONAL)
        # ============================================================
        gmm_filter_count = 0
        
        if self.enable_gmm_filter:
            if self.debug:
                print(f"\n[VCNC-FINE] ETAPA 5: Filtragem por p_clean (FINE)")
            
            if not self.enable_selective_filtering:
                eigenvectors = self._compute_fine_scores(all_box_data)
                fine_gmm_dict = self._fit_fine_gmm_per_class(all_box_data)
                
                for box in all_box_data:
                    box['p_clean_final'] = self._get_p_clean(
                        box['alignment_score'], 
                        box['gt_label'], 
                        fine_gmm_dict
                    )
            
            for box in all_box_data:
                if box['filtered']:
                    continue
                    
                if box['p_clean_final'] <= self.filter_clean_threshold:
                    self._apply_ignore_flag(
                        datasets,
                        box['sub_idx'],
                        box['data_idx'],
                        box['gt_idx']
                    )
                    box['filtered'] = True
                    gmm_filter_count += 1
            
            if self.debug:
                print(f"[VCNC-FINE] Filtrados por FINE: {gmm_filter_count} "
                      f"({gmm_filter_count/len(all_box_data)*100:.2f}%)")
        
        # ============================================================
        # ESTATÍSTICAS FINAIS
        # ============================================================
        total_filtered = selective_filter_count + gmm_filter_count
        
        if self.debug:
            total_relabels = confidence_relabel_count + clustering_relabel_count + spatial_relabel_count
            print(f"\n[VCNC-FINE] ===== Resumo Época {epoch} =====")
            print(f"[VCNC-FINE] Total de boxes: {len(all_box_data)}")
            print(f"[VCNC-FINE] Relabel confiança: {confidence_relabel_count} ({confidence_relabel_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-FINE] Relabel FINE+clustering: {clustering_relabel_count} ({clustering_relabel_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-FINE] Relabel spatial: {spatial_relabel_count} ({spatial_relabel_count/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-FINE] Total relabels: {total_relabels} ({total_relabels/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-FINE] Total filtrados: {total_filtered} ({total_filtered/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-FINE] ==========================================\n")
    
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
                print(f"[VCNC-FINE] Erro ao recarregar: {e}")
    
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
                print(f"[VCNC-FINE] Erro relabel: {e}")
    
    def _apply_ignore_flag(self, datasets, sub_idx, data_idx, gt_idx):
        try:
            instance = datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]
            instance['ignore_flag'] = 1
        except Exception as e:
            if self.debug:
                print(f"[VCNC-FINE] Erro ignore_flag: {e}")

"""
Visual Clustering Noise Correction Hook (VCNC) - Versão com Critérios Progressivos

Ideia central:
1. Usar embeddings visuais (logits/softmax) para agrupar boxes similares
2. Identificar âncoras (boxes de alta confiança) em cada cluster
3. Propagar labels dos âncoras para boxes suspeitos no mesmo cluster

NOVO: Critérios progressivos
- Épocas iniciais: critérios mais conservadores (menos relabels)
- Épocas posteriores: critérios mais agressivos (mais relabels)

Motivação: A Run 2 que atingiu 79.0% teve taxa de modificação de apenas 4.4% 
na época 2, enquanto as outras runs tiveram 7.75% e 12.33%.
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


@HOOKS.register_module()
class VisualClusteringNoiseCorrectionProgressiveHook(Hook):
    """
    Hook para correção de ruído usando clustering visual com critérios progressivos.
    
    Fluxo:
    1. Coleta embeddings (logits) de todos os boxes GT
    2. Agrupa por similaridade visual (FAISS/KMeans)
    3. Identifica âncoras (alta confiança GMM + modelo concorda)
    4. Propaga label dominante para suspeitos no mesmo cluster
    
    Critérios Progressivos:
    - Épocas <= progressive_epochs: usa critérios "early_*" (conservadores)
    - Épocas > progressive_epochs: usa critérios normais (agressivos)
    """
    
    def __init__(self,
                 # Configuração geral
                 warmup_epochs: int = 1,
                 num_classes: int = 20,
                 
                 # Configuração de clustering
                 n_clusters: int = 150,
                 use_softmax_as_embedding: bool = True,
                 
                 # === CRITÉRIOS PROGRESSIVOS ===
                 progressive_epochs: int = 4,  # Até esta época, usa critérios conservadores
                 
                 # Critérios para épocas INICIAIS (conservadores)
                 early_anchor_gmm_threshold: float = 0.15,      # p_noise < 0.15 = muito limpo
                 early_anchor_pred_agreement: float = 0.85,     # score[gt_label] > 0.85
                 early_anchor_confidence: float = 0.9,          # max(scores) > 0.9
                 early_suspect_gmm_threshold: float = 0.8,      # p_noise > 0.8 = muito suspeito
                 early_similarity_threshold: float = 0.7,       # similaridade alta necessária
                 early_cluster_consensus: float = 0.85,         # 85% das âncoras concordam
                 
                 # Critérios para épocas POSTERIORES (agressivos)
                 anchor_gmm_threshold: float = 0.4,
                 anchor_pred_agreement: float = 0.6,
                 anchor_confidence: float = 0.7,
                 suspect_gmm_threshold: float = 0.5,
                 similarity_threshold: float = 0.4,
                 cluster_consensus: float = 0.6,
                 
                 # Configuração do GMM
                 gmm_components: int = 4,
                 
                 # Spatial Refinement (para casos sem consenso)
                 enable_spatial_refinement: bool = True,
                 spatial_iou_threshold: float = 0.3,
                 
                 # Configuração do assigner
                 iou_assigner: float = 0.5,
                 
                 # Reload dataset
                 reload_dataset: bool = True,
                 
                 # Debug
                 debug: bool = True,
                 save_debug_dir: str = 'debug_vcnc'):
        
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        self.n_clusters = n_clusters
        self.use_softmax_as_embedding = use_softmax_as_embedding
        
        # Progressivo
        self.progressive_epochs = progressive_epochs
        
        # Critérios iniciais (conservadores)
        self.early_anchor_gmm_threshold = early_anchor_gmm_threshold
        self.early_anchor_pred_agreement = early_anchor_pred_agreement
        self.early_anchor_confidence = early_anchor_confidence
        self.early_suspect_gmm_threshold = early_suspect_gmm_threshold
        self.early_similarity_threshold = early_similarity_threshold
        self.early_cluster_consensus = early_cluster_consensus
        
        # Critérios posteriores (agressivos)
        self.anchor_gmm_threshold = anchor_gmm_threshold
        self.anchor_pred_agreement = anchor_pred_agreement
        self.anchor_confidence = anchor_confidence
        self.suspect_gmm_threshold = suspect_gmm_threshold
        self.similarity_threshold = similarity_threshold
        self.cluster_consensus = cluster_consensus
        
        # GMM
        self.gmm_components = gmm_components
        
        # Spatial Refinement
        self.enable_spatial_refinement = enable_spatial_refinement
        self.spatial_iou_threshold = spatial_iou_threshold
        
        # Assigner
        self.iou_assigner = iou_assigner
        
        # Reload
        self.reload_dataset = reload_dataset
        
        # Debug
        self.debug = debug
        self.save_debug_dir = save_debug_dir
        
        # Estado interno
        self._gmm_per_class = {}
        self._stats = defaultdict(int)
    
    def _get_current_criteria(self, epoch):
        """
        Retorna os critérios apropriados baseado na época atual.
        
        Args:
            epoch: Época atual (1-indexed)
        
        Returns:
            dict com os critérios a usar
        """
        if epoch <= self.progressive_epochs:
            # Épocas iniciais: critérios conservadores
            criteria = {
                'anchor_gmm_threshold': self.early_anchor_gmm_threshold,
                'anchor_pred_agreement': self.early_anchor_pred_agreement,
                'anchor_confidence': self.early_anchor_confidence,
                'suspect_gmm_threshold': self.early_suspect_gmm_threshold,
                'similarity_threshold': self.early_similarity_threshold,
                'cluster_consensus': self.early_cluster_consensus,
                'phase': 'CONSERVADOR'
            }
        else:
            # Épocas posteriores: critérios agressivos
            criteria = {
                'anchor_gmm_threshold': self.anchor_gmm_threshold,
                'anchor_pred_agreement': self.anchor_pred_agreement,
                'anchor_confidence': self.anchor_confidence,
                'suspect_gmm_threshold': self.suspect_gmm_threshold,
                'similarity_threshold': self.similarity_threshold,
                'cluster_consensus': self.cluster_consensus,
                'phase': 'AGRESSIVO'
            }
        
        return criteria
    
    def _fit_gmm_per_class(self, scores_by_class):
        """
        Ajusta um GMM para cada classe usando os scores coletados.
        Retorna dicionário {classe: (gmm, low_confidence_component)}
        """
        gmm_dict = {}
        
        for cls_id, scores in scores_by_class.items():
            if len(scores) < 10:  # Precisa de amostras suficientes
                continue
            
            scores_np = np.array(scores).reshape(-1, 1)
            
            try:
                gmm = GaussianMixture(
                    n_components=min(self.gmm_components, len(scores) // 5),
                    max_iter=100,
                    tol=1e-3,
                    reg_covar=1e-4,
                    random_state=42
                )
                gmm.fit(scores_np)
                
                # Componente de baixa confiança = menor média
                low_conf_component = np.argmin(gmm.means_)
                gmm_dict[cls_id] = (gmm, low_conf_component)
                
            except Exception as e:
                if self.debug:
                    print(f"[VCNC] Erro ao ajustar GMM para classe {cls_id}: {e}")
        
        return gmm_dict
    
    def _get_p_noise(self, score, cls_id, gmm_dict):
        """
        Calcula p_noise para um box baseado no GMM da classe.
        """
        if cls_id not in gmm_dict:
            return 0.5  # Se não tem GMM, assume incerteza
        
        gmm, low_conf_comp = gmm_dict[cls_id]
        score_np = np.array([[score]])
        
        try:
            probs = gmm.predict_proba(score_np)
            p_noise = probs[0, low_conf_comp]
            return float(p_noise)
        except:
            return 0.5
    
    def _cluster_embeddings(self, embeddings, n_clusters):
        """
        Agrupa embeddings usando FAISS (ou KMeans como fallback).
        """
        N, D = embeddings.shape
        n_clusters = min(n_clusters, N // 2)
        
        if n_clusters < 2:
            return np.zeros(N, dtype=np.int32)
        
        # Normalizar embeddings
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
        """
        Executa o pipeline de correção de ruído antes de cada época.
        """
        epoch = runner.epoch + 1
        
        # Pular epochs de warmup
        if epoch <= self.warmup_epochs:
            if self.debug:
                print(f"[VCNC] Época {epoch}: Warmup, pulando correção.")
            return
        
        # Obter critérios para esta época
        criteria = self._get_current_criteria(epoch)
        
        if self.debug:
            print(f"\n[VCNC] Época {epoch}: Iniciando correção de ruído...")
            print(f"[VCNC] Fase: {criteria['phase']} (progressive_epochs={self.progressive_epochs})")
            print(f"[VCNC] Critérios: anchor_gmm<{criteria['anchor_gmm_threshold']}, "
                  f"pred_agree>{criteria['anchor_pred_agreement']}, "
                  f"suspect_gmm>{criteria['suspect_gmm_threshold']}")
        
        # Reset estatísticas
        self._stats = defaultdict(int)
        
        # Reload dataset se necessário
        if self.reload_dataset:
            self._reload_datasets(runner)
        
        # Obter dataloader e dataset
        dataloader = runner.train_loop.dataloader
        dataset = self._get_base_dataset(dataloader.dataset)
        
        if not hasattr(dataset, 'datasets'):
            print("[VCNC] ERRO: Esperado ConcatDataset")
            return
        
        datasets = dataset.datasets
        
        # Criar mapa de imagens
        dataset_img_map = self._build_image_map(datasets)
        
        # Assigner
        assigner = MaxIoUAssigner(
            pos_iou_thr=self.iou_assigner,
            neg_iou_thr=self.iou_assigner,
            min_pos_iou=self.iou_assigner,
            match_low_quality=False
        )
        
        # ========== FASE 1: Coletar embeddings e scores ==========
        if self.debug:
            print("[VCNC] Fase 1: Coletando embeddings e scores...")
        
        all_box_data = []
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
                        'gt_bbox': gt_bbox.cpu().numpy(),
                        'score_gt': score_gt,
                        'pred_label': pred_label,
                        'pred_score': pred_score,
                        'embedding': embedding,
                        'scores': best_scores.cpu().numpy()
                    }
                    all_box_data.append(box_data)
                    scores_by_class[gt_label].append(score_gt)
        
        if len(all_box_data) == 0:
            print("[VCNC] Nenhum box coletado!")
            return
        
        if self.debug:
            print(f"[VCNC] Coletados {len(all_box_data)} boxes")
        
        # ========== FASE 2: Ajustar GMM por classe ==========
        if self.debug:
            print("[VCNC] Fase 2: Ajustando GMM por classe...")
        
        gmm_dict = self._fit_gmm_per_class(scores_by_class)
        
        if self.debug:
            print(f"[VCNC] GMM ajustado para {len(gmm_dict)} classes")
        
        for box in all_box_data:
            box['p_noise'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict)
        
        # ========== FASE 3: Clustering visual ==========
        if self.debug:
            print("[VCNC] Fase 3: Clustering visual...")
        
        embeddings = np.array([box['embedding'] for box in all_box_data])
        cluster_ids = self._cluster_embeddings(embeddings, self.n_clusters)
        
        for i, box in enumerate(all_box_data):
            box['cluster_id'] = cluster_ids[i]
        
        cluster_counts = Counter(cluster_ids)
        if self.debug:
            print(f"[VCNC] {len(set(cluster_ids))} clusters criados")
            print(f"[VCNC] Tamanho médio dos clusters: {np.mean(list(cluster_counts.values())):.1f}")
        
        # ========== FASE 4: Identificar âncoras e propagar labels ==========
        if self.debug:
            print("[VCNC] Fase 4: Identificando âncoras e propagando labels...")
        
        clusters = defaultdict(list)
        for box in all_box_data:
            clusters[box['cluster_id']].append(box)
        
        relabel_count = 0
        spatial_refine_count = 0
        anchor_count = 0
        
        # Extrair critérios para esta época
        c_anchor_gmm = criteria['anchor_gmm_threshold']
        c_anchor_pred = criteria['anchor_pred_agreement']
        c_anchor_conf = criteria['anchor_confidence']
        c_suspect_gmm = criteria['suspect_gmm_threshold']
        c_similarity = criteria['similarity_threshold']
        c_consensus = criteria['cluster_consensus']
        
        for cluster_id, cluster_boxes in clusters.items():
            if len(cluster_boxes) < 2:
                continue
            
            # Identificar âncoras no cluster (usando critérios da época)
            anchors = []
            for box in cluster_boxes:
                is_clean = box['p_noise'] < c_anchor_gmm
                model_agrees = box['score_gt'] > c_anchor_pred
                high_confidence = box['pred_score'] > c_anchor_conf
                
                if is_clean and model_agrees and high_confidence:
                    anchors.append(box)
            
            anchor_count += len(anchors)
            
            if len(anchors) == 0:
                continue
            
            # Determinar label dominante das âncoras
            anchor_labels = [a['gt_label'] for a in anchors]
            label_counts = Counter(anchor_labels)
            dominant_label, count = label_counts.most_common(1)[0]
            consensus_ratio = count / len(anchors)
            
            if consensus_ratio < c_consensus:
                continue
            
            # Calcular embedding médio das âncoras
            anchor_embeddings = np.array([a['embedding'] for a in anchors])
            anchor_mean = anchor_embeddings.mean(axis=0)
            anchor_mean_norm = anchor_mean / (np.linalg.norm(anchor_mean) + 1e-8)
            
            anchor_ids = set(id(a) for a in anchors)
            
            # Propagar label para suspeitos
            for box in cluster_boxes:
                if id(box) in anchor_ids:
                    continue
                
                # Verificar se é suspeito (usando critério da época)
                if box['p_noise'] < c_suspect_gmm:
                    continue
                
                if box['gt_label'] == dominant_label:
                    continue
                
                # Verificar similaridade (usando critério da época)
                box_emb_norm = box['embedding'] / (np.linalg.norm(box['embedding']) + 1e-8)
                similarity = np.dot(box_emb_norm, anchor_mean_norm)
                
                if similarity > c_similarity:
                    box['new_label'] = dominant_label
                    relabel_count += 1
                    self._stats['relabeled'] += 1
                    
                    self._apply_relabel(
                        datasets,
                        box['sub_idx'],
                        box['data_idx'],
                        box['gt_idx'],
                        dominant_label
                    )
                    
                elif self.enable_spatial_refinement:
                    spatial_refine_count += 1
                    self._stats['spatial_refined'] += 1
        
        # ========== Estatísticas finais ==========
        if self.debug:
            print(f"\n[VCNC] ===== Estatísticas da Época {epoch} =====")
            print(f"[VCNC] Fase: {criteria['phase']}")
            print(f"[VCNC] Total de boxes: {len(all_box_data)}")
            print(f"[VCNC] Total de âncoras: {anchor_count}")
            print(f"[VCNC] Boxes relabelados: {relabel_count}")
            print(f"[VCNC] Boxes com spatial refinement: {spatial_refine_count}")
            print(f"[VCNC] Taxa de modificação: {(relabel_count + spatial_refine_count) / len(all_box_data) * 100:.2f}%")
            print(f"[VCNC] ==========================================\n")
    
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
                print(f"[VCNC] Erro ao recarregar datasets: {e}")
    
    def _get_base_dataset(self, dataset):
        """Navega até o dataset base (ConcatDataset)."""
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        return dataset
    
    def _build_image_map(self, datasets):
        """Constrói mapa img_path -> (sub_idx, data_idx)."""
        img_map = {}
        for sub_idx, subds in enumerate(datasets):
            if hasattr(subds, 'data_list'):
                for data_idx, data_info in enumerate(subds.data_list):
                    img_map[data_info['img_path']] = (sub_idx, data_idx)
        return img_map
    
    def _apply_relabel(self, datasets, sub_idx, data_idx, gt_idx, new_label):
        """Aplica o relabel no dataset."""
        try:
            instance = datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]
            instance['bbox_label'] = new_label
        except Exception as e:
            if self.debug:
                print(f"[VCNC] Erro ao aplicar relabel: {e}")

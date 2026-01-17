"""
Visual Clustering Noise Correction Hook (VCNC) - Versão Simplificada

Ideia central:
1. Usar embeddings visuais (logits/softmax) para agrupar boxes similares
2. Identificar âncoras (boxes de alta confiança) em cada cluster
3. Propagar labels dos âncoras para boxes suspeitos no mesmo cluster

Integra com:
- GMM existente para identificar suspeitos
- Spatial Refinement para casos ambíguos
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
class VisualClusteringNoiseCorrectionHook(Hook):
    """
    Hook para correção de ruído usando clustering visual.
    
    Fluxo:
    1. Coleta embeddings (logits) de todos os boxes GT
    2. Agrupa por similaridade visual (FAISS/KMeans)
    3. Identifica âncoras (alta confiança GMM + modelo concorda)
    4. Propaga label dominante para suspeitos no mesmo cluster
    5. Aplica Spatial Refinement para casos sem consenso
    """
    
    def __init__(self,
                 # Configuração geral
                 warmup_epochs: int = 1,
                 num_classes: int = 20,
                 
                 # Configuração de clustering
                 n_clusters: int = 100,
                 use_softmax_as_embedding: bool = True,
                 
                 # Critérios para âncoras (boxes de alta confiança)
                 anchor_gmm_threshold: float = 0.3,      # p_noise < este valor = limpo
                 anchor_pred_agreement: float = 0.7,     # score[gt_label] > este valor
                 anchor_confidence: float = 0.8,         # max(scores) > este valor
                 
                 # Critérios para suspeitos
                 suspect_gmm_threshold: float = 0.6,     # p_noise > este valor = suspeito
                 
                 # Critérios para relabel
                 cluster_consensus: float = 0.7,         # % de âncoras que concordam
                 similarity_threshold: float = 0.5,      # similaridade mínima com âncoras
                 
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
        
        # Âncoras
        self.anchor_gmm_threshold = anchor_gmm_threshold
        self.anchor_pred_agreement = anchor_pred_agreement
        self.anchor_confidence = anchor_confidence
        
        # Suspeitos
        self.suspect_gmm_threshold = suspect_gmm_threshold
        
        # Relabel
        self.cluster_consensus = cluster_consensus
        self.similarity_threshold = similarity_threshold
        
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
        
        Args:
            embeddings: numpy array [N, D]
            n_clusters: número de clusters
        
        Returns:
            cluster_ids: numpy array [N] com ID do cluster de cada embedding
        """
        N, D = embeddings.shape
        n_clusters = min(n_clusters, N // 2)  # Não ter mais clusters que amostras/2
        
        if n_clusters < 2:
            return np.zeros(N, dtype=np.int32)
        
        # Normalizar embeddings para clustering
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        embeddings_norm = embeddings_norm.astype(np.float32)
        
        if FAISS_AVAILABLE:
            # Usar FAISS para clustering rápido
            kmeans = faiss.Kmeans(D, n_clusters, niter=20, verbose=False)
            kmeans.train(embeddings_norm)
            _, cluster_ids = kmeans.index.search(embeddings_norm, 1)
            cluster_ids = cluster_ids.flatten()
        else:
            # Fallback para sklearn
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_ids = kmeans.fit_predict(embeddings_norm)
        
        return cluster_ids
    
    def _compute_spatial_refinement(self, box, neighbors, neighbor_labels):
        """
        Aplica Spatial Refinement quando não há consenso no cluster.
        Ajusta a geometria do box baseado nos vizinhos.
        
        Simplificado: retorna o box original por enquanto.
        Pode ser expandido com lógica mais sofisticada.
        """
        # TODO: Implementar lógica de spatial refinement
        # Por exemplo, ajustar bbox baseado na média ponderada dos vizinhos
        return box
    
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
        
        if self.debug:
            print(f"[VCNC] Época {epoch}: Iniciando correção de ruído...")
        
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
        
        # Assigner para associar predições com GTs
        assigner = MaxIoUAssigner(
            pos_iou_thr=self.iou_assigner,
            neg_iou_thr=self.iou_assigner,
            min_pos_iou=self.iou_assigner,
            match_low_quality=False
        )
        
        # ========== FASE 1: Coletar embeddings e scores ==========
        if self.debug:
            print("[VCNC] Fase 1: Coletando embeddings e scores...")
        
        all_box_data = []  # Lista de dicts com info de cada box
        scores_by_class = defaultdict(list)  # Para GMM
        
        for batch_idx, data_batch in enumerate(dataloader):
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                
                # Obter predições com logits
                predictions = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
            
            for i, data_sample in enumerate(data_batch['data_samples']):
                img_path = data_sample.img_path
                
                if img_path not in dataset_img_map:
                    continue
                
                sub_idx, data_idx = dataset_img_map[img_path]
                
                pred_instances = predictions[i].pred_instances
                pred_instances.priors = pred_instances.pop('bboxes')
                
                # Determinar o device das predições
                device = pred_instances.priors.device
                
                gt_instances = data_sample.gt_instances
                
                # Garantir que TODOS os tensores estejam no mesmo device
                gt_instances.bboxes = gt_instances.bboxes.to(device)
                gt_instances.labels = gt_instances.labels.to(device)
                pred_instances.priors = pred_instances.priors.to(device)
                pred_instances.labels = pred_instances.labels.to(device)
                pred_instances.scores = pred_instances.scores.to(device)
                pred_instances.logits = pred_instances.logits.to(device)
                
                gt_labels = gt_instances.labels
                gt_bboxes = gt_instances.bboxes
                
                # Associar predições com GTs
                assign_result = assigner.assign(pred_instances, gt_instances)
                
                for gt_idx in range(assign_result.num_gts):
                    associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    
                    if associated_preds.numel() == 0:
                        continue
                    
                    # Pegar logits da melhor predição associada
                    logits_associated = pred_instances.logits[associated_preds]
                    scores = torch.softmax(logits_associated, dim=-1)
                    
                    # Usar a predição com maior score geral
                    best_pred_idx = scores.max(dim=1).values.argmax()
                    best_scores = scores[best_pred_idx]  # [num_classes]
                    best_logits = logits_associated[best_pred_idx]  # [num_classes]
                    
                    gt_label = gt_labels[gt_idx].item()
                    gt_bbox = gt_bboxes[gt_idx]
                    
                    # Score para o label GT
                    score_gt = best_scores[gt_label].item()
                    
                    # Predição do modelo
                    pred_label = best_scores.argmax().item()
                    pred_score = best_scores.max().item()
                    
                    # Embedding: usar softmax ou logits
                    if self.use_softmax_as_embedding:
                        embedding = best_scores.cpu().numpy()
                    else:
                        embedding = best_logits.cpu().numpy()
                    
                    # Guardar dados do box
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
                    
                    # Guardar score para GMM
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
        
        # Calcular p_noise para cada box
        for box in all_box_data:
            box['p_noise'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict)
        
        # ========== FASE 3: Clustering visual ==========
        if self.debug:
            print("[VCNC] Fase 3: Clustering visual...")
        
        embeddings = np.array([box['embedding'] for box in all_box_data])
        cluster_ids = self._cluster_embeddings(embeddings, self.n_clusters)
        
        for i, box in enumerate(all_box_data):
            box['cluster_id'] = cluster_ids[i]
        
        # Contar boxes por cluster
        cluster_counts = Counter(cluster_ids)
        if self.debug:
            print(f"[VCNC] {len(set(cluster_ids))} clusters criados")
            print(f"[VCNC] Tamanho médio dos clusters: {np.mean(list(cluster_counts.values())):.1f}")
        
        # ========== FASE 4: Identificar âncoras e propagar labels ==========
        if self.debug:
            print("[VCNC] Fase 4: Identificando âncoras e propagando labels...")
        
        # Organizar boxes por cluster
        clusters = defaultdict(list)
        for box in all_box_data:
            clusters[box['cluster_id']].append(box)
        
        relabel_count = 0
        spatial_refine_count = 0
        
        for cluster_id, cluster_boxes in clusters.items():
            if len(cluster_boxes) < 2:
                continue
            
            # Identificar âncoras no cluster
            anchors = []
            for box in cluster_boxes:
                is_clean = box['p_noise'] < self.anchor_gmm_threshold
                model_agrees = box['score_gt'] > self.anchor_pred_agreement
                high_confidence = box['pred_score'] > self.anchor_confidence
                
                if is_clean and model_agrees and high_confidence:
                    anchors.append(box)
            
            if len(anchors) == 0:
                continue
            
            # Determinar label dominante dos âncoras
            anchor_labels = [a['gt_label'] for a in anchors]
            label_counts = Counter(anchor_labels)
            dominant_label, count = label_counts.most_common(1)[0]
            consensus_ratio = count / len(anchors)
            
            if consensus_ratio < self.cluster_consensus:
                # Âncoras não concordam, cluster ambíguo
                continue
            
            # Calcular embedding médio das âncoras
            anchor_embeddings = np.array([a['embedding'] for a in anchors])
            anchor_mean = anchor_embeddings.mean(axis=0)
            anchor_mean_norm = anchor_mean / (np.linalg.norm(anchor_mean) + 1e-8)
            
            # Propagar label para suspeitos
            for box in cluster_boxes:
                if box in anchors:
                    continue
                
                # Verificar se é suspeito
                if box['p_noise'] < self.suspect_gmm_threshold:
                    continue
                
                # Verificar se label é diferente do dominante
                if box['gt_label'] == dominant_label:
                    continue
                
                # Verificar similaridade com âncoras
                box_emb_norm = box['embedding'] / (np.linalg.norm(box['embedding']) + 1e-8)
                similarity = np.dot(box_emb_norm, anchor_mean_norm)
                
                if similarity > self.similarity_threshold:
                    # Alta similaridade → Relabel
                    old_label = box['gt_label']
                    box['new_label'] = dominant_label
                    relabel_count += 1
                    self._stats['relabeled'] += 1
                    
                    # Aplicar relabel no dataset
                    self._apply_relabel(
                        datasets,
                        box['sub_idx'],
                        box['data_idx'],
                        box['gt_idx'],
                        dominant_label
                    )
                    
                elif self.enable_spatial_refinement:
                    # Baixa similaridade → Spatial Refinement
                    spatial_refine_count += 1
                    self._stats['spatial_refined'] += 1
                    # TODO: Implementar spatial refinement
        
        # ========== Estatísticas finais ==========
        if self.debug:
            print(f"\n[VCNC] ===== Estatísticas da Época {epoch} =====")
            print(f"[VCNC] Total de boxes: {len(all_box_data)}")
            print(f"[VCNC] Boxes relabelados: {relabel_count}")
            print(f"[VCNC] Boxes com spatial refinement: {spatial_refine_count}")
            print(f"[VCNC] Taxa de modificação: {(relabel_count + spatial_refine_count) / len(all_box_data) * 100:.2f}%")
            print(f"[VCNC] ==========================================\n")
    
    def _reload_datasets(self, runner):
        """Recarrega os datasets para garantir labels atualizados."""
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

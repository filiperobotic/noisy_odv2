"""
VCNC-KNN com Gate por Assimetria/Severidade da Matriz de Confusão.

Diferença em relação ao vcnc_knn_spatial_hook.py:
- Antes de cada relabel (Etapa 1 por confiança e Etapa 2 por KNN), consulta
  uma matriz de confusão construída na própria época com pred vs GT.
- Pares (i, j) com SEVERIDADE outlier (severity > median + 3*MAD na distribuição
  off-diagonal de severidades) são marcados como "pares ruidosos".
- Para esses pares, relabel é BLOQUEADO em qualquer direção (i->j ou j->i).
- Para os demais pares, comportamento idêntico ao VCNC-KNN original.

Desenho:
- Severidade S(i,j) = (C[i,j] + C[j,i]) / 2 captura tanto ruído UNIDIRECIONAL
  (uma das taxas alta) quanto BIDIRECIONAL (ambas altas).
- Em ruído simétrico, C é aproximadamente uniforme baixa em todas as células
  → MAD pequeno → poucos/nenhum outlier → comportamento ≈ VCNC-KNN original.
- Em ruído assimétrico, pares de ruído têm S muito maior que mediana → outliers
  → relabel bloqueado nesses pares → erro não propaga.

NÃO ADICIONA NOVOS HIPERPARÂMETROS TUNÁVEIS:
- 3 em "3*MAD" é convenção estatística (≈ 3-sigma)
- min_samples=50 reusa o min_samples_for_confusion já existente em outros hooks
- enable_confusion_gate é flag de ablação, não tuning
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

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] FAISS não disponível. Usando sklearn NearestNeighbors como fallback.")
    from sklearn.neighbors import NearestNeighbors


# ============================================================
# UTILITÁRIOS DE DATASET (idênticos ao hook original)
# ============================================================

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


# ============================================================
# SPATIAL REFINEMENT (idêntico ao hook original)
# ============================================================

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
    stats = {'total_boxes': len(boxes), 'high_contamination': 0, 'refinements_applied': 0}
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


# ============================================================
# HOOK PRINCIPAL
# ============================================================

@HOOKS.register_module()
class VCNCKNNConfusionAwareHook(Hook):
    """
    VCNC-KNN com gate de relabel baseado na assimetria/severidade da matriz
    de confusão observada no treinamento.

    Pipeline (mesmo do VCNC-KNN, com modificação cirúrgica em Etapa 1 e 2):
    1. Relabel por confiança alta — gate por par confuso (NEW)
    2. Relabel por KNN visual — gate por par confuso (NEW)
    3. Spatial Refinement (sem mudança)
    4. Filtragem Seletiva (opcional, sem mudança)
    5. Filtragem GMM (opcional, sem mudança)
    """

    def __init__(self,
                 # Configuração geral
                 warmup_epochs: int = 1,
                 num_classes: int = 20,

                 # === ETAPA 1: Relabel por confiança ===
                 enable_confidence_relabel: bool = True,
                 relabel_confidence_threshold: float = 0.9,

                 # === ETAPA 2: KNN Visual ===
                 enable_knn_relabel: bool = True,
                 use_softmax_as_embedding: bool = True,
                 knn_k: int = 100,
                 knn_min_anchors: int = 5,
                 knn_consensus_threshold: float = 0.6,
                 knn_distance_weighted: bool = True,

                 # Critérios progressivos
                 progressive_epochs: int = 4,

                 early_anchor_gmm_threshold: float = 0.15,
                 early_anchor_pred_agreement: float = 0.85,
                 early_anchor_confidence: float = 0.9,
                 early_suspect_gmm_threshold: float = 0.8,

                 anchor_gmm_threshold: float = 0.4,
                 anchor_pred_agreement: float = 0.6,
                 anchor_confidence: float = 0.7,
                 suspect_gmm_threshold: float = 0.5,

                 # === GATE DE CONFUSÃO (NEW) ===
                 enable_confusion_gate: bool = True,
                 # Mínimo de amostras de uma classe para entrar no cálculo de C[i,*]
                 # (mesmo critério do VCNCConfusionMatrixHook).
                 confusion_gate_min_samples: int = 50,
                 # Multiplicador de MAD para detecção de outliers de severidade.
                 # 3.0 ≈ 3-sigma (convenção estatística robusta), não tunado.
                 confusion_gate_mad_factor: float = 3.0,
                 # Razão multiplicativa: pares com severidade > k * mediana são outliers.
                 # 10x = "uma ordem de grandeza acima do típico" (estrutural).
                 confusion_gate_ratio_factor: float = 10.0,
                 # Aplica gate só na fase agressiva (após progressive_epochs)?
                 # Default False — gate ativo desde o fim do warmup.
                 confusion_gate_aggressive_only: bool = False,

                 # === ETAPA 3: Spatial ===
                 enable_spatial_refinement: bool = True,
                 spatial_difficulty_threshold: float = 0.5,

                 # === ETAPA 4: Filtragem Seletiva ===
                 enable_selective_filtering: bool = False,
                 selective_filter_gmm_threshold: float = 0.5,
                 selective_filter_confidence_threshold: float = 0.7,

                 # === ETAPA 5: Filtragem GMM ===
                 enable_gmm_filter: bool = False,
                 gmm_components: int = 4,
                 filter_gmm_threshold: float = 0.7,

                 iou_assigner: float = 0.5,
                 reload_dataset: bool = True,
                 debug: bool = True):

        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes

        self.enable_confidence_relabel = enable_confidence_relabel
        self.relabel_confidence_threshold = relabel_confidence_threshold

        self.enable_knn_relabel = enable_knn_relabel
        self.use_softmax_as_embedding = use_softmax_as_embedding
        self.knn_k = knn_k
        self.knn_min_anchors = knn_min_anchors
        self.knn_consensus_threshold = knn_consensus_threshold
        self.knn_distance_weighted = knn_distance_weighted

        self.progressive_epochs = progressive_epochs

        self.early_anchor_gmm_threshold = early_anchor_gmm_threshold
        self.early_anchor_pred_agreement = early_anchor_pred_agreement
        self.early_anchor_confidence = early_anchor_confidence
        self.early_suspect_gmm_threshold = early_suspect_gmm_threshold

        self.anchor_gmm_threshold = anchor_gmm_threshold
        self.anchor_pred_agreement = anchor_pred_agreement
        self.anchor_confidence = anchor_confidence
        self.suspect_gmm_threshold = suspect_gmm_threshold

        # Gate de confusão
        self.enable_confusion_gate = enable_confusion_gate
        self.confusion_gate_min_samples = confusion_gate_min_samples
        self.confusion_gate_mad_factor = confusion_gate_mad_factor
        self.confusion_gate_ratio_factor = confusion_gate_ratio_factor
        self.confusion_gate_aggressive_only = confusion_gate_aggressive_only

        self.enable_spatial_refinement = enable_spatial_refinement
        self.spatial_difficulty_threshold = spatial_difficulty_threshold

        self.enable_selective_filtering = enable_selective_filtering
        self.selective_filter_gmm_threshold = selective_filter_gmm_threshold
        self.selective_filter_confidence_threshold = selective_filter_confidence_threshold

        self.enable_gmm_filter = enable_gmm_filter
        self.gmm_components = gmm_components
        self.filter_gmm_threshold = filter_gmm_threshold

        self.iou_assigner = iou_assigner
        self.reload_dataset = reload_dataset
        self.debug = debug

    # --------- helpers idênticos ao hook KNN base ---------

    def _get_current_criteria(self, epoch):
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
        gmm_dict = {}
        for cls_id, scores in scores_by_class.items():
            if len(scores) < 10:
                continue
            scores_np = np.array(scores).reshape(-1, 1)
            try:
                n_comp = min(self.gmm_components, len(scores) // 5)
                if n_comp < 2:
                    n_comp = 2
                gmm = GaussianMixture(n_components=n_comp, max_iter=100, tol=1e-3,
                                      reg_covar=1e-4, random_state=42)
                gmm.fit(scores_np)
                low_conf_component = np.argmin(gmm.means_)
                gmm_dict[cls_id] = (gmm, low_conf_component)
            except Exception as e:
                if self.debug:
                    print(f"[VCNC-KNN-CA] Erro GMM classe {cls_id}: {e}")
        return gmm_dict

    def _get_p_noise(self, score, cls_id, gmm_dict):
        if cls_id not in gmm_dict:
            return 0.5
        gmm, low_conf_comp = gmm_dict[cls_id]
        try:
            probs = gmm.predict_proba(np.array([[score]]))
            return float(probs[0, low_conf_comp])
        except Exception:
            return 0.5

    def _build_knn_index(self, embeddings):
        N, D = embeddings.shape
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        embeddings_norm = embeddings_norm.astype(np.float32)
        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(D)
            index.add(embeddings_norm)
            return index, embeddings_norm
        else:
            nn = NearestNeighbors(n_neighbors=self.knn_k + 1, metric='cosine')
            nn.fit(embeddings_norm)
            return nn, embeddings_norm

    def _find_knn(self, index, embeddings_norm, query_idx, k):
        query = embeddings_norm[query_idx:query_idx+1]
        if FAISS_AVAILABLE:
            similarities, indices = index.search(query, k + 1)
            similarities, indices = similarities[0], indices[0]
            mask = indices != query_idx
            indices = indices[mask][:k]
            similarities = similarities[mask][:k]
        else:
            distances, indices = index.kneighbors(query, n_neighbors=k + 1)
            distances, indices = distances[0], indices[0]
            mask = indices != query_idx
            indices = indices[mask][:k]
            distances = distances[mask][:k]
            similarities = 1 - distances
        return indices, similarities

    def _knn_vote(self, neighbor_indices, neighbor_similarities, all_box_data,
                  anchor_mask, distance_weighted=True):
        anchor_neighbors, anchor_similarities = [], []
        for idx, sim in zip(neighbor_indices, neighbor_similarities):
            if anchor_mask[idx]:
                anchor_neighbors.append(idx)
                anchor_similarities.append(sim)
        if len(anchor_neighbors) < self.knn_min_anchors:
            return None, 0.0, len(anchor_neighbors)
        anchor_labels = [all_box_data[idx]['gt_label'] for idx in anchor_neighbors]
        if distance_weighted:
            label_weights = defaultdict(float)
            for label, sim in zip(anchor_labels, anchor_similarities):
                label_weights[label] += sim
            total_weight = sum(label_weights.values())
            best_label = max(label_weights, key=label_weights.get)
            confidence = label_weights[best_label] / total_weight
        else:
            label_counts = Counter(anchor_labels)
            best_label, count = label_counts.most_common(1)[0]
            confidence = count / len(anchor_labels)
        return best_label, confidence, len(anchor_neighbors)

    # --------- NEW: cálculo do gate de confusão ---------

    def _compute_confusion_signals(self, all_box_data):
        """
        Constrói matriz de confusão direcional C[i,j] = P(pred=j | gt=i)
        e detecta pares ruidosos por outlier de severidade S(i,j) = (C[i,j] + C[j,i])/2.

        Retorna:
            confused_pairs: set de tuplas (i, j) com i != j marcadas como ruidosas.
                Ambas direções (i,j) e (j,i) são incluídas para cada par detectado.
            stats: dict com mediana, MAD, threshold e top pares (para log/diagnóstico).
        """
        K = self.num_classes
        # Contagens
        counts = np.zeros((K, K), dtype=np.float64)  # counts[i, j] = #(GT=i, pred=j)
        gt_totals = np.zeros(K, dtype=np.float64)

        for box in all_box_data:
            i = box['gt_label']
            j = box['pred_label']
            if 0 <= i < K and 0 <= j < K:
                counts[i, j] += 1.0
                gt_totals[i] += 1.0

        # Matriz de confusão direcional (linha = GT, coluna = pred)
        C = np.zeros((K, K), dtype=np.float64)
        valid_classes = gt_totals >= self.confusion_gate_min_samples
        for i in range(K):
            if valid_classes[i] and gt_totals[i] > 0:
                C[i, :] = counts[i, :] / gt_totals[i]

        # Severidades fora da diagonal, apenas para pares com ambas as classes válidas
        sev_pairs = []  # lista de (i, j, severity) com i < j
        for i in range(K):
            for j in range(i + 1, K):
                if not (valid_classes[i] and valid_classes[j]):
                    continue
                severity = 0.5 * (C[i, j] + C[j, i])
                sev_pairs.append((i, j, severity))

        if len(sev_pairs) == 0:
            return set(), {
                'median': 0.0, 'mad': 0.0, 'threshold': 0.0,
                'n_valid_pairs': 0, 'n_confused_pairs': 0, 'top_pairs': []
            }

        severities = np.array([s for _, _, s in sev_pairs], dtype=np.float64)
        median_sev = float(np.median(severities))
        mad_sev = float(np.median(np.abs(severities - median_sev)))

        # Threshold combinado:
        # (a) MAD outlier: median + k*MAD — significância estatística
        # (b) Razão multiplicativa: 10 * median — uma ordem de grandeza acima do típico
        # Pegamos o MAIS RESTRITIVO dos dois.
        # Razão: em distribuições com bulk≈0 (típico de assimétrico), MAD≈0 e (a)
        # vira permissivo demais. Em simétrico, max/median é ~2x e (b) impede over-flag.
        thr_mad = median_sev + self.confusion_gate_mad_factor * mad_sev
        thr_ratio = self.confusion_gate_ratio_factor * median_sev
        threshold = max(thr_mad, thr_ratio)

        confused_pairs = set()
        for i, j, sev in sev_pairs:
            if sev > threshold:
                confused_pairs.add((i, j))
                confused_pairs.add((j, i))  # bidirecional, bloqueia em qualquer direção

        # Diagnóstico: top-10 pares por severidade
        sev_pairs_sorted = sorted(sev_pairs, key=lambda t: t[2], reverse=True)
        top_pairs = []
        for i, j, sev in sev_pairs_sorted[:10]:
            top_pairs.append({
                'pair': (i, j),
                'severity': sev,
                'C_ij': float(C[i, j]),
                'C_ji': float(C[j, i]),
                'is_confused': sev > threshold,
            })

        stats = {
            'median': median_sev,
            'mad': mad_sev,
            'threshold': threshold,
            'n_valid_pairs': len(sev_pairs),
            'n_confused_pairs': len(confused_pairs) // 2,  # dividido por 2 (par bidirecional)
            'top_pairs': top_pairs,
        }
        return confused_pairs, stats

    def _is_gate_active(self, epoch):
        if not self.enable_confusion_gate:
            return False
        if self.confusion_gate_aggressive_only and epoch <= self.progressive_epochs:
            return False
        return True

    # --------- before_train_epoch ---------

    def before_train_epoch(self, runner):
        epoch = runner.epoch + 1

        if epoch <= self.warmup_epochs:
            if self.debug:
                print(f"[VCNC-KNN-CA] Época {epoch}: Warmup, pulando.")
            return

        if self.debug:
            print(f"\n[VCNC-KNN-CA] ========== Época {epoch} ==========")

        dataloader = runner.train_loop.dataloader
        dataset = dataloader.dataset

        if self.reload_dataset:
            reload_leaf_datasets(dataset)

        datasets = unwrap_to_leaf_datasets(dataset)
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
            print("[VCNC-KNN-CA] Coletando dados...")

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
                        'was_relabeled': False,
                        'filtered': False,
                    }
                    all_box_data.append(box_data)
                    scores_by_class[gt_label].append(score_gt)
                    boxes_by_image[img_path].append(box_data)

        if len(all_box_data) == 0:
            print("[VCNC-KNN-CA] Nenhum box coletado!")
            return

        if self.debug:
            print(f"[VCNC-KNN-CA] Coletados {len(all_box_data)} boxes em {len(boxes_by_image)} imagens")

        # ============================================================
        # GATE DE CONFUSÃO (NEW): computa pares ruidosos por outlier de severidade
        # ============================================================
        gate_active = self._is_gate_active(epoch)
        confused_pairs = set()
        confusion_stats = None
        if gate_active:
            confused_pairs, confusion_stats = self._compute_confusion_signals(all_box_data)
            if self.debug:
                print(f"\n[VCNC-KNN-CA] === GATE DE CONFUSÃO ===")
                print(f"[VCNC-KNN-CA] Pares válidos analisados: {confusion_stats['n_valid_pairs']}")
                print(f"[VCNC-KNN-CA] Pares marcados como ruidosos: {confusion_stats['n_confused_pairs']}")
                print(f"[VCNC-KNN-CA] Severidade — mediana: {confusion_stats['median']:.4f}, "
                      f"MAD: {confusion_stats['mad']:.4f}")
                print(f"[VCNC-KNN-CA] Threshold = max(med+{self.confusion_gate_mad_factor}*MAD, "
                      f"{self.confusion_gate_ratio_factor}*med) = "
                      f"{confusion_stats['threshold']:.4f}")
                print(f"[VCNC-KNN-CA] Top 10 pares por severidade:")
                for tp in confusion_stats['top_pairs']:
                    flag = "  ★ RUIDOSO" if tp['is_confused'] else ""
                    i, j = tp['pair']
                    print(f"[VCNC-KNN-CA]   classes ({i}, {j}): sev={tp['severity']:.4f} "
                          f"C[{i}->{j}]={tp['C_ij']:.4f}  C[{j}->{i}]={tp['C_ji']:.4f}{flag}")
        else:
            if self.debug:
                print(f"[VCNC-KNN-CA] Gate de confusão DESATIVADO nesta época.")

        gate_skip_count = {'confidence': 0, 'knn': 0}

        # ============================================================
        # ETAPA 1: RELABEL POR CONFIANÇA ALTA (com gate)
        # ============================================================
        confidence_relabel_count = 0

        if self.enable_confidence_relabel:
            if self.debug:
                print(f"\n[VCNC-KNN-CA] ETAPA 1: Relabel por confiança > {self.relabel_confidence_threshold}")

            for box in all_box_data:
                if (box['pred_score'] > self.relabel_confidence_threshold and
                        box['pred_label'] != box['gt_label']):

                    # GATE: se par (gt, pred) está marcado como ruidoso, pula relabel
                    if gate_active and (box['gt_label'], box['pred_label']) in confused_pairs:
                        gate_skip_count['confidence'] += 1
                        continue

                    new_label = box['pred_label']
                    self._apply_relabel(datasets, box['sub_idx'], box['data_idx'],
                                        box['gt_idx'], new_label)
                    box['gt_label'] = new_label
                    box['score_gt'] = box['scores'][new_label].item()
                    box['relabeled_by'] = 'confidence'
                    box['was_relabeled'] = True
                    confidence_relabel_count += 1

            if self.debug:
                print(f"[VCNC-KNN-CA] Relabelados por confiança: {confidence_relabel_count} "
                      f"({confidence_relabel_count/len(all_box_data)*100:.2f}%)")
                if gate_active:
                    print(f"[VCNC-KNN-CA] Bloqueados pelo gate (Etapa 1): {gate_skip_count['confidence']}")

        # ============================================================
        # ETAPA 2: RELABEL POR KNN VISUAL (com gate)
        # ============================================================
        knn_relabel_count = 0

        if self.enable_knn_relabel:
            if self.debug:
                print(f"\n[VCNC-KNN-CA] ETAPA 2: Relabel por KNN visual (K={self.knn_k})")

            scores_by_class_updated = defaultdict(list)
            for box in all_box_data:
                scores_by_class_updated[box['gt_label']].append(box['score_gt'])
            gmm_dict = self._fit_gmm_per_class(scores_by_class_updated)

            for box in all_box_data:
                box['p_noise'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict)

            criteria = self._get_current_criteria(epoch)
            if self.debug:
                print(f"[VCNC-KNN-CA] Fase: {criteria['phase']}")

            c_anchor_gmm = criteria['anchor_gmm_threshold']
            c_anchor_pred = criteria['anchor_pred_agreement']
            c_anchor_conf = criteria['anchor_confidence']
            c_suspect_gmm = criteria['suspect_gmm_threshold']

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
                print(f"[VCNC-KNN-CA] Âncoras: {n_anchors} ({n_anchors/len(all_box_data)*100:.2f}%)")
                print(f"[VCNC-KNN-CA] Suspeitos: {n_suspects} ({n_suspects/len(all_box_data)*100:.2f}%)")

            if n_anchors < self.knn_min_anchors:
                if self.debug:
                    print(f"[VCNC-KNN-CA] Poucas âncoras, pulando relabeling por KNN")
            else:
                embeddings = np.array([box['embedding'] for box in all_box_data])
                knn_index, embeddings_norm = self._build_knn_index(embeddings)

                knn_stats = {
                    'processed': 0, 'few_anchor_neighbors': 0, 'no_consensus': 0,
                    'same_label': 0, 'gate_blocked': 0, 'relabeled': 0
                }

                for suspect_idx in suspect_indices:
                    box = all_box_data[suspect_idx]
                    knn_stats['processed'] += 1

                    neighbor_indices, neighbor_similarities = self._find_knn(
                        knn_index, embeddings_norm, suspect_idx, self.knn_k
                    )

                    suggested_label, confidence, n_anchor_neighbors = self._knn_vote(
                        neighbor_indices, neighbor_similarities, all_box_data,
                        anchor_mask, distance_weighted=self.knn_distance_weighted
                    )

                    if n_anchor_neighbors < self.knn_min_anchors:
                        knn_stats['few_anchor_neighbors'] += 1
                        continue
                    if confidence < self.knn_consensus_threshold:
                        knn_stats['no_consensus'] += 1
                        continue
                    if suggested_label == box['gt_label']:
                        knn_stats['same_label'] += 1
                        continue

                    # GATE: bloqueia relabel para par ruidoso
                    if gate_active and (box['gt_label'], suggested_label) in confused_pairs:
                        knn_stats['gate_blocked'] += 1
                        gate_skip_count['knn'] += 1
                        continue

                    self._apply_relabel(datasets, box['sub_idx'], box['data_idx'],
                                        box['gt_idx'], suggested_label)
                    box['gt_label'] = suggested_label
                    box['score_gt'] = box['scores'][suggested_label].item()
                    box['relabeled_by'] = 'knn'
                    box['was_relabeled'] = True
                    knn_relabel_count += 1
                    knn_stats['relabeled'] += 1

                if self.debug:
                    print(f"[VCNC-KNN-CA] Estatísticas KNN:")
                    print(f"[VCNC-KNN-CA]   - Processados: {knn_stats['processed']}")
                    print(f"[VCNC-KNN-CA]   - Poucas âncoras vizinhas: {knn_stats['few_anchor_neighbors']}")
                    print(f"[VCNC-KNN-CA]   - Sem consenso: {knn_stats['no_consensus']}")
                    print(f"[VCNC-KNN-CA]   - Mesmo label: {knn_stats['same_label']}")
                    print(f"[VCNC-KNN-CA]   - Bloqueados pelo gate: {knn_stats['gate_blocked']}")
                    print(f"[VCNC-KNN-CA]   - Relabelados: {knn_stats['relabeled']}")

            if self.debug:
                print(f"[VCNC-KNN-CA] Relabelados por KNN: {knn_relabel_count} "
                      f"({knn_relabel_count/len(all_box_data)*100:.2f}%)")

        # ============================================================
        # ETAPA 3: SPATIAL REFINEMENT (sem mudança)
        # ============================================================
        spatial_relabel_count = 0
        spatial_stats = {'total_boxes': 0, 'high_contamination': 0, 'refinements_applied': 0}

        if self.enable_spatial_refinement:
            if self.debug:
                print(f"\n[VCNC-KNN-CA] ETAPA 3: Spatial Refinement")
            for img_path, img_boxes in boxes_by_image.items():
                if len(img_boxes) < 2:
                    continue
                boxes_tensor = torch.stack([b['gt_bbox'] for b in img_boxes])
                pred_labels = torch.tensor([b['pred_label'] for b in img_boxes])
                pred_scores = torch.stack([b['scores'] for b in img_boxes])
                refined_labels, stats = spatial_aware_relabeling(
                    boxes_tensor, pred_labels, pred_scores,
                    difficulty_threshold=self.spatial_difficulty_threshold
                )
                spatial_stats['total_boxes'] += stats['total_boxes']
                spatial_stats['high_contamination'] += stats['high_contamination']
                spatial_stats['refinements_applied'] += stats['refinements_applied']

                for idx, box in enumerate(img_boxes):
                    if refined_labels[idx] != pred_labels[idx]:
                        if box['relabeled_by'] is None:
                            new_label = refined_labels[idx].item()
                            self._apply_relabel(datasets, box['sub_idx'], box['data_idx'],
                                                box['gt_idx'], new_label)
                            box['gt_label'] = new_label
                            box['score_gt'] = box['scores'][new_label].item()
                            box['relabeled_by'] = 'spatial'
                            box['was_relabeled'] = True
                            spatial_relabel_count += 1
            if self.debug:
                print(f"[VCNC-KNN-CA] Relabelados por spatial: {spatial_relabel_count}")

        # ============================================================
        # ETAPA 4 + 5: Filtragem (sem mudança)
        # ============================================================
        selective_filter_count = 0
        gmm_filter_count = 0

        if self.enable_selective_filtering:
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
                self._apply_ignore_flag(datasets, box['sub_idx'], box['data_idx'], box['gt_idx'])
                box['filtered'] = True
                selective_filter_count += 1

        if self.enable_gmm_filter:
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
                    self._apply_ignore_flag(datasets, box['sub_idx'], box['data_idx'], box['gt_idx'])
                    box['filtered'] = True
                    gmm_filter_count += 1

        # ============================================================
        # RESUMO
        # ============================================================
        if self.debug:
            total_relabels = confidence_relabel_count + knn_relabel_count + spatial_relabel_count
            total_filtered = selective_filter_count + gmm_filter_count
            total_skipped = gate_skip_count['confidence'] + gate_skip_count['knn']
            print(f"\n[VCNC-KNN-CA] ===== Resumo Época {epoch} =====")
            print(f"[VCNC-KNN-CA] Total de boxes: {len(all_box_data)}")
            print(f"[VCNC-KNN-CA] Relabel confiança: {confidence_relabel_count}")
            print(f"[VCNC-KNN-CA] Relabel KNN: {knn_relabel_count}")
            print(f"[VCNC-KNN-CA] Relabel spatial: {spatial_relabel_count}")
            print(f"[VCNC-KNN-CA] Total relabels: {total_relabels} "
                  f"({total_relabels/len(all_box_data)*100:.2f}%)")
            print(f"[VCNC-KNN-CA] Bloqueados pelo gate: {total_skipped} "
                  f"(conf={gate_skip_count['confidence']}, knn={gate_skip_count['knn']})")
            print(f"[VCNC-KNN-CA] Total filtrados: {total_filtered}")
            print(f"[VCNC-KNN-CA] ==========================================\n")

    # --------- helpers de IO de dataset ---------

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
                print(f"[VCNC-KNN-CA] Erro relabel: {e}")

    def _apply_ignore_flag(self, datasets, sub_idx, data_idx, gt_idx):
        try:
            instance = datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]
            instance['ignore_flag'] = 1
        except Exception as e:
            if self.debug:
                print(f"[VCNC-KNN-CA] Erro ignore_flag: {e}")
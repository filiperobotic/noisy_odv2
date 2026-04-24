"""
VCNC Diagnóstico Detalhado de Clusters

Este hook NÃO modifica o dataset. Apenas coleta e imprime estatísticas
detalhadas sobre cada cluster para entendermos:

1. Quantos clusters são classificados como simétrico/assimétrico/limpo
2. Por que clusters limpos podem ser confundidos com assimétricos
3. Diferenças entre cenários simétrico e assimétrico

Informações por cluster:
- % de âncoras
- Distribuição de GT labels
- Distribuição de top-2 nas âncoras
- Classificação do cluster
- Exemplos de amostras
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


@HOOKS.register_module()
class VCNCClusterDiagnosticHook(Hook):
    """
    Hook de diagnóstico - NÃO modifica o dataset.
    Apenas analisa e imprime estatísticas detalhadas dos clusters.
    """
    
    def __init__(self,
                 warmup_epochs: int = 1,
                 num_classes: int = 20,
                 
                 # === ANÁLISE POR CLUSTER ===
                 anchor_ratio_threshold: float = 0.7,
                 top2_concentration_threshold: float = 0.3,
                 min_cluster_size: int = 5,
                 
                 # === CLUSTERING ===
                 n_clusters: int = 30,
                 use_softmax_as_embedding: bool = True,
                 
                 # === CRITÉRIOS DE ÂNCORA ===
                 anchor_gmm_threshold: float = 0.4,
                 anchor_pred_agreement: float = 0.6,
                 anchor_confidence: float = 0.7,
                 suspect_gmm_threshold: float = 0.5,
                 
                 # === CONFIGURAÇÃO ===
                 iou_assigner: float = 0.5,
                 diagnostic_epochs: list = [2, 4, 7, 12],  # Épocas para diagnóstico detalhado
                 output_dir: str = './vcnc_cluster_diagnostic',
                 
                 # Nomes das classes (VOC)
                 class_names: list = None):
        
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        
        self.anchor_ratio_threshold = anchor_ratio_threshold
        self.top2_concentration_threshold = top2_concentration_threshold
        self.min_cluster_size = min_cluster_size
        
        self.n_clusters = n_clusters
        self.use_softmax_as_embedding = use_softmax_as_embedding
        
        self.anchor_gmm_threshold = anchor_gmm_threshold
        self.anchor_pred_agreement = anchor_pred_agreement
        self.anchor_confidence = anchor_confidence
        self.suspect_gmm_threshold = suspect_gmm_threshold
        
        self.iou_assigner = iou_assigner
        self.diagnostic_epochs = diagnostic_epochs
        self.output_dir = output_dir
        
        # Nomes das classes VOC
        self.class_names = class_names or [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_class_name(self, label):
        if 0 <= label < len(self.class_names):
            return f"{label}:{self.class_names[label]}"
        return str(label)
    
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
    
    def before_train_epoch(self, runner):
        epoch = runner.epoch + 1
        
        if epoch <= self.warmup_epochs:
            return
        
        # Só faz diagnóstico detalhado em épocas específicas
        if epoch not in self.diagnostic_epochs:
            return
        
        print(f"\n{'='*80}")
        print(f"[DIAG] ========== DIAGNÓSTICO DETALHADO - Época {epoch} ==========")
        print(f"{'='*80}")
        
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
                            
                            gt_label = gt_labels[gt_idx].item()
                            score_gt = best_scores[gt_label].item()
                            
                            top2_scores, top2_indices = best_scores.topk(2)
                            pred_label = top2_indices[0].item()
                            pred_score = top2_scores[0].item()
                            top2_label = top2_indices[1].item()
                            margin = (top2_scores[0] - top2_scores[1]).item()
                            
                            if self.use_softmax_as_embedding:
                                embedding = best_scores.cpu().numpy()
                            else:
                                embedding = assigned_scores.mean(dim=0).cpu().numpy()
                            
                            box_data = {
                                'gt_label': gt_label,
                                'score_gt': score_gt,
                                'pred_label': pred_label,
                                'pred_score': pred_score,
                                'top2_label': top2_label,
                                'margin': margin,
                                'embedding': embedding,
                            }
                            all_box_data.append(box_data)
                            scores_by_class[gt_label].append(score_gt)
                    except Exception as e:
                        continue
        
        if len(all_box_data) == 0:
            print("[DIAG] Nenhum box coletado!")
            return
        
        print(f"[DIAG] Total de boxes coletados: {len(all_box_data)}")
        
        # ============================================================
        # GMM E ÂNCORAS
        # ============================================================
        gmm_dict = self._fit_gmm_per_class(scores_by_class)
        
        for box in all_box_data:
            box['p_noise'] = self._get_p_noise(box['score_gt'], box['gt_label'], gmm_dict)
            
            is_clean = box['p_noise'] < self.anchor_gmm_threshold
            model_agrees = box['score_gt'] > self.anchor_pred_agreement
            high_confidence = box['pred_score'] > self.anchor_confidence
            box['is_anchor'] = is_clean and model_agrees and high_confidence
        
        total_anchors = sum(1 for b in all_box_data if b['is_anchor'])
        print(f"[DIAG] Total de âncoras: {total_anchors} ({total_anchors/len(all_box_data)*100:.1f}%)")
        
        # ============================================================
        # CLUSTERING
        # ============================================================
        embeddings = np.array([box['embedding'] for box in all_box_data])
        cluster_ids = self._cluster_embeddings(embeddings, self.n_clusters)
        
        for i, box in enumerate(all_box_data):
            box['cluster_id'] = cluster_ids[i]
        
        clusters = defaultdict(list)
        for box in all_box_data:
            clusters[box['cluster_id']].append(box)
        
        print(f"[DIAG] Número de clusters: {len(clusters)}")
        
        # ============================================================
        # ANÁLISE DETALHADA POR CLUSTER
        # ============================================================
        cluster_analysis = []
        
        type_counts = {'symmetric': 0, 'asymmetric': 0, 'clean': 0, 'skip': 0}
        
        print(f"\n[DIAG] {'='*70}")
        print(f"[DIAG] ANÁLISE POR CLUSTER")
        print(f"[DIAG] {'='*70}")
        
        for cluster_id in sorted(clusters.keys()):
            cluster_boxes = clusters[cluster_id]
            
            if len(cluster_boxes) < self.min_cluster_size:
                type_counts['skip'] += 1
                continue
            
            # Âncoras do cluster
            anchors = [b for b in cluster_boxes if b['is_anchor']]
            anchor_ratio = len(anchors) / len(cluster_boxes)
            
            # Distribuição de GT
            gt_counts = Counter([b['gt_label'] for b in cluster_boxes])
            dominant_gt, dominant_gt_count = gt_counts.most_common(1)[0]
            gt_purity = dominant_gt_count / len(cluster_boxes)
            
            # Distribuição de top-2 nas âncoras
            top2_counts = Counter()
            for anchor in anchors:
                if anchor['top2_label'] != anchor['gt_label']:
                    top2_counts[anchor['top2_label']] += 1
            
            top2_concentration = 0
            concentrated_class = None
            if len(anchors) > 0 and len(top2_counts) > 0:
                concentrated_class, conc_count = top2_counts.most_common(1)[0]
                top2_concentration = conc_count / len(anchors)
            
            # Margem média
            margins = [b['margin'] for b in cluster_boxes]
            avg_margin = np.mean(margins)
            
            # Margem média das âncoras
            anchor_margins = [b['margin'] for b in anchors] if anchors else [0]
            avg_anchor_margin = np.mean(anchor_margins)
            
            # Classificação do cluster
            if anchor_ratio < self.anchor_ratio_threshold:
                cluster_type = 'symmetric'
            elif top2_concentration >= self.top2_concentration_threshold:
                cluster_type = 'asymmetric'
            else:
                cluster_type = 'clean'
            
            type_counts[cluster_type] += 1
            
            # Salvar análise
            analysis = {
                'cluster_id': cluster_id,
                'size': len(cluster_boxes),
                'num_anchors': len(anchors),
                'anchor_ratio': anchor_ratio,
                'dominant_gt': dominant_gt,
                'dominant_gt_name': self._get_class_name(dominant_gt),
                'gt_purity': gt_purity,
                'gt_distribution': {self._get_class_name(k): v for k, v in gt_counts.most_common(5)},
                'top2_concentration': top2_concentration,
                'concentrated_class': concentrated_class,
                'concentrated_class_name': self._get_class_name(concentrated_class) if concentrated_class is not None else None,
                'top2_distribution': {self._get_class_name(k): v for k, v in top2_counts.most_common(5)},
                'avg_margin': avg_margin,
                'avg_anchor_margin': avg_anchor_margin,
                'cluster_type': cluster_type,
            }
            cluster_analysis.append(analysis)
            
            # Imprimir clusters interessantes (assimétricos ou com anomalias)
            if cluster_type == 'asymmetric' or (cluster_type == 'clean' and top2_concentration > 0.2):
                print(f"\n[DIAG] --- Cluster {cluster_id} ({cluster_type.upper()}) ---")
                print(f"[DIAG]   Tamanho: {len(cluster_boxes)}, Âncoras: {len(anchors)} ({anchor_ratio*100:.1f}%)")
                print(f"[DIAG]   GT dominante: {self._get_class_name(dominant_gt)} ({gt_purity*100:.1f}%)")
                print(f"[DIAG]   Distribuição GT: {dict(list(analysis['gt_distribution'].items())[:3])}")
                print(f"[DIAG]   Top-2 concentrado: {self._get_class_name(concentrated_class) if concentrated_class else 'N/A'} ({top2_concentration*100:.1f}%)")
                print(f"[DIAG]   Distribuição Top-2: {dict(list(analysis['top2_distribution'].items())[:3])}")
                print(f"[DIAG]   Margem média: {avg_margin:.3f}, Margem âncoras: {avg_anchor_margin:.3f}")
        
        # ============================================================
        # RESUMO GLOBAL
        # ============================================================
        print(f"\n[DIAG] {'='*70}")
        print(f"[DIAG] RESUMO GLOBAL")
        print(f"[DIAG] {'='*70}")
        print(f"[DIAG] Clusters por tipo:")
        print(f"[DIAG]   - Simétrico (âncoras < {self.anchor_ratio_threshold*100:.0f}%): {type_counts['symmetric']}")
        print(f"[DIAG]   - Assimétrico (top-2 > {self.top2_concentration_threshold*100:.0f}%): {type_counts['asymmetric']}")
        print(f"[DIAG]   - Limpo: {type_counts['clean']}")
        print(f"[DIAG]   - Skip (< {self.min_cluster_size} amostras): {type_counts['skip']}")
        
        # Análise por tipo de cluster
        for ctype in ['symmetric', 'asymmetric', 'clean']:
            type_clusters = [c for c in cluster_analysis if c['cluster_type'] == ctype]
            if len(type_clusters) == 0:
                continue
            
            avg_anchor_ratio = np.mean([c['anchor_ratio'] for c in type_clusters])
            avg_gt_purity = np.mean([c['gt_purity'] for c in type_clusters])
            avg_top2_conc = np.mean([c['top2_concentration'] for c in type_clusters])
            avg_margin = np.mean([c['avg_margin'] for c in type_clusters])
            
            print(f"\n[DIAG] {ctype.upper()} clusters ({len(type_clusters)}):")
            print(f"[DIAG]   - Média % âncoras: {avg_anchor_ratio*100:.1f}%")
            print(f"[DIAG]   - Média pureza GT: {avg_gt_purity*100:.1f}%")
            print(f"[DIAG]   - Média concentração top-2: {avg_top2_conc*100:.1f}%")
            print(f"[DIAG]   - Média margem: {avg_margin:.3f}")
        
        # ============================================================
        # ANÁLISE DE PARES (GT, Top-2) - GLOBAL
        # ============================================================
        print(f"\n[DIAG] {'='*70}")
        print(f"[DIAG] PARES (GT, Top-2) MAIS FREQUENTES NAS ÂNCORAS")
        print(f"[DIAG] {'='*70}")
        
        anchor_pairs = Counter()
        for box in all_box_data:
            if box['is_anchor'] and box['gt_label'] != box['top2_label']:
                pair = (box['gt_label'], box['top2_label'])
                anchor_pairs[pair] += 1
        
        for (gt, top2), count in anchor_pairs.most_common(10):
            print(f"[DIAG]   {self._get_class_name(gt)} -> {self._get_class_name(top2)}: {count}")
        
        # ============================================================
        # SALVAR ANÁLISE
        # ============================================================
        output_data = {
            'epoch': epoch,
            'total_boxes': len(all_box_data),
            'total_anchors': total_anchors,
            'anchor_ratio_global': total_anchors / len(all_box_data),
            'type_counts': type_counts,
            'cluster_analysis': cluster_analysis,
            'top_anchor_pairs': [
                {'gt': self._get_class_name(gt), 'top2': self._get_class_name(top2), 'count': count}
                for (gt, top2), count in anchor_pairs.most_common(20)
            ]
        }
        
        output_file = os.path.join(self.output_dir, f'cluster_diagnostic_epoch_{epoch}.json')
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n[DIAG] Análise salva em: {output_file}")
        print(f"[DIAG] {'='*80}\n")
        
        model.train()
    
    def _get_base_dataset(self, dataset):
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        return dataset
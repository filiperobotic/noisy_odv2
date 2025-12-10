from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import torch
from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.models.task_modules.assigners import MaxIoUAssigner
from collections import Counter
from torch.utils.data import DataLoader
import copy
import torch.nn.functional as F
from mmengine.structures import InstanceData
from collections import defaultdict
import numpy as np
from sklearn.mixture import GaussianMixture
from custom_configs.hooks.my_utils import weighted_knn

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

import cv2
import os
import random

# --- Safe import for Weights & Biases (wandb) ---
# Tenta importar wandb para logging, mas não falha se não estiver disponível
try:
    import wandb
except Exception:
    wandb = None


def tensor_to_numpy_img(tensor_img, mean=None, std=None, to_rgb=True):
    """
    Converte um tensor de imagem [3, H, W] (normalizado) em uma imagem numpy [H, W, 3] para visualização.
    Parâmetros:
      - tensor_img: tensor de imagem normalizada (C, H, W)
      - mean, std: normalização usada no pipeline
      - to_rgb: se True, converte RGB para BGR (OpenCV espera BGR)
    """
    img = tensor_img.detach().cpu().numpy().astype(np.float32)
    img = img.transpose(1, 2, 0)  # [C, H, W] → [H, W, C]
    if mean is None:
        mean = [123.675, 116.28, 103.53]
    if std is None:
        std = [58.395, 57.12, 57.375]
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img = img * std + mean  # Desfaz a normalização
    img = np.clip(img, 0, 255).astype(np.uint8)
    # OpenCV espera BGR; se o tensor está em RGB (to_rgb=True), converte para BGR
    if to_rgb:
        img = img[..., ::-1]
    return img.copy()


def desenhar_bboxesv3(img_path, tensor_img, instances, save_path='bbox_debug.jpg', color=(0, 255, 0), thickness=2, img_norm_cfg=None):
    """
    Desenha bounding boxes sobre a imagem, marcando ignore_flag com cor diferente.
    """
    # Reconstrói imagem a partir do tensor e normalização
    mean = (img_norm_cfg or {}).get('mean', [123.675, 116.28, 103.53])
    std = (img_norm_cfg or {}).get('std', [58.395, 57.12, 57.375])
    to_rgb = (img_norm_cfg or {}).get('to_rgb', True)
    img_np = tensor_to_numpy_img(tensor_img.cpu(), mean=mean, std=std, to_rgb=to_rgb)

    # Desenha cada instância (GT) na imagem
    for inst in instances:
        bbox = inst['bboxes'].tensor[0].cpu().numpy().astype(int)
        label = inst['labels'].cpu().item()
        ignore_flag = inst.get('ignore_flag', 0)
        color = (0, 0, 255) if ignore_flag else (0, 255, 0)

        cv2.rectangle(img_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(img_np, f"Cls:{label}", (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        

    # Cria pasta de destino se necessário
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Salva imagem anotada
    cv2.imwrite(save_path, img_np)
    print(f"[INFO] Imagem salva em: {save_path}")

    

#def desenhar_bboxesv3_pred(img_path, tensor_img, gt_instances, pred_instances,associated_preds, save_path='bbox_debug.jpg', color=(0, 255, 0), thickness=2):
def desenhar_bboxesv3_pred(tensor_img, gt_instances, pred_instances, iou, associated_preds, gt_pred, save_path='bbox_debug.jpg', color=(0, 255, 0), thickness=2, img_norm_cfg=None):
    """
    Desenha GTs e predições associadas na imagem, mostrando scores, rótulos e IOU.
    """
    mean = (img_norm_cfg or {}).get('mean', [123.675, 116.28, 103.53])
    std = (img_norm_cfg or {}).get('std', [58.395, 57.12, 57.375])
    to_rgb = (img_norm_cfg or {}).get('to_rgb', True)
    img_np = tensor_to_numpy_img(tensor_img.cpu(), mean=mean, std=std, to_rgb=to_rgb)

    # Desenha GTs
    for inst in gt_instances:
        bbox = inst['bboxes'].tensor[0].cpu().numpy().astype(int)
        label = inst['labels'].cpu().item()
        ignore_flag = inst.get('ignore_flag', 0)
        color = (0, 0, 255) if ignore_flag else (0, 255, 0)
        cv2.rectangle(img_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(img_np, f"Cls:{label}", (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    # Desenha predições associadas (em azul/vermelho)
    for bbox, label, score, inter, logits in zip(pred_instances['priors'][associated_preds], pred_instances['labels'][associated_preds], pred_instances['scores'][associated_preds],iou[associated_preds], pred_instances['logits'][associated_preds]):
        bbox = bbox.cpu().numpy().astype(int)
        label = int(label)
        score = float(score)
        # Calcula softmax dos logits para obter scores por classe
        myscores = torch.softmax(logits ,dim=-1)
        myscores_pred =  myscores.max(dim=0).values.item()
        myscore_gt = myscores[gt_pred].item()
        mylabel_pred_temp = myscores.argmax(dim=0).item()
        # Desenha bbox da predição e texto com informações relevantes
        cv2.rectangle(img_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.putText(img_np, f"Pred:{mylabel_pred_temp}/{label}-{gt_pred} ({myscores_pred:.2f}/{myscore_gt:.2f}/{inter:.2f})", (bbox[0], bbox[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        

    # Cria pasta de destino se necessário
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Salva imagem anotada
    cv2.imwrite(save_path, img_np)
    print(f"[INFO] Imagem salva em: {save_path}")

def calculate_gmm(c_class_scores, n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42  ):
    """
    Ajusta um GMM aos scores e retorna a probabilidade de pertencer ao cluster de menor média (baixa confiança).
    """
    gmm = GaussianMixture(n_components=n_components, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
    gmm.fit(c_class_scores)
    # Identifica o cluster de menor média (baixa confiança)
    low_confidence_component = np.argmin(gmm.means_)
    low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]
    return low_confidence_scores

def draw_score_histogram(c_class_scores, low_confidence_indices, save_path, epoch, c, threshold ):
    """
    Plota histograma dos scores de uma classe, marcando o ponto de corte do GMM.
    """
    # Determina ponto de corte (máximo dos scores de baixa confiança)
    if len(low_confidence_indices) > 0:
        score_cutoff = np.max(c_class_scores[low_confidence_indices])
    else:
        print(f"Aviso: Nenhuma amostra com confiança > {threshold} encontrada.")
        score_cutoff = np.min(c_class_scores)  # valor alternativo seguro
    plt.figure()
    plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
    # Linha vertical no corte do GMM
    plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
    plt.title(f'Classe {c} - Histograma de scores (época {epoch} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
    plt.xlabel('Score')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.tight_layout()
    # Cria pasta de destino se necessário
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


# -----------------------------
# Lightweight GNN components (no external deps)
# -----------------------------
class _EdgeMLP(nn.Module):
    """
    Pequena MLP para processar atributos de aresta em grafos.
    """
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )
    def forward(self, e):
        return self.net(e)

class _GATLayer(nn.Module):
    """
    Implementação mínima de uma camada tipo GAT (Graph Attention Network).
    Permite usar atributos de aresta no cálculo da atenção.
    - edge_index: [2, E] com índices de origem e destino das arestas.
    - edge_attr:  [E, He] (opcional) atributos das arestas.
    """
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        # Se tiver atributos de aresta, projeta para o mesmo dim. dos nós
        self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim > 0 else None
        # Entrada da atenção: concatena origem, destino e atributo (se houver)
        att_in = out_dim * (2 if edge_dim == 0 else 3)
        self.att = nn.Linear(att_in, 1, bias=False)
        self.act = nn.ELU(inplace=True)

    def forward(self, x, edge_index, edge_attr=None):
        if x.numel() == 0:
            return x
        h = self.lin(x)  # [N, F]
        # Se não há arestas, retorna representação transformada
        if edge_index is None or edge_index.numel() == 0:
            return self.act(h)
        src, dst = edge_index  # [E], [E]
        # Calcula atenção: concatena origem, destino e atributo (se houver)
        if edge_attr is not None and self.edge_proj is not None and edge_attr.numel() > 0:
            eproj = self.edge_proj(edge_attr)
            a_in = torch.cat([h[src], h[dst], eproj], dim=1)  # [E, 3F]
        else:
            a_in = torch.cat([h[src], h[dst]], dim=1)        # [E, 2F]
        if a_in.numel() == 0:
            return self.act(h)
        a = F.leaky_relu(self.att(a_in).squeeze(-1), negative_slope=0.2)
        # Softmax sobre as arestas que chegam em cada nó de destino
        num_nodes = h.size(0)
        a_exp = torch.exp(a - a.max())
        denom = torch.zeros(num_nodes, device=h.device).index_add(0, dst, a_exp)
        alpha = a_exp / (denom[dst] + 1e-9)  # [E]
        # Message passing: soma ponderada das mensagens dos vizinhos
        out = torch.zeros_like(h).index_add(0, dst, h[src] * alpha.unsqueeze(-1))
        return self.act(out)
    
class ConG(nn.Module):
    """
    Cabeça de grafo de contexto (inspirada no GCRN):
      - Nó: concatenação de softmax(RepG) (probs de classe, C) e spatial7 (7 features geométricas)
      - Grafo: totalmente conectado (sem autolaço)
      - Saída: logits por classe (C)
    """
    def __init__(self, num_classes: int, hidden: int = 128):
        super().__init__()
        in_dim = num_classes + 7
        self.lin_in  = nn.Linear(in_dim, hidden)
        self.gcn1    = _GATLayer(hidden, hidden, edge_dim=7)
        self.gcn2    = _GATLayer(hidden, hidden, edge_dim=7)
        self.lin_out = nn.Linear(hidden, num_classes)
        self.act     = nn.ReLU(inplace=True)

    def forward(self, x, edge_index, edge_attr=None):
        # x: [N, C+7], edge_index: [2, E], edge_attr: [E,2]
        h = self.act(self.lin_in(x))
        h = self.gcn1(h, edge_index, edge_attr)
        h = self.gcn2(h, edge_index, edge_attr)
        return self.lin_out(h)

class GraphNoiseNet(nn.Module):
    """
    Pequena rede de grafos para predizer p_noise por nó.
    - Entrada do nó: embedding da classe (rótulo efetivo) + projeção do vetor de probabilidade (softmax dos logits)
    - Entrada da aresta: atributos como similaridade, co-ocorrência, etc.
    - Saída: escalar p_noise para cada nó
    """
    def __init__(self, num_classes: int, cls_emb_dim: int = 32, prob_dim: int = 64, hidden: int = 128, edge_dim: int = 0):
        super().__init__()
        self.cls_emb = nn.Embedding(num_classes, cls_emb_dim)
        self.prob_proj = nn.Linear(num_classes, prob_dim)
        proj_in = cls_emb_dim + prob_dim
        self.proj = nn.Linear(proj_in, hidden)
        self.gnn1 = _GATLayer(hidden, hidden, edge_dim=edge_dim)
        self.gnn2 = _GATLayer(hidden, hidden, edge_dim=edge_dim)
        self.out = nn.Linear(hidden, 1)
        self.edge_dim = edge_dim

    def forward(self, labels, probs, edge_index, edge_attr=None):
        # labels: [N] (effective labels); probs: [N, C]
        z = torch.cat([self.cls_emb(labels), self.prob_proj(probs)], dim=1)
        h = self.proj(z)
        h = self.gnn1(h, edge_index, edge_attr)
        h = self.gnn2(h, edge_index, edge_attr)
        return self.out(h).squeeze(1)

@torch.no_grad()
def _adaptive_k(N: int, k_min: int = 1, k_max: int = 8, mode: str = 'sqrt') -> int:
    """
    Calcula o valor de k (nº de vizinhos) adaptativo para kNN, baseado em N.
    - mode: 'sqrt' usa raiz quadrada, 'log' usa log2.
    """
    if N <= 1:
        return 0
    if mode == 'log':
        kval = int(np.log2(N) + 1)
    else:  # 'sqrt' default
        kval = int(np.sqrt(N))
    kval = max(k_min, min(k_max, kval))
    return min(kval, N - 1)

# --- Helper: convert MMDet BaseBoxes or Tensor to Tensor ---
@torch.no_grad()
def _boxes_to_tensor(bboxes):
    """
    Converte uma lista de bboxes (MMDet BaseBoxes ou Tensor) em Tensor float [N,4].
    """
    if hasattr(bboxes, 'tensor'):
        return bboxes.tensor
    if isinstance(bboxes, torch.Tensor):
        return bboxes
    return torch.as_tensor(bboxes, dtype=torch.float32)

@torch.no_grad()
def _build_knn_graph(bboxes: torch.Tensor, k: int = 8, k_min: int = 1, k_max: int = 8, mode: str = 'sqrt'):
    """
    Constrói um grafo kNN adaptativo (por L2 do centro das boxes) para uma imagem.
    Retorna edge_index [2, E] e atributos por aresta [E, 4].
    """
    b = _boxes_to_tensor(bboxes)
    if b.numel() == 0:
        device = b.device
        return torch.empty(2, 0, dtype=torch.long, device=device), torch.empty(0, 4, device=device)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1).clamp(min=1e-3)
    h = (y2 - y1).clamp(min=1e-3)
    centers = torch.stack([cx, cy], dim=1)
    N = b.size(0)
    kk = _adaptive_k(N, k_min=k_min, k_max=k_max, mode=mode)
    if kk == 0:
        device = b.device
        return torch.empty(2, 0, dtype=torch.long, device=device), torch.empty(0, 4, device=device)
    # Calcula matriz de distâncias L2 entre centros
    D = torch.cdist(centers, centers, p=2)
    knn = D.topk(kk, largest=False).indices  # [N, kk]
    src = torch.arange(N, device=b.device).unsqueeze(1).expand(-1, kk).reshape(-1)
    dst = knn.reshape(-1)
    dx = (cx[src] - cx[dst]) / (w[dst] + 1e-3)
    dy = (cy[src] - cy[dst]) / (h[dst] + 1e-3)
    rw = torch.log(w[src] / w[dst])
    rh = torch.log(h[src] / h[dst])
    eattr = torch.stack([dx, dy, rw, rh], dim=1)
    return torch.stack([src, dst], dim=0), eattr

@torch.no_grad()
def _node_geom_feat(bboxes: torch.Tensor, scores: torch.Tensor):
    """
    Extrai features geométricas do nó: [cx, cy, log w, log h, max_score], normalizados pelo tamanho da imagem.
    """
    if bboxes.numel() == 0:
        return bboxes.new_zeros((0, 5))
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1).clamp(min=1)
    h = (y2 - y1).clamp(min=1)
    # Normalização simples pelo maior lado da imagem (aproximado pelo maior lado das boxes)
    scale = torch.max(torch.stack([x2.max() - x1.min(), y2.max() - y1.min()])) + 1e-6
    feat = torch.stack([cx / scale, cy / scale, torch.log(w), torch.log(h), scores], dim=1)
    return feat

@torch.no_grad()
def spatial7_from_xyxy(bbox_xyxy: torch.Tensor, W: int, H: int):
    """
    Extrai 7 features espaciais normalizadas de uma bbox xyxy, dado W,H da imagem.
    """
    x1, y1, x2, y2 = bbox_xyxy
    w = (x2 - x1).clamp(min=1.0)
    h = (y2 - y1).clamp(min=1.0)
    a = w * h
    return torch.tensor([
        w / max(W, 1),
        h / max(H, 1),
        a / max(W * H, 1),
        x1 / max(W, 1),
        y1 / max(H, 1),
        x2 / max(W, 1),
        y2 / max(H, 1),
    ], device=bbox_xyxy.device, dtype=torch.float32)

@torch.no_grad()
def fully_connected_edge_index(N: int, device):
    """
    Retorna edge_index de grafo totalmente conectado (sem autolaço) para N nós.
    """
    if N <= 1:
        return torch.empty(2, 0, dtype=torch.long, device=device)
    src = torch.arange(N, device=device).unsqueeze(1).expand(N, N).reshape(-1)
    dst = torch.arange(N, device=device).unsqueeze(0).expand(N, N).reshape(-1)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


# --- Helper: make a side-by-side bar chart for pr vs pr_ctx ---
def _make_prctx_figure(pr: torch.Tensor, pr_ctx: torch.Tensor, qc: torch.Tensor = None, labels: torch.Tensor = None, topk: int = 5, max_nodes: int = 3, class_names=None, title: str = "pr vs pr_ctx vs qc"):
    """
    Gera uma figura comparando pr (prob RepG), pr_ctx (prob contexto), e opcionalmente qc (ConG) lado a lado para os topk classes de cada nó.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        N, C = pr.shape
        K = int(min(topk, C))
        M = int(min(max_nodes, N))
        fig, axes = plt.subplots(M, 1, figsize=(6, 3*M), squeeze=False)
        for r in range(M):
            ax = axes[r, 0]
            p = pr[r].detach().cpu().numpy()
            q = pr_ctx[r].detach().cpu().numpy()
            qc_r = None
            if qc is not None:
                qc_r = qc[r].detach().cpu().numpy()

            # top-k por fonte
            idx_p  = np.argsort(p)[-K:]
            idx_q  = np.argsort(q)[-K:]
            idx_qc = np.argsort(qc_r)[-K:] if qc_r is not None else np.array([], dtype=int)

            # união dos índices
            union_idx = np.unique(np.concatenate([idx_p, idx_q, idx_qc])).tolist()

            # ordenar a união por uma pontuação combinada (máximo entre pr/pr_ctx/qc)
            comb_scores = []
            for j in union_idx:
                s_pr  = p[j]
                s_ctx = q[j]
                s_qc  = qc_r[j] if qc_r is not None else 0.0
                comb_scores.append(max(s_pr, s_ctx, s_qc))
            order = np.argsort(np.array(comb_scores))[::-1]
            union_idx = [union_idx[k] for k in order]

            U = len(union_idx)
            ind = np.arange(U)
            width = 0.23

            if qc_r is not None:
                ax.bar(ind - 0.25, p[union_idx], width=width, label='pr (RepG)')
                ax.bar(ind + 0.00, q[union_idx], width=width, label='pr_ctx (context)')
                ax.bar(ind + 0.25, qc_r[union_idx], width=width, label='qc (ConG)')
            else:
                ax.bar(ind - 0.15, p[union_idx], width=0.3, label='pr (RepG)')
                ax.bar(ind + 0.15, q[union_idx], width=0.3, label='pr_ctx (context)')

            # rótulos do eixo X
            if class_names is not None:
                xt = [str(class_names[j]) for j in union_idx]
            else:
                xt = [str(j) for j in union_idx]
            ax.set_xticks(ind)
            ax.set_xticklabels(xt, rotation=30, ha='right')
            gt = int(labels[r].item()) if labels is not None and labels.numel() > r else -1
            ax.set_title(f"node {r} | gt={gt}")
            ax.set_ylim(0, 1.0)
            ax.legend(loc='upper right', fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img
    except Exception:
        return None



# === Lightweight GNN noise filter hook ===
@HOOKS.register_module()
class MyHookGraphNoiseRelabelFilterGMM(Hook):
    @torch.no_grad()
    def _compute_co_probs(self, device):
        co_counts_cpu = self._cooc_counts.clone().float()
        row_sum = co_counts_cpu.sum(dim=1, keepdim=True).clamp_min(self.cooc_eps)
        co_probs = (co_counts_cpu / row_sum).to(device)
        co_probs.fill_diagonal_(1.0)
        return co_probs
    """
    Antes de cada época, constrói um grafo por imagem usando as detecções atuais,
    executa uma GNN leve para estimar p_ruido por GT e:
      (i) marca instâncias com p_ruido alto como ignore_flag=1, ou
      (ii) opcionalmente relabela para a maioria semântica dos vizinhos confiáveis.

    Parâmetros principais:
      - warmup_epochs: até esse ponto só treina estatísticas, sem filtrar.
      - thr_noise: limiar fixo de p_ruido (0..1) OU usa percentil por classe se use_percentile=True.
      - use_percentile: se True, usa percentil por classe (ex.: 80) como corte dinâmico.
      - k: número de vizinhos no k-NN.
      - do_relabel: se True, tenta relabelar ao invés de ignorar quando houver maioria forte.
      - relabel_min_agree: fração mínima (0..1) de vizinhos confiáveis que concordam.
      - num_classes: nº de classes do seu dataset.
    """
    def __init__(self,
                 warmup_epochs: int = 1,
                 thr_noise: float = 0.7,
                 use_percentile: bool = False,
                 percentile: float = 80.0,
                 k: int = 8,
                 do_relabel: bool = False,
                 relabel_min_agree: float = 0.6,
                 num_classes: int = 20,
                 reload_dataset: bool = True,
                 iou_assigner=0.5,
                 low_quality=False,
                 train_ratio: float = 1.0,
                 k_min: int = 1,
                 k_max: int = 8,
                 k_mode: str = 'sqrt',
                 gnn_lr: float = 1e-3, 
                 gnn_train_steps: int = 50, 
                 tau_gt: float = 0.3, 
                 neighbor_agree: float = 0.6,
                 trust_thr: float = 0.9,
                 corr_thr_low = 0.3,
                 # --- contexto com "gate" ---
                 ctx_conf_beta: float = 2.0,    # peso dos vizinhos por confiança (pmax^beta)
                 ctx_gate_gamma: float = 8.0,   # quão abrupto é o gate sigmoide
                 ctx_dist_sigma: float = 0.75,
                 cong_hidden: int = 128,
                 cong_lr: float = 1e-3,
                 cong_train_steps: int = 100,
                 cong_alpha: float = 0.5,
                 # --- W&B logging params ---
                 use_wandb: bool = True,
                 wandb_project: str = 'noisy-od',
                 wandb_run_name: str = '',
                 wandb_max_images: int = 8,
                 # --- prctx logging knobs ---
                 wandb_log_prctx: bool = True,
                 prctx_topk: int = 5,
                 prctx_max_nodes: int = 3,
                 relabel_thr_ctx: float = 0.7,
                 relabel_thr_high: float = 0.9,
                 # pesos e piso do componente visual no contexto
                 ctx_feat_weight: float = 0.6,
                 ctx_prob_weight: float = 0.4,
                 ctx_sim_min: float = 0.2,
                 filter_thr: float = 0.7,
                 ):
        self.warmup_epochs = warmup_epochs
        self.thr_noise = float(thr_noise)
        self.use_percentile = bool(use_percentile)
        self.percentile = float(percentile)
        self.k = int(k)
        self.do_relabel = bool(do_relabel)
        self.relabel_min_agree = float(relabel_min_agree)
        self.num_classes = int(num_classes)
        self.reload_dataset = reload_dataset
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        # será inicializada no primeiro uso para o device correto
        self._gnn = None
        self.train_ratio = float(train_ratio)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.k_mode = str(k_mode)
        self.gnn_lr = float(gnn_lr)
        self.gnn_train_steps = int(gnn_train_steps)
        self.tau_gt = float(tau_gt)
        self.neighbor_agree = float(neighbor_agree)
        self._opt = None  # optimizer da GNN (lazily created)
        self._trained = False  # flag para indicar se a GNN já foi treinada nesta execução
        self.trust_thr = float(trust_thr)
        self.corr_thr_low = corr_thr_low  # limiar para p_ruido só pela co-ocorrência (baixo relacionamento com classes presentes)   
        # knobs de gate de contexto
        self.ctx_conf_beta = float(ctx_conf_beta)
        self.ctx_gate_gamma = float(ctx_gate_gamma)
        self.ctx_dist_sigma = float(ctx_dist_sigma)
        # co-ocorrência (CPU): inicia suavizada para evitar zeros
        self._cooc_counts = torch.ones(self.num_classes, self.num_classes, dtype=torch.float32)
        self.cooc_eps = 1e-3          # suavização mínima ao normalizar (em vez de 1e-6 mencionado)
        self.reset_cooc_each_epoch = True  # se True, zera (para 1s) a cada época
        # ConG (context graph head)
        self.cong_hidden = int(cong_hidden)
        self.cong_lr = float(cong_lr)
        self.cong_train_steps = int(cong_train_steps)
        self.cong_alpha = float(cong_alpha)  # força do reweight: lw = 1 - alpha * p_noise
        self._cong = None
        self._opt_cong = None
        # W&B logging
        self.use_wandb = bool(use_wandb)
        self.wandb_project = str(wandb_project)
        self.wandb_run_name = str(wandb_run_name)
        self.wandb_max_images = int(wandb_max_images)
        self._wandb_ready = False
        self._wandb_img_budget = 0
        self._wandb_imgs = []  # lista de wandb.Image para log único por época
        # --- Phase-1 relabel memory (per epoch, per image) ---
        self._phase1_relabels = {}

        self.wandb_log_prctx = bool(wandb_log_prctx)
        self.prctx_topk = int(prctx_topk)
        self.prctx_max_nodes = int(prctx_max_nodes)

        # W&B extra knobs
        self.wandb_log_if_any_kld = True  # loga imagem se houver qualquer KLD acima do corte, mesmo vetado

        self.relabel_thr_ctx = float(relabel_thr_ctx)
        self.relabel_thr_high = float(relabel_thr_high)
        self.ctx_feat_weight = float(ctx_feat_weight)
        self.ctx_prob_weight = float(ctx_prob_weight)
        self.ctx_sim_min     = float(ctx_sim_min)
        self.filter_thr = float(filter_thr)

    def _ensure_gnn(self, device):
        if self._gnn is None:
            #self._gnn = GraphNoiseNet(num_classes=self.num_classes, cls_emb_dim=32, prob_dim=64, hidden=128, edge_dim=3).to(device)
            self._gnn = GraphNoiseNet(num_classes=self.num_classes, cls_emb_dim=32, prob_dim=64, hidden=128, edge_dim=2).to(device)
        else:
            self._gnn.to(device)
        self._gnn.eval()  # usamos apenas para scoring no hook
        if self._opt is None:
            self._opt = optim.Adam(self._gnn.parameters(), lr=self.gnn_lr)


    # --- Helper for effective labels from logits and semantic graph ---
    @torch.no_grad()
    def _effective_labels_from_logits(self, node_logits_t: torch.Tensor, gt_labels_t: torch.Tensor, trust_thr: float):
        probs = node_logits_t.softmax(dim=-1)
        pmax, argmax = probs.max(dim=-1)
        eff = gt_labels_t.clone()
        eff[pmax >= trust_thr] = argmax[pmax >= trust_thr]
        return eff, probs, pmax, argmax

    @torch.no_grad()
    def _build_semantic_graph(self, probs: torch.Tensor, eff_labels: torch.Tensor, cooc_probs: torch.Tensor,
                              bboxes_xyxy: torch.Tensor, img_w: int, img_h: int, k: int = 4,
                              features: torch.Tensor = None, feat_weight: float = 0.5, prob_weight: float = 0.5):
        """
        Build a semantic kNN graph using cosine similarity on class-prob vectors and visual features, plus spatial proximity.
        Returns edge_index [2,E] and edge_attr [E,7] = [sim_prob, f_sim, dx, dy, rw, rh, co].
        - sim_prob : similarity between probability distributions (cosine, [0,1])
        - f_sim    : similarity between features (cosine, [0,1])
        - dx,dy    : normalized offsets (src→dst) scaled by dst size
        - rw,rh    : log size ratios (src/dst)
        - co       : co-occurrence prior p(li→lj)
        """
        device = probs.device
        N = probs.size(0)
        if N <= 1:
            return torch.empty(2,0, dtype=torch.long, device=device), torch.empty(0,7, device=device)

        # --- Similaridade entre distribuições de probabilidade (softmax dos logits) ---
        Pn = F.normalize(probs, p=2, dim=-1)
        S_prob = torch.mm(Pn, Pn.t())
        S_prob = (S_prob + 1.0) * 0.5
        S_prob.fill_diagonal_(0.0)

        # --- Similaridade entre features visuais (softmax dos logits ou embeddings) ---
        if features is None:
            # Usa softmax dos logits como features padrão
            features = probs
        Fn = F.normalize(features, p=2, dim=-1)
        S_feat = torch.mm(Fn, Fn.t())
        S_feat = (S_feat + 1.0) * 0.5
        S_feat.fill_diagonal_(0.0)

        # combina do jeito que o chamador pediu (sem renormalizar aqui)
        S = prob_weight * S_prob + feat_weight * S_feat
        S.fill_diagonal_(0.0)

        # top-k by combined similarity
        kk = min(max(1, k), N-1)
        topk = S.topk(kk, dim=1).indices
        src = torch.arange(N, device=device).unsqueeze(1).expand(-1, kk).reshape(-1)
        dst = topk.reshape(-1)

        # spatial attributes from bboxes
        b = bboxes_xyxy
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w  = (x2 - x1).clamp(min=1e-3)
        h  = (y2 - y1).clamp(min=1e-3)

        dx = (cx[src] - cx[dst]) / (w[dst] + 1e-3)
        dy = (cy[src] - cy[dst]) / (h[dst] + 1e-3)
        rw = torch.log(w[src] / w[dst])
        rh = torch.log(h[src] / h[dst])

        sim_prob = S_prob[src, dst].unsqueeze(1)
        f_sim = S_feat[src, dst].unsqueeze(1)
        co  = cooc_probs[eff_labels[src], eff_labels[dst]].unsqueeze(1)

        edge_attr = torch.cat([
            sim_prob,
            f_sim,
            dx.unsqueeze(1), dy.unsqueeze(1),
            rw.unsqueeze(1), rh.unsqueeze(1),
            co
        ], dim=1)
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index, edge_attr

    @torch.no_grad()
    def _make_pseudo_targets(self, node_labels_t, edge_index, node_logits_t, gt_idx_t, cut_prob: float):
        """Constroi y_noise (0/1) por nó.
        Critérios:
            A) p_gt < tau_gt
            B) Fração de vizinhos confiáveis (p_gt >= tau_gt) que concordam com GT é < neighbor_agree
        Marca 1 se (A) OU (B).
        """
        if node_logits_t.numel() == 0:
            return torch.zeros(0, device=node_logits_t.device)
        probs = node_logits_t.softmax(dim=-1)
        idx = torch.arange(probs.size(0), device=probs.device)
        p_gt = probs[idx, gt_idx_t]
        critA = (p_gt < self.tau_gt)

        src, dst = edge_index
        agree_cnt = torch.zeros_like(p_gt)
        total_cnt = torch.zeros_like(p_gt)
        neigh_pgt = p_gt[src]
        neigh_lbl = gt_idx_t[src]
        is_conf = neigh_pgt >= self.tau_gt
        same = (neigh_lbl == gt_idx_t[dst]) & is_conf
        agree_cnt.index_add_(0, dst, same.float())
        total_cnt.index_add_(0, dst, is_conf.float())
        frac_agree = torch.zeros_like(p_gt)
        mask = total_cnt > 0
        frac_agree[mask] = agree_cnt[mask] / total_cnt[mask]
        critB = torch.zeros_like(critA)
        critB[mask] = (frac_agree[mask] < self.neighbor_agree)
        y = (critA | critB).float()
        return y
    
            

    @torch.no_grad()
    def _pnoise_from_cooc(self, eff_labels_t: torch.Tensor, co_probs_dev: torch.Tensor, thr: float):
        """Retorna p_noise em [0,1] por nó baseado APENAS na matriz de co-ocorrência.
        Para cada nó i (classe li), considera as classes dos demais nós j na imagem e
        pega best = max_j co_probs[li, lj]. Se best >= thr → p_noise=0; caso contrário
        p_noise = (thr - best)/thr (quanto menor a co-ocorrência, maior o p_noise).
        """
        N = eff_labels_t.numel()
        if N <= 1:
            return torch.zeros(N, device=eff_labels_t.device)
        li = eff_labels_t                            # [N]
        lj = eff_labels_t                            # [N]
        # matriz cooc para todos pares (i,j)
        C = co_probs_dev[li][:, lj]                  # [N,N]
        # ignora diagonal (relacionar consigo mesmo não conta)
        C = C.masked_fill(torch.eye(N, dtype=torch.bool, device=C.device), 0.0)
        best, _ = C.max(dim=1)                       # [N]
        p = (thr - best).clamp(min=0.0) / max(thr, 1e-6)
        return p
    
    def _ensure_cong(self, device):
        if self._cong is None:
            self._cong = ConG(num_classes=self.num_classes, hidden=self.cong_hidden).to(device)
        else:
            self._cong.to(device)
        if self._opt_cong is None:
            self._opt_cong = optim.AdamW(self._cong.parameters(), lr=self.cong_lr, weight_decay=0.0)

    def before_train_epoch(self, runner):
        if (runner.epoch + 1) <= 1:
            return

        # Reinicia lista de imagens do W&B a cada época
        self._wandb_imgs = []
        dataloader = runner.train_loop.dataloader
        dataset = dataloader.dataset
        # opcional: reiniciar co-ocorrência por época
        if self.reset_cooc_each_epoch:
            self._cooc_counts = torch.ones(self.num_classes, self.num_classes, dtype=torch.float32)

        # --- W&B: init per epoch (lazy) and reset epoch buffers ---
        self._wandb_img_budget = self.wandb_max_images
        self._wandb_imgs = []
        # --- Clear Phase-1 relabel memory for the new epoch ---
        self._phase1_relabels = {}
        if self.use_wandb and (wandb is not None):
            if not self._wandb_ready:
                try:
                    wandb.init(project=self.wandb_project or 'noisy-od',
                               name=self.wandb_run_name or f'run-{os.path.basename(runner.work_dir)}',
                               dir=runner.work_dir,
                               reinit=True)
                    self._wandb_ready = True
                except Exception as _:
                    self._wandb_ready = False

        reload_dataset = self.reload_dataset

        if reload_dataset:
            runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
            runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

            runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
            runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        if not hasattr(dataset, 'datasets'):
            raise ValueError("Esperado um ConcatDataset.")
        datasets = dataset.datasets

        # mapa rápido: img_path -> (sub_dataset_idx, data_idx)
        dataset_img_map = {di['img_path']: (sidx, didx)
                            for sidx, subds in enumerate(datasets)
                            if hasattr(subds, 'data_list')
                            for didx, di in enumerate(subds.data_list)}

        # ----------------- GNN TRAIN/FILTER PHASE (unified pass) -----------------
        # === PHASE 1: build co-occurrence (presence-based) and apply only high-conf relabel ===
        for data_batch in dataloader:
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0 else next(runner.model.parameters()).device)
            self._ensure_cong(device)
            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue
                pred_instances = preds[i].pred_instances.to(device)
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')
                gt = ds.gt_instances.to(device)
                assigner = MaxIoUAssigner(pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner,
                                          min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality)
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue
                logits = pred_instances.logits
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                sub_idx, d_idx = dataset_img_map[img_path]
                subds = datasets[sub_idx]
                valid_instance_indices = [idx for idx, inst in enumerate(subds.data_list[d_idx]['instances']) if inst['ignore_flag'] == 0]

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []
                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    
                    choice = assoc[j]
                    node_labels.append(int(gt.labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)
                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)
                vis_feats = node_logits_t  # usar logits (pré-softmax) como descritor visual
                pmax, pred_cls = pr.max(dim=1)

                # Relabel only by high confidence (no context yet)
                for relabel_idx in range(node_labels_t.shape[0]):
                    if float(pmax[relabel_idx]) >= self.relabel_thr_high:
                        new_lab = int(pred_cls[relabel_idx].item())
                        gt_local_idx = node_img_local_to_valid[relabel_idx]
                        if 0 <= gt_local_idx < len(valid_instance_indices):
                            valid_idx = valid_instance_indices[gt_local_idx]
                            old_lab = int(node_labels_t[relabel_idx].item())
                            # Atualiza tensores/listas deste batch
                            node_labels_t[relabel_idx] = new_lab
                            inst = subds.data_list[d_idx]['instances'][valid_idx]
                            updated = False
                            for key in ['labels','label','bbox_label']:
                                if key in inst:
                                    inst[key] = new_lab
                                    updated = True
                            if not updated:
                                inst['labels'] = new_lab
                            # Memoriza para overlay na Fase 3 (por imagem, indexado pelo GT local)
                            if img_path not in self._phase1_relabels:
                                self._phase1_relabels[img_path] = {}
                            # chave: índice local de GT usado para construir os nós (node_img_local_to_valid)
                            self._phase1_relabels[img_path][int(gt_local_idx)] = int(old_lab)
                        else:
                            if hasattr(runner, 'logger'):
                                runner.logger.warning(f"[Phase1] Skip relabel write: gt_local_idx={gt_local_idx} out of range for {os.path.basename(img_path)}")

                # Presence-based co-occurrence update (after relabel-high)
                eff_labels_t = node_labels_t.clone()
                if eff_labels_t.numel() > 1:
                    uniq = torch.unique(eff_labels_t.detach().to(torch.long))
                    if uniq.numel() > 1:
                        ui = uniq.unsqueeze(1).expand(-1, uniq.numel())
                        uj = uniq.unsqueeze(0).expand(uniq.numel(), -1)
                        mask = (ui != uj)
                        pairs_i = ui[mask].reshape(-1).to('cpu', dtype=torch.long)
                        pairs_j = uj[mask].reshape(-1).to('cpu', dtype=torch.long)
                        self._cooc_counts.index_put_((pairs_i, pairs_j), torch.ones_like(pairs_i, dtype=torch.float32), accumulate=True)

        # Freeze epoch prior
        # epoch_device = runner.model.device
        epoch_device = next(runner.model.parameters()).device
        co_probs_epoch = self._compute_co_probs(epoch_device)

        # === PHASE 2: Train ConG using the fixed co_probs_epoch ===
        steps = 0
        for data_batch in dataloader:
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0 else next(runner.model.parameters()).device)
            self._ensure_cong(device)
            batch_loss = None
            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue
                pred_instances = preds[i].pred_instances.to(device)
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')
                gt = ds.gt_instances.to(device)
                assigner = MaxIoUAssigner(pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner,
                                          min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality)
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue
                logits = pred_instances.logits
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []
                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    choice = assoc[j]
                    node_labels.append(int(gt.labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)
                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)
                # usar logits (pré-softmax) como descritor visual
                vis_feats = node_logits_t

                # Build context inputs
                H_img = inputs.shape[-2]
                W_img = inputs.shape[-1]
                gt_boxes_xyxy = gt.bboxes.tensor
                sel_gt_xyxy = gt_boxes_xyxy[node_img_local_to_valid]
                sp = torch.stack([spatial7_from_xyxy(sel_gt_xyxy[k], W_img, H_img) for k in range(sel_gt_xyxy.size(0))], dim=0)
                # === Consistência de dimensões entre pr/vis_feats/labels/sp/boxes ===
                N_pr   = pr.size(0)
                N_feat = vis_feats.size(0)
                N_lab  = node_labels_t.size(0)
                N_box  = sel_gt_xyxy.size(0)
                N_sp   = sp.size(0)
                min_n = min(N_pr, N_feat, N_lab, N_box, N_sp)
                if not (N_pr == N_feat == N_lab == N_box == N_sp):
                    pr = pr[:min_n]
                    vis_feats = vis_feats[:min_n]
                    node_logits_t = node_logits_t[:min_n]
                    node_labels_t = node_labels_t[:min_n]
                    sel_gt_xyxy = sel_gt_xyxy[:min_n]
                    sp = sp[:min_n]
                Nn = pr.size(0)
                if Nn <= 1:
                    pr_ctx = pr
                else:
                    probs = pr
                    # confidences (neighbors j)
                    pmax = probs.max(dim=-1).values
                    w_conf = (pmax.clamp_min(1e-6) ** self.ctx_conf_beta)
                    # cosine similarity nas probabilidades
                    Pn = F.normalize(pr, p=2, dim=-1)
                    S_prob = torch.mm(Pn, Pn.t())
                    S_prob = (S_prob + 1.0) * 0.5
                    # cosine similarity visual (logits)
                    Fn = F.normalize(vis_feats, p=2, dim=-1)
                    S_feat = torch.mm(Fn, Fn.t())
                    S_feat = (S_feat + 1.0) * 0.5
                    # combinação com pesos configuráveis (com ajuste defensivo de tamanho)
                    if S_prob.shape != S_feat.shape:
                        min_n = min(S_prob.shape[0], S_feat.shape[0])
                        S_prob = S_prob[:min_n, :min_n]
                        S_feat = S_feat[:min_n, :min_n]
                    S = self.ctx_prob_weight * S_prob + self.ctx_feat_weight * S_feat
                    S.fill_diagonal_(0.0)
                    # máscara para suprimir pares visualmente dissimilares
                    S_mask = (S_feat >= self.ctx_sim_min).float()
                    # spatial Gaussian kernel using dest box sizes
                    b = sel_gt_xyxy
                    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
                    cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                    w  = (x2-x1).clamp(min=1e-3); h=(y2-y1).clamp(min=1e-3)
                    DX = (cx.unsqueeze(1) - cx.unsqueeze(0)).abs() / (w.unsqueeze(0) + 1e-3)
                    DY = (cy.unsqueeze(1) - cy.unsqueeze(0)).abs() / (h.unsqueeze(0) + 1e-3)
                    D  = torch.sqrt(DX**2 + DY**2)
                    sigma = self.ctx_dist_sigma
                    Ksp = torch.exp(-D / max(1e-6, sigma))
                    # co-occurrence attenuation between labels
                    li = node_labels_t
                    Cij = co_probs_epoch.to(device)[li][:, li]
                    # final weights W[i,j] (j contributes to i)
                    W = (S * Ksp * Cij) * S_mask
                    W = W * w_conf.unsqueeze(0)
                    W.fill_diagonal_(0.0)
                    den = W.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    pr_ctx = (W @ probs) / den
                pmax = pr.max(dim=-1).values
                alpha_conf = torch.sigmoid(self.ctx_gate_gamma * (pmax - self.trust_thr)).unsqueeze(1)
                alpha = alpha_conf
                pr_mix = alpha * pr + (1.0 - alpha) * pr_ctx
                x_cong = torch.cat([pr_mix, sp], dim=1)

                # Graph with fixed epoch prior
                co_probs = co_probs_epoch.to(device)
                edge_index, edge_attr = self._build_semantic_graph(
                    pr, node_labels_t, co_probs_epoch.to(device), sel_gt_xyxy, W_img, H_img, k=4,
                    features=vis_feats, feat_weight=self.ctx_feat_weight, prob_weight=self.ctx_prob_weight)

                # Train ConG
                self._cong.train()
                logits_cong = self._cong(x_cong, edge_index, edge_attr)
                loss_cong = F.cross_entropy(logits_cong, node_labels_t)
                if batch_loss is None:
                    batch_loss = loss_cong
                else:
                    batch_loss = batch_loss + loss_cong

            if batch_loss is not None:
                self._opt_cong.zero_grad()
                batch_loss.backward()
                self._opt_cong.step()
                steps += 1
                if steps >= self.cong_train_steps:
                    break
        if steps >= self.cong_train_steps and hasattr(runner, 'logger'):
            runner.logger.info(f"[ConG] Trained with fixed epoch co_probs: steps={steps}")

     
        for data_batch in dataloader:
            # forward do detector SEM grad (apenas extrair preds)
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0 else next(runner.model.parameters()).device)

            self._ensure_cong(device)
            co_probs = self._compute_co_probs(device)
            # ===== por-amostra do batch =====
            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue
                # garanta que TUDO está no mesmo device
                pred_instances = preds[i].pred_instances.to(device)
                # alguns heads expõem 'bboxes'; o assigner aceita 'priors' também
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')

                gt = ds.gt_instances.to(device)

                # referências locais (já no device)
                bboxes = gt.bboxes
                labels = gt.labels
                priors = pred_instances.priors
                logits = pred_instances.logits  # [Np, C]
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                # assign predições -> GTs
                assigner = MaxIoUAssigner(pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner, min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality)
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []

                # mapeamento de instâncias válidas no dataset (para escrita)
                sub_idx, d_idx = dataset_img_map[img_path]
                subds = datasets[sub_idx]
                valid_instance_indices = [idx for idx, inst in enumerate(subds.data_list[d_idx]['instances']) if inst['ignore_flag'] == 0]

                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    choice = assoc[j]
                    node_labels.append(int(labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)
                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)  # [N,C]
                vis_feats = node_logits_t
                # veto: se a classe predita == GT, não considerar como ruído
                pred_cls_t = pr.argmax(dim=1)                  # [N]
                veto_eq_t = (pred_cls_t == node_labels_t)      # [N] bool



                # rótulos efetivos para consultar co-ocorrência (usa pred quando muito confiante)
                eff_labels_t, _, pmax_t, argmax_t = self._effective_labels_from_logits(node_logits_t, node_labels_t, self.trust_thr)

                H_img = inputs.shape[-2]
                W_img = inputs.shape[-1]
                gt_boxes_xyxy = gt.bboxes.tensor
                sel_gt_xyxy = gt_boxes_xyxy[node_img_local_to_valid]
                sp_list = [spatial7_from_xyxy(sel_gt_xyxy[k], W_img, H_img) for k in range(sel_gt_xyxy.size(0))]
                if len(sp_list) == 0:
                    continue
                sp = torch.stack(sp_list, dim=0)  # [N,7]

                # --- contexto ponderado por confiança + gate por confiança e minoritarismo ---
                Nn = pr.size(0)
                # cosine similarities: probs e visual (logits)
                Pn = F.normalize(pr, p=2, dim=-1)
                S_prob = torch.mm(Pn, Pn.t()); S_prob = (S_prob + 1.0) * 0.5
                Fn = F.normalize(vis_feats, p=2, dim=-1)
                S_feat = torch.mm(Fn, Fn.t()); S_feat = (S_feat + 1.0) * 0.5

                # combinação com pesos configuráveis (com ajuste defensivo de tamanho)
                if S_prob.shape != S_feat.shape:
                    min_n = min(S_prob.shape[0], S_feat.shape[0])
                    S_prob = S_prob[:min_n, :min_n]
                    S_feat = S_feat[:min_n, :min_n]
                S = self.ctx_prob_weight * S_prob + self.ctx_feat_weight * S_feat
                S.fill_diagonal_(0.0)
                # gate visual
                S_mask = (S_feat >= self.ctx_sim_min).float()

                # usar logits como features visuais adiante
                features = vis_feats
                if Nn <= 1:
                    pr_ctx = pr
                else:
                    probs = pr
                    pmax = probs.max(dim=-1).values
                    w_conf = (pmax.clamp_min(1e-6) ** self.ctx_conf_beta)
                    b = sel_gt_xyxy
                    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
                    cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                    w  = (x2-x1).clamp(min=1e-3); h=(y2-y1).clamp(min=1e-3)
                    DX = (cx.unsqueeze(1) - cx.unsqueeze(0)).abs() / (w.unsqueeze(0) + 1e-3)
                    DY = (cy.unsqueeze(1) - cy.unsqueeze(0)).abs() / (h.unsqueeze(0) + 1e-3)
                    D  = torch.sqrt(DX**2 + DY**2)
                    sigma = self.ctx_dist_sigma
                    Ksp = torch.exp(-D / max(1e-6, sigma))
                    li = eff_labels_t
                    Cij = co_probs[li][:, li] if (li.numel() == Nn) else torch.ones_like(S)
                    # Peso final: combinação da similaridade de features, probabilidade e co-ocorrência
                    W = (S * Ksp * Cij) * S_mask
                    W = W * w_conf.unsqueeze(0)
                    W.fill_diagonal_(0.0)
                    den = W.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    pr_ctx = (W @ probs) / den
                # gate: self vs contexto condicionado por confiança e minoritarismo
                probs = pr
                pmax = probs.max(dim=-1).values
                alpha_conf = torch.sigmoid(self.ctx_gate_gamma * (pmax - self.trust_thr)).unsqueeze(1)
                alpha = alpha_conf
                pr_mix = alpha * probs + (1.0 - alpha) * pr_ctx
                x_cong = torch.cat([pr_mix, sp], dim=1)
                # --- Build semantic graph with co-occurrence attributes and feature similarity ---
                # edge_attr agora inclui f_sim como segunda coluna
                edge_index, edge_attr = self._build_semantic_graph(
                    pr, eff_labels_t, co_probs, sel_gt_xyxy, W_img, H_img, k=4,
                    features=vis_feats, feat_weight=self.ctx_feat_weight, prob_weight=self.ctx_prob_weight
                )
                self._cong.eval()
                with torch.no_grad():
                    logits_cong = self._cong(x_cong, edge_index, edge_attr)
                    qc = logits_cong.softmax(dim=-1)  # [N,C]

                old_labels_before_agree = node_labels_t.clone()
                relabeled_pairs = []  # lista de tuples (local_idx, old_gt)
                # --- AGREEMENT RELABEL (modelo & contexto concordam) ---
                if qc.shape[0] == pr.shape[0] and pr.shape[0] > 1:
                    pmax_t = pr.max(dim=1).values               # confiança do modelo por nó
                    pred_cls_t = pr.argmax(dim=1)               # classe do modelo por nó
                    ctx_cls_t = qc.argmax(dim=1)                # classe de contexto por nó
                    agree = (pred_cls_t == ctx_cls_t) & (pmax_t >= self.relabel_thr_ctx)
                    agree_idx = torch.nonzero(agree, as_tuple=False).flatten()
                    for _li in agree_idx.tolist():
                        new_lab = int(pred_cls_t[_li].item())
                        old_lab = int(node_labels_t[_li].item())
                        if new_lab == old_lab:
                            # nada a fazer; não marca como relabel
                            continue
                        # atualiza o tensor de labels do batch
                        node_labels_t[_li] = new_lab
                        relabeled_pairs.append((_li, old_lab))
                        # persiste no dataset para próximas épocas
                        if _li < len(node_img_local_to_valid):
                            gt_idx = node_img_local_to_valid[_li]
                            if gt_idx < len(valid_instance_indices):
                                _valid_idx = valid_instance_indices[gt_idx]
                                inst = subds.data_list[d_idx]['instances'][_valid_idx]
                                updated = False
                                for key in ['bbox_label', 'label', 'labels']:
                                    if key in inst:
                                        inst[key] = new_lab
                                        updated = True
                                if not updated:
                                    inst['labels'] = new_lab
                    # Como alteramos node_labels_t, recomputamos eff_labels_t e veto_eq_t
                    eff_labels_t, _, pmax_t, argmax_t = self._effective_labels_from_logits(node_logits_t, node_labels_t, self.trust_thr)
                    veto_eq_t = (pred_cls_t == node_labels_t)
                    disagree_mask = (pred_cls_t != node_labels_t).detach().cpu().numpy().astype(bool)
                else:
                    if hasattr(runner, 'logger'):
                        runner.logger.warning(f"[ConG] Skip agreement relabel: "
                                              f"N={pr.shape[0]}, qc={qc.shape[0]} pr={pr.shape[0]} on {os.path.basename(img_path)}")

                eps = 1e-7
                p = pr.clamp_min(eps)
                q = qc.clamp_min(eps)
                # --- filtro contextual + co-ocorrência real ---
                kld = (p * (p.log() - q.log())).sum(dim=1)  # [N]
                kld_np = kld.detach().cpu().numpy()
                # usa KLD absoluto como p_noise (sem normalização por imagem)
                p_noise = kld
                p_noise_np = kld_np

                # --- filtro auxiliar de co-ocorrência real ---
                # para cada nó i (classe li), verifica se existe alguma classe w presente na imagem tal que co(li,w) >= corr_thr_low
                Nn = eff_labels_t.numel()
                if Nn > 1:
                    li = eff_labels_t
                    lj = eff_labels_t
                    C = co_probs[li][:, lj]  # [N,N]
                    C = C.masked_fill(torch.eye(Nn, dtype=torch.bool, device=C.device), 0.0)
                    best, _ = C.max(dim=1)   # [N]
                    best_np = best.detach().cpu().numpy()
                    low_corr_mask = (best < float(self.corr_thr_low))  # True quando baixa correlação com classes presentes
                else:
                    low_corr_mask = torch.zeros_like(kld, dtype=torch.bool)
                low_corr_np = low_corr_mask.detach().cpu().numpy().astype(bool)

                # define corte da parte contextual (KLD normalizado)
                if self.use_percentile and p_noise_np.size > 0:
                    cut = float(np.percentile(p_noise_np, self.percentile))
                else:
                    cut = float(self.thr_noise)

                # --- veto para minoritários plausíveis ---
                # veto_np = np.zeros_like(low_corr_np, dtype=bool)
                # if Nn > 0:
                #     # maioria por frequência na imagem (com rótulos efetivos)
                #     counts = torch.bincount(eff_labels_t, minlength=self.num_classes).float()
                #     majority = int(counts.argmax().item())
                #     # co-ocorrência com a classe majoritária
                #     co_with_major = co_probs[eff_labels_t, majority]  # [N]
                #     co_with_major_np = co_with_major.detach().cpu().numpy()
                #     freq_i = (counts[eff_labels_t] / max(float(eff_labels_t.numel()), 1.0)).detach().cpu().numpy()
                #     minority_np = (freq_i < float(self.rho_min))
                #     strong_cooc_np = (co_with_major_np >= float(self.corr_thr_pos))
                #     veto_np = np.logical_and(minority_np, strong_cooc_np)

                # veto adicional: predição igual ao GT nunca é ruído
                veto_eq_np = veto_eq_t.detach().cpu().numpy().astype(bool)

                # --- W&B: loga alguns exemplos visuais ---
                if self.use_wandb and (wandb is not None):
                    # loga até `wandb_max_images` imagens por época com overlay
                    if self._wandb_img_budget > 0:
                        # Log only if at least one bbox had its label actually changed in this image
                        relabeled_pairs = relabeled_pairs if 'relabeled_pairs' in locals() else []
                        has_relabel = len(relabeled_pairs) > 0
                        should_log = has_relabel
                        try:
                            relabeled_pairs = relabeled_pairs if 'relabeled_pairs' in locals() else []
                            relabeled_set = set([int(i) for (i, _) in relabeled_pairs]) if 'relabeled_pairs' in locals() else set()
                            old_gt_map = {int(i): int(old) for (i, old) in relabeled_pairs} if 'relabeled_pairs' in locals() else {}
                            # --- Merge with Phase-1 relabels (pink) ---
                            phase1_map_for_img = self._phase1_relabels.get(img_path, {})  # {gt_local_idx -> old_lab}
                            relabel_high_set = set()
                            old_gt_map_phase1 = {}
                            # Mapear do índice de nó local (local_idx) para info da Fase-1 via gt_local_idx
                            for local_idx, gt_local_idx in enumerate(node_img_local_to_valid[:len(sel_gt_xyxy)]):
                                if int(gt_local_idx) in phase1_map_for_img:
                                    relabel_high_set.add(int(local_idx))
                                    old_gt_map_phase1[int(local_idx)] = int(phase1_map_for_img[int(gt_local_idx)])
                            # noisy se (KLD alto) E (baixa co-ocorrência real)
                            noisy_kld = (p_noise_np >= cut)
                            noisy_mask = np.logical_and(noisy_kld, low_corr_np)
                            # aplica vetos: minoria plausível e pred==GT
                            # if 'veto_np' in locals() and len(veto_np) == len(noisy_mask):
                            #     noisy_mask = np.logical_and(noisy_mask, np.logical_not(veto_np))
                            if 'veto_eq_np' in locals() and len(veto_eq_np) == len(noisy_mask):
                                noisy_mask = np.logical_and(noisy_mask, np.logical_not(veto_eq_np))
                            if should_log:
                                # reconstrói imagem BGR para desenho usando img_norm_cfg real
                                # Carrega a imagem original em BGR e redimensiona para o img_shape atual do batch
                                # (mantém cores corretas para desenho via OpenCV)
                                meta = getattr(data_samples[i], 'metainfo', {})
                                img_shape = tuple(meta.get('img_shape', (inputs.shape[-2], inputs.shape[-1])))  # (H, W, 3)
                                H_v, W_v = int(img_shape[0]), int(img_shape[1])
                                img_np = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
                                if img_np is None:
                                    # fallback: tenta usar o tensor (pode ter cores trocadas)
                                    norm_cfg = meta.get('img_norm_cfg', {})
                                    mean = norm_cfg.get('mean', [123.675, 116.28, 103.53])
                                    std = norm_cfg.get('std', [58.395, 57.12, 57.375])
                                    to_rgb = norm_cfg.get('to_rgb', True)
                                    img_np = tensor_to_numpy_img(inputs[i].cpu(), mean=mean, std=std, to_rgb=to_rgb)
                                else:
                                    img_np = cv2.resize(img_np, (W_v, H_v), interpolation=cv2.INTER_LINEAR)
                                    # aplica o mesmo flip da pipeline para alinhar as boxes
                                    if bool(meta.get('flip', False)):
                                        fd = str(meta.get('flip_direction', 'horizontal'))
                                        if 'diagonal' in fd:
                                            img_np = cv2.flip(img_np, -1)
                                        else:
                                            if 'horizontal' in fd:
                                                img_np = cv2.flip(img_np, 1)
                                            if 'vertical' in fd:
                                                img_np = cv2.flip(img_np, 0)
                                # desenha GTs associados com cor conforme noise
                                sel_gt_xyxy_np = sel_gt_xyxy.detach().cpu().numpy().astype(int)
                                for local_idx in range(min(len(node_img_local_to_valid), len(p_noise_np))):
                                    x1, y1, x2, y2 = sel_gt_xyxy_np[local_idx].tolist()
                                    is_noisy = bool(noisy_mask[local_idx])
                                    is_kld = bool(noisy_kld[local_idx])
                                    # prioridade de cor: relabel (AZUL/Fase-2) > relabel-high (ROSA/Fase-1) > noisy (VERMELHO) > KLD alto vetado (LARANJA) > discordância (AMARELO) > limpo (VERDE)
                                    if local_idx in relabeled_set:
                                        color = (255, 0, 0)            # AZUL (reanotado pela rede de grafos / Phase-2)
                                    elif local_idx in relabel_high_set:
                                        color = (255, 105, 180)        # ROSA (reanotado na Phase-1: alta confiança do modelo)
                                    elif 'disagree_mask' in locals() and local_idx < len(disagree_mask) and bool(disagree_mask[local_idx]):
                                        color = (0, 255, 255)          # AMARELO (pred != label atual)
                                    else:
                                        color = (0, 255, 0)            # VERDE

                                    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
                                    # obter classes GT/pred, KLD e pmax para overlay
                                    _gt_lab = int(node_labels[local_idx])
                                    _pred_lab = int(pr[local_idx].argmax().item())
                                    _kld = float(kld_np[local_idx])
                                    _pmax = float(pr[local_idx].max().item())
                                    _co = float(best_np[local_idx]) if 'best_np' in locals() and local_idx < len(best_np) else 0.0
                                    # Adiciona classe esperada pelo contexto (ctx) usando qc.argmax(dim=1)
                                    _ctx_lab = int(qc[local_idx].argmax().item()) if qc is not None and local_idx < qc.shape[0] else -1
                                    _v_eq = bool(veto_eq_np[local_idx]) if 'veto_eq_np' in locals() and local_idx < len(veto_eq_np) else False
                                    _old2 = old_gt_map.get(local_idx, None)                 # Phase-2 old label (ConG)
                                    _old1 = old_gt_map_phase1.get(local_idx, None)          # Phase-1 old label (high confidence)
                                    # Preferir mostrar o marcador correspondente à cor aplicada
                                    if local_idx in relabeled_set and _old2 is not None:
                                        rx_suffix = f"|R{_old2}"
                                    elif local_idx in relabel_high_set and _old1 is not None:
                                        rx_suffix = f"|R{_old1}"
                                    else:
                                        rx_suffix = ""
                                    _overlay_txt = f"gt{_gt_lab}|p{_pred_lab}|n{float(p_noise_np[local_idx]):.2f}|k{_kld:.2f}|s{_pmax:.2f}|c{_co:.2f}|ct{_ctx_lab}{rx_suffix}"
                                    cv2.putText(img_np,
                                                _overlay_txt,
                                                (max(0, x1), max(15, y1-4)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                # Converte BGR (OpenCV) → RGB antes de enviar ao W&B
                                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                                self._wandb_imgs.append(
                                    wandb.Image(img_rgb, caption=f"epoch={runner.epoch+1} | {os.path.basename(img_path)}")
                                )
                                self._wandb_img_budget -= 1
                                # Opcional: logar figura pr vs pr_ctx **apenas para nós noisy**
                                # if getattr(self, 'wandb_log_prctx', False):
                                #     noisy_idx = np.where(noisy_mask)[0].tolist()
                                #     if len(noisy_idx) > 0:
                                #         # respeita o limite de nós; seleciona os primeiros noisy
                                #         noisy_idx = noisy_idx[:max(1, getattr(self, 'prctx_max_nodes', 8))]
                                #         idx_t = torch.as_tensor(noisy_idx, device=pr.device, dtype=torch.long)
                                #         pr_sel = pr.index_select(0, idx_t)
                                #         pr_ctx_sel = pr_ctx.index_select(0, idx_t)
                                #         qc_sel = qc.index_select(0, idx_t) if qc is not None else None
                                #         labels_sel = node_labels_t.index_select(0, idx_t)

                                #         cls_names = None
                                #         try:
                                #             cls_names = data_samples[i].metainfo.get('classes', None)
                                #         except Exception:
                                #             cls_names = None
                                #         title = f"pr vs pr_ctx vs qc (noisy only) | {os.path.basename(img_path)}"
                                #         prctx_img = _make_prctx_figure(pr_sel, pr_ctx_sel, qc_sel, labels_sel,
                                #                                        topk=getattr(self, 'prctx_topk', 5),
                                #                                        max_nodes=len(noisy_idx),
                                #                                        class_names=cls_names,
                                #                                        title=title)
                                #         if prctx_img is not None:
                                #             self._wandb_imgs.append(wandb.Image(prctx_img, caption=f"pr_ctx noisy | epoch={runner.epoch+1} | {os.path.basename(img_path)}"))
                                # Extra: logar pr/pr_ctx/qc para casos com discordância modelo x label
                                if getattr(self, 'wandb_log_prctx', False) and 'disagree_mask' in locals():
                                    disagree_idx = np.where(disagree_mask)[0].tolist()
                                    if len(disagree_idx) > 0:
                                        disagree_idx = disagree_idx[:max(1, getattr(self, 'prctx_max_nodes', 8))]
                                        idx_t2 = torch.as_tensor(disagree_idx, device=pr.device, dtype=torch.long)
                                        pr_sel = pr.index_select(0, idx_t2)
                                        pr_ctx_sel = pr_ctx.index_select(0, idx_t2)
                                        qc_sel = qc.index_select(0, idx_t2) if qc is not None else None
                                        labels_sel = node_labels_t.index_select(0, idx_t2)

                                        cls_names = None
                                        try:
                                            cls_names = data_samples[i].metainfo.get('classes', None)
                                        except Exception:
                                            cls_names = None

                                        title2 = f"pr vs pr_ctx vs qc (pred!=label) | {os.path.basename(img_path)}"
                                        prctx_img2 = _make_prctx_figure(pr_sel, pr_ctx_sel, qc_sel, labels_sel,
                                                                        topk=getattr(self, 'prctx_topk', 5),
                                                                        max_nodes=len(disagree_idx),
                                                                        class_names=cls_names,
                                                                        title=title2)
                                        if prctx_img2 is not None:
                                            self._wandb_imgs.append(
                                                wandb.Image(prctx_img2, caption=f"pr_ctx disagree | epoch={runner.epoch+1} | {os.path.basename(img_path)}")
                                            )
                        except Exception as e:
                            if hasattr(runner, 'logger'):
                                runner.logger.warning(f"[W&B] Falha ao montar/registrar imagem: {e}")

                # aplica REWEIGHT (ignore_flag comentado)
                L = min(len(node_img_local_to_valid), len(p_noise_np))
                for local_idx in range(L):
                    gt_idx = node_img_local_to_valid[local_idx]
                    valid_idx = valid_instance_indices[gt_idx] if gt_idx < len(valid_instance_indices) else None
                    if valid_idx is None:
                        continue
                    if (runner.epoch + 1) <= self.warmup_epochs:
                        continue
                    is_lowcorr = bool(low_corr_np[local_idx])
                    # is_veto = bool(veto_np[local_idx]) if 'veto_np' in locals() and local_idx < len(veto_np) else False
                    is_veto_eq = bool(veto_eq_np[local_idx]) if 'veto_eq_np' in locals() and local_idx < len(veto_eq_np) else False
                    #if (float(p_noise_np[local_idx]) >= cut) and is_lowcorr and (not is_veto) and (not is_veto_eq):
                    if (float(p_noise_np[local_idx]) >= cut) and is_lowcorr and (not is_veto_eq):
                        lw = max(0.2, 1.0 - self.cong_alpha * float(p_noise_np[local_idx]))
                        # subds.data_list[d_idx]['instances'][valid_idx]['loss_weight'] = lw
                        pass
                        # subds.data_list[d_idx]['instances'][valid_idx]['ignore_flag'] = 1  # ignorar apenas quando também houver baixa co-ocorrência
                    
                    # pred_label = int(node_labels_t[local_idx].item())
                    # pred_labelv2 = int(eff_labels_t[local_idx].item())
                    # import pdb; pdb.set_trace()
                    # subds.data_list[d_idx]['instances'][valid_idx]['bbox_label'] = pred_label

          

        # Salvar matriz de co-ocorrência como imagem de heatmap usando matplotlib
        try:
            # 1) Salvar heatmap da co-ocorrência
            cooc_dir = os.path.join(runner.work_dir, 'debug_cooc')
            os.makedirs(cooc_dir, exist_ok=True)
            cooc_path = os.path.join(cooc_dir, f'cooc_matrix_epoch{runner.epoch + 1}.png')
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patheffects as _pe
                plt.figure(figsize=(6, 5))
                _co = self._cooc_counts.clone().float()
                _row = _co.sum(dim=1, keepdim=True).clamp_min(1e-6)
                co_probs = (_co / _row)
                co_probs.fill_diagonal_(1.0)
                _vis = co_probs.numpy()
                im = plt.imshow(co_probs.cpu(), cmap='viridis')
                plt.title(f"Co-occurrence Matrix - Epoch {runner.epoch + 1}")
                plt.colorbar(im)
                plt.xlabel("j (given i→j)")
                plt.ylabel("i")

                # Ajusta os eixos para valores inteiros
                n_classes = co_probs.shape[0]
                plt.xticks(range(n_classes), range(n_classes))
                plt.yticks(range(n_classes), range(n_classes))
                plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

                # === NEW: escreve o número de ocorrências (contagem bruta) em cada célula ===
                _co_np = _co.detach().cpu().numpy()
                for i_tick in range(n_classes):
                    for j_tick in range(n_classes):
                        cnt = int(_co_np[i_tick, j_tick])
                        if cnt <= 0:
                            continue
                        # escolhe cor do texto conforme intensidade para contraste
                        txt_color = 'white' if _vis[i_tick, j_tick] > 0.2 else 'white'
                        plt.text(j_tick, i_tick, str(cnt), ha='center', va='center', fontsize=6,
                                 color=txt_color, path_effects=[_pe.withStroke(linewidth=1, foreground='black')])

                plt.tight_layout()
                plt.savefig(cooc_path)
                plt.close()
            except Exception as e_heat:
                if hasattr(runner, 'logger'):
                    runner.logger.warning(f"[Cooc] Falha ao salvar heatmap: {e_heat}")

            # 2) Forçar log no W&B (imagem, tabela e heatmap interativo)
            if getattr(self, 'use_wandb', False) and (wandb is not None) and getattr(self, '_wandb_ready', False):
                log_dict = {}
                if hasattr(self, '_wandb_imgs') and len(self._wandb_imgs) > 0:
                    log_dict['debug_imgs'] = self._wandb_imgs

                # Recalcula co_probs normalizado (CPU→numpy) para tabela/heatmap
                _co_counts = self._cooc_counts.clone().float()
                _row = _co_counts.sum(dim=1, keepdim=True).clamp_min(1e-6)
                _co_probs = (_co_counts / _row)
                _co_probs.fill_diagonal_(1.0)
                _co_probs_np = _co_probs.detach().cpu().numpy()

                # Tenta obter nomes de classes
                class_names = None
                try:
                    # tenta via datasets concatenados
                    while hasattr(dataset, 'dataset'):
                        dataset_ = dataset.dataset
                    # fallback: tenta pelo primeiro subdataset
                    if 'datasets' in dir(dataset) and len(dataset.datasets) > 0:
                        sub0 = dataset.datasets[0]
                        if hasattr(sub0, 'METAINFO') and isinstance(sub0.METAINFO, dict):
                            class_names = sub0.METAINFO.get('classes', None)
                except Exception:
                    class_names = None

                # Constrói Tabela (i,j,value) para o heatmap
                data_rows = []
                C = int(_co_probs_np.shape[0])
                for i in range(C):
                    for j in range(C):
                        name_i = class_names[i] if (class_names is not None and i < len(class_names)) else str(i)
                        name_j = class_names[j] if (class_names is not None and j < len(class_names)) else str(j)
                        data_rows.append([name_i, name_j, float(_co_probs_np[i, j])])
                cooc_table = wandb.Table(data=data_rows, columns=["i", "j", "value"])  # tabela completa

                # Heatmap interativo no W&B
                try:
                    cooc_heatmap = wandb.plot.heatmap(cooc_table, x="j", y="i", value="value",
                                                      title=f"Co-occurrence (row-normalized) - Epoch {runner.epoch + 1}")
                    log_dict['cooc_heatmap'] = cooc_heatmap
                except Exception as e_hm:
                    if hasattr(runner, 'logger'):
                        runner.logger.warning(f"[W&B] Falha ao criar heatmap: {e_hm}")

                # === NEW: também loga a matriz de contagens brutas ===
                _co_counts_np = _co_counts.detach().cpu().numpy()
                data_rows_cnt = []
                for i_idx in range(C):
                    for j_idx in range(C):
                        name_i = class_names[i_idx] if (class_names is not None and i_idx < len(class_names)) else str(i_idx)
                        name_j = class_names[j_idx] if (class_names is not None and j_idx < len(class_names)) else str(j_idx)
                        data_rows_cnt.append([name_i, name_j, int(_co_counts_np[i_idx, j_idx])])
                cooc_counts_table = wandb.Table(data=data_rows_cnt, columns=["i", "j", "count"])  # contagens
                try:
                    cooc_counts_heatmap = wandb.plot.heatmap(cooc_counts_table, x="j", y="i", value="count",
                                                             title=f"Co-occurrence COUNTS - Epoch {runner.epoch + 1}")
                    log_dict['cooc_counts_heatmap'] = cooc_counts_heatmap
                except Exception as e_hm2:
                    if hasattr(runner, 'logger'):
                        runner.logger.warning(f"[W&B] Falha ao criar heatmap de contagens: {e_hm2}")
                log_dict['cooc_counts_table'] = cooc_counts_table

                # Anexa imagem PNG gerada
                if os.path.exists(cooc_path):
                    log_dict['cooc_matrix'] = wandb.Image(cooc_path)
                # Anexa a tabela numérica
                log_dict['cooc_table'] = cooc_table

                if len(log_dict) > 0:
                    wandb.log(log_dict, commit=True)
                    # esvazia o buffer de imagens para próxima época
                    if 'debug_imgs' in log_dict:
                        self._wandb_imgs.clear()
                    if hasattr(runner, 'logger'):
                        runner.logger.info(f"[W&B] Imagens, cooc_table e cooc_heatmap logados na epoch {runner.epoch + 1}")
        except Exception as e_final:
            if hasattr(runner, 'logger'):
                runner.logger.warning(f"[W&B] Falha no log final da epoch: {e_final}")

        
        # === PHASE 3: GMM-based noise filtering (after relabeling) ===
        # Faz um processo em duas passagens:
        #   Passo 3A) varre toda a época e acumula scores por classe (p_gt)
        #   Passo 3B) ajusta um GMM por classe usando TODOS os scores acumulados e marca low-confidence como noisy

        # ---- knobs (se preferir, promova para __init__) ----
        gmm_filter_thr = 0.95    # prob. do componente de baixa confiança acima da qual marcamos como noisy
        gmm_min_samples = 8     # só aplica GMM para classes com pelo menos esse nº de amostras na época

        # ---- Passo 3A: acumula p_gt por classe em toda a época ----
        class_scores = [[] for _ in range(self.num_classes)]   # [ [float], ... ]
        class_ptrs   = [[] for _ in range(self.num_classes)]   # [ [(sub_idx, d_idx, valid_idx)], ... ]

        for data_batch in dataloader:
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0
                        else next(runner.model.parameters()).device)

            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue

                pred_instances = preds[i].pred_instances.to(device)
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')
                gt = ds.gt_instances.to(device)

                assigner = MaxIoUAssigner(
                    pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner,
                    min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality
                )
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue

                logits = pred_instances.logits
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []

                # mapeamento para escrita
                sub_idx, d_idx = dataset_img_map[img_path]
                subds = datasets[sub_idx]
                valid_instance_indices = [
                    idx for idx, inst in enumerate(subds.data_list[d_idx]['instances'])
                    if inst.get('ignore_flag', 0) == 0
                ]

                # escolher 1 predição por GT (a de maior score dentre as associadas)
                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    choice = assoc[j]

                    node_labels.append(int(gt.labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)

                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)

                idx = torch.arange(pr.size(0), device=pr.device)
                p_gt = pr[idx, node_labels_t].detach().cpu().numpy()
                node_labels_np = node_labels_t.detach().cpu().numpy()

                # Acumula por classe com ponteiro para onde escrever depois
                for loc_i, c in enumerate(node_labels_np):
                    gt_local_idx = node_img_local_to_valid[loc_i]
                    if 0 <= gt_local_idx < len(valid_instance_indices):
                        valid_idx = valid_instance_indices[gt_local_idx]
                        class_scores[c].append(float(p_gt[loc_i]))
                        class_ptrs[c].append((sub_idx, d_idx, valid_idx))
                    else:
                        if hasattr(runner, 'logger'):
                            runner.logger.debug(
                                f"[Phase3-GMM] Skip accumulate: gt_local_idx={gt_local_idx} "
                                f"out of range for {os.path.basename(img_path)}"
                            )

        # ---- Passo 3B: Ajuste GMM por classe e filtra ----
        for c in range(self.num_classes):
            scores_c = class_scores[c]
            if len(scores_c) < gmm_min_samples:
                continue
            scores_arr = np.array(scores_c, dtype=np.float32).reshape(-1, 1)
            try:
                low_confidence_scores = calculate_gmm(scores_arr)  # prob. do cluster de menor média
            except Exception as e:
                if hasattr(runner, 'logger'):
                    runner.logger.warning(f"[Phase3-GMM] GMM failed for class {c}: {e}")
                continue

            noisy_mask = (low_confidence_scores > gmm_filter_thr)
            noisy_indices = np.where(noisy_mask)[0]

            # (Opcional) salvar histograma por classe
            try:
                save_dir = os.path.join(runner.work_dir, 'phase3_gmm_hist')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'gmm_hist_epoch{runner.epoch+1}_class{c}.png')
                draw_score_histogram(scores_arr.flatten(), noisy_indices, save_path, runner.epoch+1, c, gmm_filter_thr)
            except Exception:
                pass

            # Marca ignore_flag=1 nos ponteiros correspondentes
            for k in noisy_indices.tolist():
                sub_idx, d_idx, valid_idx = class_ptrs[c][k]
                try:
                    # datasets[sub_idx].data_list[d_idx]['instances'][valid_idx]['ignore_flag'] = 1
                    inst = datasets[sub_idx].data_list[d_idx]['instances'][valid_idx]
                    pred_score = inst.get('pred', 0)  # se não existir, assume 0
                    if pred_score < self.filter_thr:
                        inst['ignore_flag'] = 1
                        if hasattr(runner, 'logger'):
                            runner.logger.debug(f"[Phase3-GMM] Filtered (low GMM + low pred < {self.filter_thr}) "
                                                f"for class {c}, img_idx={d_idx}, instance={valid_idx}")
                except Exception as e:
                    if hasattr(runner, 'logger'):
                        runner.logger.warning(
                            f"[Phase3-GMM] Failed to set ignore_flag: class {c}, idx={k}: {e}")

            if hasattr(runner, 'logger'):
                runner.logger.info(
                    f"[Phase3-GMM] Class {c}: filtered {len(noisy_indices)}/{len(scores_c)} as low-confidence (thr={gmm_filter_thr})")


# === Lightweight GNN noise filter hook ===
@HOOKS.register_module()
class MyHookGraphNoiseRelabelFilterGMMSanity(Hook):
    @torch.no_grad()
    def _compute_co_probs(self, device):
        co_counts_cpu = self._cooc_counts.clone().float()
        row_sum = co_counts_cpu.sum(dim=1, keepdim=True).clamp_min(self.cooc_eps)
        co_probs = (co_counts_cpu / row_sum).to(device)
        co_probs.fill_diagonal_(1.0)
        return co_probs
    """
    Antes de cada época, constrói um grafo por imagem usando as detecções atuais,
    executa uma GNN leve para estimar p_ruido por GT e:
      (i) marca instâncias com p_ruido alto como ignore_flag=1, ou
      (ii) opcionalmente relabela para a maioria semântica dos vizinhos confiáveis.

    Parâmetros principais:
      - warmup_epochs: até esse ponto só treina estatísticas, sem filtrar.
      - thr_noise: limiar fixo de p_ruido (0..1) OU usa percentil por classe se use_percentile=True.
      - use_percentile: se True, usa percentil por classe (ex.: 80) como corte dinâmico.
      - k: número de vizinhos no k-NN.
      - do_relabel: se True, tenta relabelar ao invés de ignorar quando houver maioria forte.
      - relabel_min_agree: fração mínima (0..1) de vizinhos confiáveis que concordam.
      - num_classes: nº de classes do seu dataset.
    """
    def __init__(self,
                 warmup_epochs: int = 1,
                 thr_noise: float = 0.7,
                 use_percentile: bool = False,
                 percentile: float = 80.0,
                 k: int = 8,
                 do_relabel: bool = False,
                 relabel_min_agree: float = 0.6,
                 num_classes: int = 20,
                 reload_dataset: bool = True,
                 iou_assigner=0.5,
                 low_quality=False,
                 train_ratio: float = 1.0,
                 k_min: int = 1,
                 k_max: int = 8,
                 k_mode: str = 'sqrt',
                 gnn_lr: float = 1e-3, 
                 gnn_train_steps: int = 50, 
                 tau_gt: float = 0.3, 
                 neighbor_agree: float = 0.6,
                 trust_thr: float = 0.9,
                 corr_thr_low = 0.3,
                 # --- contexto com "gate" ---
                 ctx_conf_beta: float = 2.0,    # peso dos vizinhos por confiança (pmax^beta)
                 ctx_gate_gamma: float = 8.0,   # quão abrupto é o gate sigmoide
                 ctx_dist_sigma: float = 0.75,
                 cong_hidden: int = 128,
                 cong_lr: float = 1e-3,
                 cong_train_steps: int = 100,
                 cong_alpha: float = 0.5,
                 # --- W&B logging params ---
                 use_wandb: bool = True,
                 wandb_project: str = 'noisy-od',
                 wandb_run_name: str = '',
                 wandb_max_images: int = 8,
                 # --- prctx logging knobs ---
                 wandb_log_prctx: bool = True,
                 prctx_topk: int = 5,
                 prctx_max_nodes: int = 3,
                 relabel_thr_ctx: float = 0.7,
                 relabel_thr_high: float = 0.9,
                 # pesos e piso do componente visual no contexto
                 ctx_feat_weight: float = 0.6,
                 ctx_prob_weight: float = 0.4,
                 ctx_sim_min: float = 0.2,
                 filter_thr: float = 0.7,
                 filter_confgmm: float = 0.95
                 ):
        self.warmup_epochs = warmup_epochs
        self.thr_noise = float(thr_noise)
        self.use_percentile = bool(use_percentile)
        self.percentile = float(percentile)
        self.k = int(k)
        self.do_relabel = bool(do_relabel)
        self.relabel_min_agree = float(relabel_min_agree)
        self.num_classes = int(num_classes)
        self.reload_dataset = reload_dataset
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        # será inicializada no primeiro uso para o device correto
        self._gnn = None
        self.train_ratio = float(train_ratio)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.k_mode = str(k_mode)
        self.gnn_lr = float(gnn_lr)
        self.gnn_train_steps = int(gnn_train_steps)
        self.tau_gt = float(tau_gt)
        self.neighbor_agree = float(neighbor_agree)
        self._opt = None  # optimizer da GNN (lazily created)
        self._trained = False  # flag para indicar se a GNN já foi treinada nesta execução
        self.trust_thr = float(trust_thr)
        self.corr_thr_low = corr_thr_low  # limiar para p_ruido só pela co-ocorrência (baixo relacionamento com classes presentes)   
        # knobs de gate de contexto
        self.ctx_conf_beta = float(ctx_conf_beta)
        self.ctx_gate_gamma = float(ctx_gate_gamma)
        self.ctx_dist_sigma = float(ctx_dist_sigma)
        # co-ocorrência (CPU): inicia suavizada para evitar zeros
        self._cooc_counts = torch.ones(self.num_classes, self.num_classes, dtype=torch.float32)
        self.cooc_eps = 1e-3          # suavização mínima ao normalizar (em vez de 1e-6 mencionado)
        self.reset_cooc_each_epoch = True  # se True, zera (para 1s) a cada época
        # ConG (context graph head)
        self.cong_hidden = int(cong_hidden)
        self.cong_lr = float(cong_lr)
        self.cong_train_steps = int(cong_train_steps)
        self.cong_alpha = float(cong_alpha)  # força do reweight: lw = 1 - alpha * p_noise
        self._cong = None
        self._opt_cong = None
        # W&B logging
        self.use_wandb = bool(use_wandb)
        self.wandb_project = str(wandb_project)
        self.wandb_run_name = str(wandb_run_name)
        self.wandb_max_images = int(wandb_max_images)
        self._wandb_ready = False
        self._wandb_img_budget = 0
        self._wandb_imgs = []  # lista de wandb.Image para log único por época
        # --- Phase-1 relabel memory (per epoch, per image) ---
        self._phase1_relabels = {}

        self.wandb_log_prctx = bool(wandb_log_prctx)
        self.prctx_topk = int(prctx_topk)
        self.prctx_max_nodes = int(prctx_max_nodes)

        # W&B extra knobs
        self.wandb_log_if_any_kld = True  # loga imagem se houver qualquer KLD acima do corte, mesmo vetado

        self.relabel_thr_ctx = float(relabel_thr_ctx)
        self.relabel_thr_high = float(relabel_thr_high)
        self.ctx_feat_weight = float(ctx_feat_weight)
        self.ctx_prob_weight = float(ctx_prob_weight)
        self.ctx_sim_min     = float(ctx_sim_min)
        self.filter_thr = float(filter_thr)
        self.filter_confgmm = float(filter_confgmm)
        self.selcand = "max" # max | iou
        self.numGMM = 4
        self.filter_type = "pred" # pred| logit | aps
        

    def _ensure_gnn(self, device):
        if self._gnn is None:
            #self._gnn = GraphNoiseNet(num_classes=self.num_classes, cls_emb_dim=32, prob_dim=64, hidden=128, edge_dim=3).to(device)
            self._gnn = GraphNoiseNet(num_classes=self.num_classes, cls_emb_dim=32, prob_dim=64, hidden=128, edge_dim=2).to(device)
        else:
            self._gnn.to(device)
        self._gnn.eval()  # usamos apenas para scoring no hook
        if self._opt is None:
            self._opt = optim.Adam(self._gnn.parameters(), lr=self.gnn_lr)


    # --- Helper for effective labels from logits and semantic graph ---
    @torch.no_grad()
    def _effective_labels_from_logits(self, node_logits_t: torch.Tensor, gt_labels_t: torch.Tensor, trust_thr: float):
        probs = node_logits_t.softmax(dim=-1)
        pmax, argmax = probs.max(dim=-1)
        eff = gt_labels_t.clone()
        eff[pmax >= trust_thr] = argmax[pmax >= trust_thr]
        return eff, probs, pmax, argmax

    @torch.no_grad()
    def _build_semantic_graph(self, probs: torch.Tensor, eff_labels: torch.Tensor, cooc_probs: torch.Tensor,
                              bboxes_xyxy: torch.Tensor, img_w: int, img_h: int, k: int = 4,
                              features: torch.Tensor = None, feat_weight: float = 0.5, prob_weight: float = 0.5):
        """
        Build a semantic kNN graph using cosine similarity on class-prob vectors and visual features, plus spatial proximity.
        Returns edge_index [2,E] and edge_attr [E,7] = [sim_prob, f_sim, dx, dy, rw, rh, co].
        - sim_prob : similarity between probability distributions (cosine, [0,1])
        - f_sim    : similarity between features (cosine, [0,1])
        - dx,dy    : normalized offsets (src→dst) scaled by dst size
        - rw,rh    : log size ratios (src/dst)
        - co       : co-occurrence prior p(li→lj)
        """
        device = probs.device
        N = probs.size(0)
        if N <= 1:
            return torch.empty(2,0, dtype=torch.long, device=device), torch.empty(0,7, device=device)

        # --- Similaridade entre distribuições de probabilidade (softmax dos logits) ---
        Pn = F.normalize(probs, p=2, dim=-1)
        S_prob = torch.mm(Pn, Pn.t())
        S_prob = (S_prob + 1.0) * 0.5
        S_prob.fill_diagonal_(0.0)

        # --- Similaridade entre features visuais (softmax dos logits ou embeddings) ---
        if features is None:
            # Usa softmax dos logits como features padrão
            features = probs
        Fn = F.normalize(features, p=2, dim=-1)
        S_feat = torch.mm(Fn, Fn.t())
        S_feat = (S_feat + 1.0) * 0.5
        S_feat.fill_diagonal_(0.0)

        # combina do jeito que o chamador pediu (sem renormalizar aqui)
        S = prob_weight * S_prob + feat_weight * S_feat
        S.fill_diagonal_(0.0)

        # top-k by combined similarity
        kk = min(max(1, k), N-1)
        topk = S.topk(kk, dim=1).indices
        src = torch.arange(N, device=device).unsqueeze(1).expand(-1, kk).reshape(-1)
        dst = topk.reshape(-1)

        # spatial attributes from bboxes
        b = bboxes_xyxy
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w  = (x2 - x1).clamp(min=1e-3)
        h  = (y2 - y1).clamp(min=1e-3)

        dx = (cx[src] - cx[dst]) / (w[dst] + 1e-3)
        dy = (cy[src] - cy[dst]) / (h[dst] + 1e-3)
        rw = torch.log(w[src] / w[dst])
        rh = torch.log(h[src] / h[dst])

        sim_prob = S_prob[src, dst].unsqueeze(1)
        f_sim = S_feat[src, dst].unsqueeze(1)
        co  = cooc_probs[eff_labels[src], eff_labels[dst]].unsqueeze(1)

        edge_attr = torch.cat([
            sim_prob,
            f_sim,
            dx.unsqueeze(1), dy.unsqueeze(1),
            rw.unsqueeze(1), rh.unsqueeze(1),
            co
        ], dim=1)
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index, edge_attr

    @torch.no_grad()
    def _make_pseudo_targets(self, node_labels_t, edge_index, node_logits_t, gt_idx_t, cut_prob: float):
        """Constroi y_noise (0/1) por nó.
        Critérios:
            A) p_gt < tau_gt
            B) Fração de vizinhos confiáveis (p_gt >= tau_gt) que concordam com GT é < neighbor_agree
        Marca 1 se (A) OU (B).
        """
        if node_logits_t.numel() == 0:
            return torch.zeros(0, device=node_logits_t.device)
        probs = node_logits_t.softmax(dim=-1)
        idx = torch.arange(probs.size(0), device=probs.device)
        p_gt = probs[idx, gt_idx_t]
        critA = (p_gt < self.tau_gt)

        src, dst = edge_index
        agree_cnt = torch.zeros_like(p_gt)
        total_cnt = torch.zeros_like(p_gt)
        neigh_pgt = p_gt[src]
        neigh_lbl = gt_idx_t[src]
        is_conf = neigh_pgt >= self.tau_gt
        same = (neigh_lbl == gt_idx_t[dst]) & is_conf
        agree_cnt.index_add_(0, dst, same.float())
        total_cnt.index_add_(0, dst, is_conf.float())
        frac_agree = torch.zeros_like(p_gt)
        mask = total_cnt > 0
        frac_agree[mask] = agree_cnt[mask] / total_cnt[mask]
        critB = torch.zeros_like(critA)
        critB[mask] = (frac_agree[mask] < self.neighbor_agree)
        y = (critA | critB).float()
        return y
    
            

    @torch.no_grad()
    def _pnoise_from_cooc(self, eff_labels_t: torch.Tensor, co_probs_dev: torch.Tensor, thr: float):
        """Retorna p_noise em [0,1] por nó baseado APENAS na matriz de co-ocorrência.
        Para cada nó i (classe li), considera as classes dos demais nós j na imagem e
        pega best = max_j co_probs[li, lj]. Se best >= thr → p_noise=0; caso contrário
        p_noise = (thr - best)/thr (quanto menor a co-ocorrência, maior o p_noise).
        """
        N = eff_labels_t.numel()
        if N <= 1:
            return torch.zeros(N, device=eff_labels_t.device)
        li = eff_labels_t                            # [N]
        lj = eff_labels_t                            # [N]
        # matriz cooc para todos pares (i,j)
        C = co_probs_dev[li][:, lj]                  # [N,N]
        # ignora diagonal (relacionar consigo mesmo não conta)
        C = C.masked_fill(torch.eye(N, dtype=torch.bool, device=C.device), 0.0)
        best, _ = C.max(dim=1)                       # [N]
        p = (thr - best).clamp(min=0.0) / max(thr, 1e-6)
        return p
    
    def _ensure_cong(self, device):
        if self._cong is None:
            self._cong = ConG(num_classes=self.num_classes, hidden=self.cong_hidden).to(device)
        else:
            self._cong.to(device)
        if self._opt_cong is None:
            self._opt_cong = optim.AdamW(self._cong.parameters(), lr=self.cong_lr, weight_decay=0.0)

    def before_train_epoch(self, runner):
        if (runner.epoch + 1) <= 0:
            return

        # Reinicia lista de imagens do W&B a cada época
        self._wandb_imgs = []
        dataloader = runner.train_loop.dataloader
        dataset = dataloader.dataset
        # opcional: reiniciar co-ocorrência por época
        if self.reset_cooc_each_epoch:
            self._cooc_counts = torch.ones(self.num_classes, self.num_classes, dtype=torch.float32)

        # --- W&B: init per epoch (lazy) and reset epoch buffers ---
        self._wandb_img_budget = self.wandb_max_images
        self._wandb_imgs = []
        # --- Clear Phase-1 relabel memory for the new epoch ---
        self._phase1_relabels = {}
        if self.use_wandb and (wandb is not None):
            if not self._wandb_ready:
                try:
                    wandb.init(project=self.wandb_project or 'noisy-od',
                               name=self.wandb_run_name or f'run-{os.path.basename(runner.work_dir)}',
                               dir=runner.work_dir,
                               reinit=True)
                    self._wandb_ready = True
                except Exception as _:
                    self._wandb_ready = False

        reload_dataset = self.reload_dataset

        if reload_dataset:
            runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
            runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

            runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
            runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        if not hasattr(dataset, 'datasets'):
            raise ValueError("Esperado um ConcatDataset.")
        datasets = dataset.datasets

        # mapa rápido: img_path -> (sub_dataset_idx, data_idx)
        dataset_img_map = {di['img_path']: (sidx, didx)
                            for sidx, subds in enumerate(datasets)
                            if hasattr(subds, 'data_list')
                            for didx, di in enumerate(subds.data_list)}

        # ----------------- GNN TRAIN/FILTER PHASE (unified pass) -----------------
        # === PHASE 1: build co-occurrence (presence-based) and apply only high-conf relabel ===
        for data_batch in dataloader:
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0 else next(runner.model.parameters()).device)
            self._ensure_cong(device)
            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue
                pred_instances = preds[i].pred_instances.to(device)
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')
                gt = ds.gt_instances.to(device)
                assigner = MaxIoUAssigner(pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner,
                                          min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality)
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue
                logits = pred_instances.logits
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                sub_idx, d_idx = dataset_img_map[img_path]
                subds = datasets[sub_idx]
                valid_instance_indices = [idx for idx, inst in enumerate(subds.data_list[d_idx]['instances']) if inst['ignore_flag'] == 0]

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []
                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    
                    choice = assoc[j]
                    node_labels.append(int(gt.labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)
                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)
                vis_feats = node_logits_t  # usar logits (pré-softmax) como descritor visual
                pmax, pred_cls = pr.max(dim=1)

                # Relabel only by high confidence (no context yet)
                for relabel_idx in range(node_labels_t.shape[0]):
                    if float(pmax[relabel_idx]) >= self.relabel_thr_high:
                        new_lab = int(pred_cls[relabel_idx].item())
                        gt_local_idx = node_img_local_to_valid[relabel_idx]
                        if 0 <= gt_local_idx < len(valid_instance_indices):
                            valid_idx = valid_instance_indices[gt_local_idx]
                            old_lab = int(node_labels_t[relabel_idx].item())
                            # Atualiza tensores/listas deste batch
                            node_labels_t[relabel_idx] = new_lab
                            inst = subds.data_list[d_idx]['instances'][valid_idx]
                            updated = False
                            for key in ['labels','label','bbox_label']:
                                if key in inst:
                                    inst[key] = new_lab
                                    updated = True
                            if not updated:
                                inst['labels'] = new_lab
                            # Memoriza para overlay na Fase 3 (por imagem, indexado pelo GT local)
                            if img_path not in self._phase1_relabels:
                                self._phase1_relabels[img_path] = {}
                            # chave: índice local de GT usado para construir os nós (node_img_local_to_valid)
                            self._phase1_relabels[img_path][int(gt_local_idx)] = int(old_lab)
                        # else:
                        #     if hasattr(runner, 'logger'):
                        #         runner.logger.warning(f"[Phase1] Skip relabel write: gt_local_idx={gt_local_idx} out of range for {os.path.basename(img_path)}")

                # Presence-based co-occurrence update (after relabel-high)
                eff_labels_t = node_labels_t.clone()
                if eff_labels_t.numel() > 1:
                    uniq = torch.unique(eff_labels_t.detach().to(torch.long))
                    if uniq.numel() > 1:
                        ui = uniq.unsqueeze(1).expand(-1, uniq.numel())
                        uj = uniq.unsqueeze(0).expand(uniq.numel(), -1)
                        mask = (ui != uj)
                        pairs_i = ui[mask].reshape(-1).to('cpu', dtype=torch.long)
                        pairs_j = uj[mask].reshape(-1).to('cpu', dtype=torch.long)
                        self._cooc_counts.index_put_((pairs_i, pairs_j), torch.ones_like(pairs_i, dtype=torch.float32), accumulate=True)

        # Freeze epoch prior
        # epoch_device = runner.model.device
        epoch_device = next(runner.model.parameters()).device
        co_probs_epoch = self._compute_co_probs(epoch_device)

        # === PHASE 2: Train ConG using the fixed co_probs_epoch ===
        steps = 0
        for data_batch in dataloader:
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0 else next(runner.model.parameters()).device)
            self._ensure_cong(device)
            batch_loss = None
            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue
                pred_instances = preds[i].pred_instances.to(device)
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')
                gt = ds.gt_instances.to(device)
                assigner = MaxIoUAssigner(pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner,
                                          min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality)
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue
                logits = pred_instances.logits
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []
                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    choice = assoc[j]
                    node_labels.append(int(gt.labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)
                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)
                # usar logits (pré-softmax) como descritor visual
                vis_feats = node_logits_t

                # Build context inputs
                H_img = inputs.shape[-2]
                W_img = inputs.shape[-1]
                gt_boxes_xyxy = gt.bboxes.tensor
                sel_gt_xyxy = gt_boxes_xyxy[node_img_local_to_valid]
                sp = torch.stack([spatial7_from_xyxy(sel_gt_xyxy[k], W_img, H_img) for k in range(sel_gt_xyxy.size(0))], dim=0)
                # === Consistência de dimensões entre pr/vis_feats/labels/sp/boxes ===
                N_pr   = pr.size(0)
                N_feat = vis_feats.size(0)
                N_lab  = node_labels_t.size(0)
                N_box  = sel_gt_xyxy.size(0)
                N_sp   = sp.size(0)
                min_n = min(N_pr, N_feat, N_lab, N_box, N_sp)
                if not (N_pr == N_feat == N_lab == N_box == N_sp):
                    pr = pr[:min_n]
                    vis_feats = vis_feats[:min_n]
                    node_logits_t = node_logits_t[:min_n]
                    node_labels_t = node_labels_t[:min_n]
                    sel_gt_xyxy = sel_gt_xyxy[:min_n]
                    sp = sp[:min_n]
                Nn = pr.size(0)
                if Nn <= 1:
                    pr_ctx = pr
                else:
                    probs = pr
                    # confidences (neighbors j)
                    pmax = probs.max(dim=-1).values
                    w_conf = (pmax.clamp_min(1e-6) ** self.ctx_conf_beta)
                    # cosine similarity nas probabilidades
                    Pn = F.normalize(pr, p=2, dim=-1)
                    S_prob = torch.mm(Pn, Pn.t())
                    S_prob = (S_prob + 1.0) * 0.5
                    # cosine similarity visual (logits)
                    Fn = F.normalize(vis_feats, p=2, dim=-1)
                    S_feat = torch.mm(Fn, Fn.t())
                    S_feat = (S_feat + 1.0) * 0.5
                    # combinação com pesos configuráveis (com ajuste defensivo de tamanho)
                    if S_prob.shape != S_feat.shape:
                        min_n = min(S_prob.shape[0], S_feat.shape[0])
                        S_prob = S_prob[:min_n, :min_n]
                        S_feat = S_feat[:min_n, :min_n]
                    S = self.ctx_prob_weight * S_prob + self.ctx_feat_weight * S_feat
                    S.fill_diagonal_(0.0)
                    # máscara para suprimir pares visualmente dissimilares
                    S_mask = (S_feat >= self.ctx_sim_min).float()
                    # spatial Gaussian kernel using dest box sizes
                    b = sel_gt_xyxy
                    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
                    cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                    w  = (x2-x1).clamp(min=1e-3); h=(y2-y1).clamp(min=1e-3)
                    DX = (cx.unsqueeze(1) - cx.unsqueeze(0)).abs() / (w.unsqueeze(0) + 1e-3)
                    DY = (cy.unsqueeze(1) - cy.unsqueeze(0)).abs() / (h.unsqueeze(0) + 1e-3)
                    D  = torch.sqrt(DX**2 + DY**2)
                    sigma = self.ctx_dist_sigma
                    Ksp = torch.exp(-D / max(1e-6, sigma))
                    # co-occurrence attenuation between labels
                    li = node_labels_t
                    Cij = co_probs_epoch.to(device)[li][:, li]
                    # final weights W[i,j] (j contributes to i)
                    W = (S * Ksp * Cij) * S_mask
                    W = W * w_conf.unsqueeze(0)
                    W.fill_diagonal_(0.0)
                    den = W.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    pr_ctx = (W @ probs) / den
                pmax = pr.max(dim=-1).values
                alpha_conf = torch.sigmoid(self.ctx_gate_gamma * (pmax - self.trust_thr)).unsqueeze(1)
                alpha = alpha_conf
                pr_mix = alpha * pr + (1.0 - alpha) * pr_ctx
                x_cong = torch.cat([pr_mix, sp], dim=1)

                # Graph with fixed epoch prior
                co_probs = co_probs_epoch.to(device)
                edge_index, edge_attr = self._build_semantic_graph(
                    pr, node_labels_t, co_probs_epoch.to(device), sel_gt_xyxy, W_img, H_img, k=4,
                    features=vis_feats, feat_weight=self.ctx_feat_weight, prob_weight=self.ctx_prob_weight)

                # Train ConG
                self._cong.train()
                logits_cong = self._cong(x_cong, edge_index, edge_attr)
                loss_cong = F.cross_entropy(logits_cong, node_labels_t)
                if batch_loss is None:
                    batch_loss = loss_cong
                else:
                    batch_loss = batch_loss + loss_cong

            if batch_loss is not None:
                self._opt_cong.zero_grad()
                batch_loss.backward()
                self._opt_cong.step()
                steps += 1
                if steps >= self.cong_train_steps:
                    break
        if steps >= self.cong_train_steps and hasattr(runner, 'logger'):
            runner.logger.info(f"[ConG] Trained with fixed epoch co_probs: steps={steps}")

     
        for data_batch in dataloader:
            # forward do detector SEM grad (apenas extrair preds)
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0 else next(runner.model.parameters()).device)

            self._ensure_cong(device)
            co_probs = self._compute_co_probs(device)
            # ===== por-amostra do batch =====
            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue
                # garanta que TUDO está no mesmo device
                pred_instances = preds[i].pred_instances.to(device)
                # alguns heads expõem 'bboxes'; o assigner aceita 'priors' também
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')

                gt = ds.gt_instances.to(device)

                # referências locais (já no device)
                bboxes = gt.bboxes
                labels = gt.labels
                priors = pred_instances.priors
                logits = pred_instances.logits  # [Np, C]
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                # assign predições -> GTs
                assigner = MaxIoUAssigner(pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner, min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality)
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []

                # mapeamento de instâncias válidas no dataset (para escrita)
                sub_idx, d_idx = dataset_img_map[img_path]
                subds = datasets[sub_idx]
                valid_instance_indices = [idx for idx, inst in enumerate(subds.data_list[d_idx]['instances']) if inst['ignore_flag'] == 0]

                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    choice = assoc[j]
                    node_labels.append(int(labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)
                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)  # [N,C]
                vis_feats = node_logits_t
                # veto: se a classe predita == GT, não considerar como ruído
                pred_cls_t = pr.argmax(dim=1)                  # [N]
                veto_eq_t = (pred_cls_t == node_labels_t)      # [N] bool



                # rótulos efetivos para consultar co-ocorrência (usa pred quando muito confiante)
                eff_labels_t, _, pmax_t, argmax_t = self._effective_labels_from_logits(node_logits_t, node_labels_t, self.trust_thr)

                H_img = inputs.shape[-2]
                W_img = inputs.shape[-1]
                gt_boxes_xyxy = gt.bboxes.tensor
                sel_gt_xyxy = gt_boxes_xyxy[node_img_local_to_valid]
                sp_list = [spatial7_from_xyxy(sel_gt_xyxy[k], W_img, H_img) for k in range(sel_gt_xyxy.size(0))]
                if len(sp_list) == 0:
                    continue
                sp = torch.stack(sp_list, dim=0)  # [N,7]

                # --- contexto ponderado por confiança + gate por confiança e minoritarismo ---
                Nn = pr.size(0)
                # cosine similarities: probs e visual (logits)
                Pn = F.normalize(pr, p=2, dim=-1)
                S_prob = torch.mm(Pn, Pn.t()); S_prob = (S_prob + 1.0) * 0.5
                Fn = F.normalize(vis_feats, p=2, dim=-1)
                S_feat = torch.mm(Fn, Fn.t()); S_feat = (S_feat + 1.0) * 0.5

                # combinação com pesos configuráveis (com ajuste defensivo de tamanho)
                if S_prob.shape != S_feat.shape:
                    min_n = min(S_prob.shape[0], S_feat.shape[0])
                    S_prob = S_prob[:min_n, :min_n]
                    S_feat = S_feat[:min_n, :min_n]
                S = self.ctx_prob_weight * S_prob + self.ctx_feat_weight * S_feat
                S.fill_diagonal_(0.0)
                # gate visual
                S_mask = (S_feat >= self.ctx_sim_min).float()

                # usar logits como features visuais adiante
                features = vis_feats
                if Nn <= 1:
                    pr_ctx = pr
                else:
                    probs = pr
                    pmax = probs.max(dim=-1).values
                    w_conf = (pmax.clamp_min(1e-6) ** self.ctx_conf_beta)
                    b = sel_gt_xyxy
                    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
                    cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                    w  = (x2-x1).clamp(min=1e-3); h=(y2-y1).clamp(min=1e-3)
                    DX = (cx.unsqueeze(1) - cx.unsqueeze(0)).abs() / (w.unsqueeze(0) + 1e-3)
                    DY = (cy.unsqueeze(1) - cy.unsqueeze(0)).abs() / (h.unsqueeze(0) + 1e-3)
                    D  = torch.sqrt(DX**2 + DY**2)
                    sigma = self.ctx_dist_sigma
                    Ksp = torch.exp(-D / max(1e-6, sigma))
                    li = eff_labels_t
                    Cij = co_probs[li][:, li] if (li.numel() == Nn) else torch.ones_like(S)
                    # Peso final: combinação da similaridade de features, probabilidade e co-ocorrência
                    W = (S * Ksp * Cij) * S_mask
                    W = W * w_conf.unsqueeze(0)
                    W.fill_diagonal_(0.0)
                    den = W.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    pr_ctx = (W @ probs) / den
                # gate: self vs contexto condicionado por confiança e minoritarismo
                probs = pr
                pmax = probs.max(dim=-1).values
                alpha_conf = torch.sigmoid(self.ctx_gate_gamma * (pmax - self.trust_thr)).unsqueeze(1)
                alpha = alpha_conf
                pr_mix = alpha * probs + (1.0 - alpha) * pr_ctx
                x_cong = torch.cat([pr_mix, sp], dim=1)
                # --- Build semantic graph with co-occurrence attributes and feature similarity ---
                # edge_attr agora inclui f_sim como segunda coluna
                edge_index, edge_attr = self._build_semantic_graph(
                    pr, eff_labels_t, co_probs, sel_gt_xyxy, W_img, H_img, k=4,
                    features=vis_feats, feat_weight=self.ctx_feat_weight, prob_weight=self.ctx_prob_weight
                )
                self._cong.eval()
                with torch.no_grad():
                    logits_cong = self._cong(x_cong, edge_index, edge_attr)
                    qc = logits_cong.softmax(dim=-1)  # [N,C]

                old_labels_before_agree = node_labels_t.clone()
                relabeled_pairs = []  # lista de tuples (local_idx, old_gt)
                # --- AGREEMENT RELABEL (modelo & contexto concordam) ---
                if qc.shape[0] == pr.shape[0] and pr.shape[0] > 1:
                    pmax_t = pr.max(dim=1).values               # confiança do modelo por nó
                    pred_cls_t = pr.argmax(dim=1)               # classe do modelo por nó
                    ctx_cls_t = qc.argmax(dim=1)                # classe de contexto por nó
                    agree = (pred_cls_t == ctx_cls_t) & (pmax_t >= self.relabel_thr_ctx)
                    agree_idx = torch.nonzero(agree, as_tuple=False).flatten()
                    for _li in agree_idx.tolist():
                        new_lab = int(pred_cls_t[_li].item())
                        old_lab = int(node_labels_t[_li].item())
                        if new_lab == old_lab:
                            # nada a fazer; não marca como relabel
                            continue
                        # atualiza o tensor de labels do batch
                        node_labels_t[_li] = new_lab
                        relabeled_pairs.append((_li, old_lab))
                        # persiste no dataset para próximas épocas
                        if _li < len(node_img_local_to_valid):
                            gt_idx = node_img_local_to_valid[_li]
                            if gt_idx < len(valid_instance_indices):
                                _valid_idx = valid_instance_indices[gt_idx]
                                inst = subds.data_list[d_idx]['instances'][_valid_idx]
                                updated = False
                                for key in ['bbox_label', 'label', 'labels']:
                                    if key in inst:
                                        inst[key] = new_lab
                                        updated = True
                                if not updated:
                                    inst['labels'] = new_lab
                    # Como alteramos node_labels_t, recomputamos eff_labels_t e veto_eq_t
                    eff_labels_t, _, pmax_t, argmax_t = self._effective_labels_from_logits(node_logits_t, node_labels_t, self.trust_thr)
                    veto_eq_t = (pred_cls_t == node_labels_t)
                    disagree_mask = (pred_cls_t != node_labels_t).detach().cpu().numpy().astype(bool)
                # else:
                #     if hasattr(runner, 'logger'):
                #         runner.logger.warning(f"[ConG] Skip agreement relabel: "
                #                               f"N={pr.shape[0]}, qc={qc.shape[0]} pr={pr.shape[0]} on {os.path.basename(img_path)}")

                eps = 1e-7
                p = pr.clamp_min(eps)
                q = qc.clamp_min(eps)
                # --- filtro contextual + co-ocorrência real ---
                kld = (p * (p.log() - q.log())).sum(dim=1)  # [N]
                kld_np = kld.detach().cpu().numpy()
                # usa KLD absoluto como p_noise (sem normalização por imagem)
                p_noise = kld
                p_noise_np = kld_np

                # --- filtro auxiliar de co-ocorrência real ---
                # para cada nó i (classe li), verifica se existe alguma classe w presente na imagem tal que co(li,w) >= corr_thr_low
                Nn = eff_labels_t.numel()
                if Nn > 1:
                    li = eff_labels_t
                    lj = eff_labels_t
                    C = co_probs[li][:, lj]  # [N,N]
                    C = C.masked_fill(torch.eye(Nn, dtype=torch.bool, device=C.device), 0.0)
                    best, _ = C.max(dim=1)   # [N]
                    best_np = best.detach().cpu().numpy()
                    low_corr_mask = (best < float(self.corr_thr_low))  # True quando baixa correlação com classes presentes
                else:
                    low_corr_mask = torch.zeros_like(kld, dtype=torch.bool)
                low_corr_np = low_corr_mask.detach().cpu().numpy().astype(bool)

                # define corte da parte contextual (KLD normalizado)
                if self.use_percentile and p_noise_np.size > 0:
                    cut = float(np.percentile(p_noise_np, self.percentile))
                else:
                    cut = float(self.thr_noise)

                # --- veto para minoritários plausíveis ---
                # veto_np = np.zeros_like(low_corr_np, dtype=bool)
                # if Nn > 0:
                #     # maioria por frequência na imagem (com rótulos efetivos)
                #     counts = torch.bincount(eff_labels_t, minlength=self.num_classes).float()
                #     majority = int(counts.argmax().item())
                #     # co-ocorrência com a classe majoritária
                #     co_with_major = co_probs[eff_labels_t, majority]  # [N]
                #     co_with_major_np = co_with_major.detach().cpu().numpy()
                #     freq_i = (counts[eff_labels_t] / max(float(eff_labels_t.numel()), 1.0)).detach().cpu().numpy()
                #     minority_np = (freq_i < float(self.rho_min))
                #     strong_cooc_np = (co_with_major_np >= float(self.corr_thr_pos))
                #     veto_np = np.logical_and(minority_np, strong_cooc_np)

                # veto adicional: predição igual ao GT nunca é ruído
                veto_eq_np = veto_eq_t.detach().cpu().numpy().astype(bool)

                # --- W&B: loga alguns exemplos visuais ---
                if self.use_wandb and (wandb is not None):
                    # loga até `wandb_max_images` imagens por época com overlay
                    if self._wandb_img_budget > 0:
                        # Log only if at least one bbox had its label actually changed in this image
                        relabeled_pairs = relabeled_pairs if 'relabeled_pairs' in locals() else []
                        has_relabel = len(relabeled_pairs) > 0
                        should_log = has_relabel
                        try:
                            relabeled_pairs = relabeled_pairs if 'relabeled_pairs' in locals() else []
                            relabeled_set = set([int(i) for (i, _) in relabeled_pairs]) if 'relabeled_pairs' in locals() else set()
                            old_gt_map = {int(i): int(old) for (i, old) in relabeled_pairs} if 'relabeled_pairs' in locals() else {}
                            # --- Merge with Phase-1 relabels (pink) ---
                            phase1_map_for_img = self._phase1_relabels.get(img_path, {})  # {gt_local_idx -> old_lab}
                            relabel_high_set = set()
                            old_gt_map_phase1 = {}
                            # Mapear do índice de nó local (local_idx) para info da Fase-1 via gt_local_idx
                            for local_idx, gt_local_idx in enumerate(node_img_local_to_valid[:len(sel_gt_xyxy)]):
                                if int(gt_local_idx) in phase1_map_for_img:
                                    relabel_high_set.add(int(local_idx))
                                    old_gt_map_phase1[int(local_idx)] = int(phase1_map_for_img[int(gt_local_idx)])
                            # noisy se (KLD alto) E (baixa co-ocorrência real)
                            noisy_kld = (p_noise_np >= cut)
                            noisy_mask = np.logical_and(noisy_kld, low_corr_np)
                            # aplica vetos: minoria plausível e pred==GT
                            # if 'veto_np' in locals() and len(veto_np) == len(noisy_mask):
                            #     noisy_mask = np.logical_and(noisy_mask, np.logical_not(veto_np))
                            if 'veto_eq_np' in locals() and len(veto_eq_np) == len(noisy_mask):
                                noisy_mask = np.logical_and(noisy_mask, np.logical_not(veto_eq_np))
                            if should_log:
                                # reconstrói imagem BGR para desenho usando img_norm_cfg real
                                # Carrega a imagem original em BGR e redimensiona para o img_shape atual do batch
                                # (mantém cores corretas para desenho via OpenCV)
                                meta = getattr(data_samples[i], 'metainfo', {})
                                img_shape = tuple(meta.get('img_shape', (inputs.shape[-2], inputs.shape[-1])))  # (H, W, 3)
                                H_v, W_v = int(img_shape[0]), int(img_shape[1])
                                img_np = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
                                if img_np is None:
                                    # fallback: tenta usar o tensor (pode ter cores trocadas)
                                    norm_cfg = meta.get('img_norm_cfg', {})
                                    mean = norm_cfg.get('mean', [123.675, 116.28, 103.53])
                                    std = norm_cfg.get('std', [58.395, 57.12, 57.375])
                                    to_rgb = norm_cfg.get('to_rgb', True)
                                    img_np = tensor_to_numpy_img(inputs[i].cpu(), mean=mean, std=std, to_rgb=to_rgb)
                                else:
                                    img_np = cv2.resize(img_np, (W_v, H_v), interpolation=cv2.INTER_LINEAR)
                                    # aplica o mesmo flip da pipeline para alinhar as boxes
                                    if bool(meta.get('flip', False)):
                                        fd = str(meta.get('flip_direction', 'horizontal'))
                                        if 'diagonal' in fd:
                                            img_np = cv2.flip(img_np, -1)
                                        else:
                                            if 'horizontal' in fd:
                                                img_np = cv2.flip(img_np, 1)
                                            if 'vertical' in fd:
                                                img_np = cv2.flip(img_np, 0)
                                # desenha GTs associados com cor conforme noise
                                sel_gt_xyxy_np = sel_gt_xyxy.detach().cpu().numpy().astype(int)
                                for local_idx in range(min(len(node_img_local_to_valid), len(p_noise_np))):
                                    x1, y1, x2, y2 = sel_gt_xyxy_np[local_idx].tolist()
                                    is_noisy = bool(noisy_mask[local_idx])
                                    is_kld = bool(noisy_kld[local_idx])
                                    # prioridade de cor: relabel (AZUL/Fase-2) > relabel-high (ROSA/Fase-1) > noisy (VERMELHO) > KLD alto vetado (LARANJA) > discordância (AMARELO) > limpo (VERDE)
                                    if local_idx in relabeled_set:
                                        color = (255, 0, 0)            # AZUL (reanotado pela rede de grafos / Phase-2)
                                    elif local_idx in relabel_high_set:
                                        color = (255, 105, 180)        # ROSA (reanotado na Phase-1: alta confiança do modelo)
                                    elif 'disagree_mask' in locals() and local_idx < len(disagree_mask) and bool(disagree_mask[local_idx]):
                                        color = (0, 255, 255)          # AMARELO (pred != label atual)
                                    else:
                                        color = (0, 255, 0)            # VERDE

                                    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
                                    # obter classes GT/pred, KLD e pmax para overlay
                                    _gt_lab = int(node_labels[local_idx])
                                    _pred_lab = int(pr[local_idx].argmax().item())
                                    _kld = float(kld_np[local_idx])
                                    _pmax = float(pr[local_idx].max().item())
                                    _co = float(best_np[local_idx]) if 'best_np' in locals() and local_idx < len(best_np) else 0.0
                                    # Adiciona classe esperada pelo contexto (ctx) usando qc.argmax(dim=1)
                                    _ctx_lab = int(qc[local_idx].argmax().item()) if qc is not None and local_idx < qc.shape[0] else -1
                                    _v_eq = bool(veto_eq_np[local_idx]) if 'veto_eq_np' in locals() and local_idx < len(veto_eq_np) else False
                                    _old2 = old_gt_map.get(local_idx, None)                 # Phase-2 old label (ConG)
                                    _old1 = old_gt_map_phase1.get(local_idx, None)          # Phase-1 old label (high confidence)
                                    # Preferir mostrar o marcador correspondente à cor aplicada
                                    if local_idx in relabeled_set and _old2 is not None:
                                        rx_suffix = f"|R{_old2}"
                                    elif local_idx in relabel_high_set and _old1 is not None:
                                        rx_suffix = f"|R{_old1}"
                                    else:
                                        rx_suffix = ""
                                    _overlay_txt = f"gt{_gt_lab}|p{_pred_lab}|n{float(p_noise_np[local_idx]):.2f}|k{_kld:.2f}|s{_pmax:.2f}|c{_co:.2f}|ct{_ctx_lab}{rx_suffix}"
                                    cv2.putText(img_np,
                                                _overlay_txt,
                                                (max(0, x1), max(15, y1-4)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                # Converte BGR (OpenCV) → RGB antes de enviar ao W&B
                                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                                self._wandb_imgs.append(
                                    wandb.Image(img_rgb, caption=f"epoch={runner.epoch+1} | {os.path.basename(img_path)}")
                                )
                                self._wandb_img_budget -= 1
                                # Opcional: logar figura pr vs pr_ctx **apenas para nós noisy**
                                # if getattr(self, 'wandb_log_prctx', False):
                                #     noisy_idx = np.where(noisy_mask)[0].tolist()
                                #     if len(noisy_idx) > 0:
                                #         # respeita o limite de nós; seleciona os primeiros noisy
                                #         noisy_idx = noisy_idx[:max(1, getattr(self, 'prctx_max_nodes', 8))]
                                #         idx_t = torch.as_tensor(noisy_idx, device=pr.device, dtype=torch.long)
                                #         pr_sel = pr.index_select(0, idx_t)
                                #         pr_ctx_sel = pr_ctx.index_select(0, idx_t)
                                #         qc_sel = qc.index_select(0, idx_t) if qc is not None else None
                                #         labels_sel = node_labels_t.index_select(0, idx_t)

                                #         cls_names = None
                                #         try:
                                #             cls_names = data_samples[i].metainfo.get('classes', None)
                                #         except Exception:
                                #             cls_names = None
                                #         title = f"pr vs pr_ctx vs qc (noisy only) | {os.path.basename(img_path)}"
                                #         prctx_img = _make_prctx_figure(pr_sel, pr_ctx_sel, qc_sel, labels_sel,
                                #                                        topk=getattr(self, 'prctx_topk', 5),
                                #                                        max_nodes=len(noisy_idx),
                                #                                        class_names=cls_names,
                                #                                        title=title)
                                #         if prctx_img is not None:
                                #             self._wandb_imgs.append(wandb.Image(prctx_img, caption=f"pr_ctx noisy | epoch={runner.epoch+1} | {os.path.basename(img_path)}"))
                                # Extra: logar pr/pr_ctx/qc para casos com discordância modelo x label
                                if getattr(self, 'wandb_log_prctx', False) and 'disagree_mask' in locals():
                                    disagree_idx = np.where(disagree_mask)[0].tolist()
                                    if len(disagree_idx) > 0:
                                        disagree_idx = disagree_idx[:max(1, getattr(self, 'prctx_max_nodes', 8))]
                                        idx_t2 = torch.as_tensor(disagree_idx, device=pr.device, dtype=torch.long)
                                        pr_sel = pr.index_select(0, idx_t2)
                                        pr_ctx_sel = pr_ctx.index_select(0, idx_t2)
                                        qc_sel = qc.index_select(0, idx_t2) if qc is not None else None
                                        labels_sel = node_labels_t.index_select(0, idx_t2)

                                        cls_names = None
                                        try:
                                            cls_names = data_samples[i].metainfo.get('classes', None)
                                        except Exception:
                                            cls_names = None

                                        title2 = f"pr vs pr_ctx vs qc (pred!=label) | {os.path.basename(img_path)}"
                                        prctx_img2 = _make_prctx_figure(pr_sel, pr_ctx_sel, qc_sel, labels_sel,
                                                                        topk=getattr(self, 'prctx_topk', 5),
                                                                        max_nodes=len(disagree_idx),
                                                                        class_names=cls_names,
                                                                        title=title2)
                                        if prctx_img2 is not None:
                                            self._wandb_imgs.append(
                                                wandb.Image(prctx_img2, caption=f"pr_ctx disagree | epoch={runner.epoch+1} | {os.path.basename(img_path)}")
                                            )
                        except Exception as e:
                            if hasattr(runner, 'logger'):
                                runner.logger.warning(f"[W&B] Falha ao montar/registrar imagem: {e}")

                # aplica REWEIGHT (ignore_flag comentado)
                L = min(len(node_img_local_to_valid), len(p_noise_np))
                for local_idx in range(L):
                    gt_idx = node_img_local_to_valid[local_idx]
                    valid_idx = valid_instance_indices[gt_idx] if gt_idx < len(valid_instance_indices) else None
                    if valid_idx is None:
                        continue
                    if (runner.epoch + 1) <= self.warmup_epochs:
                        continue
                    is_lowcorr = bool(low_corr_np[local_idx])
                    # is_veto = bool(veto_np[local_idx]) if 'veto_np' in locals() and local_idx < len(veto_np) else False
                    is_veto_eq = bool(veto_eq_np[local_idx]) if 'veto_eq_np' in locals() and local_idx < len(veto_eq_np) else False
                    #if (float(p_noise_np[local_idx]) >= cut) and is_lowcorr and (not is_veto) and (not is_veto_eq):
                    if (float(p_noise_np[local_idx]) >= cut) and is_lowcorr and (not is_veto_eq):
                        lw = max(0.2, 1.0 - self.cong_alpha * float(p_noise_np[local_idx]))
                        # subds.data_list[d_idx]['instances'][valid_idx]['loss_weight'] = lw
                        pass
                        # subds.data_list[d_idx]['instances'][valid_idx]['ignore_flag'] = 1  # ignorar apenas quando também houver baixa co-ocorrência
                    
                    # pred_label = int(node_labels_t[local_idx].item())
                    # pred_labelv2 = int(eff_labels_t[local_idx].item())
                    # import pdb; pdb.set_trace()
                    # subds.data_list[d_idx]['instances'][valid_idx]['bbox_label'] = pred_label

          

        # Salvar matriz de co-ocorrência como imagem de heatmap usando matplotlib
        try:
            # 1) Salvar heatmap da co-ocorrência
            cooc_dir = os.path.join(runner.work_dir, 'debug_cooc')
            os.makedirs(cooc_dir, exist_ok=True)
            cooc_path = os.path.join(cooc_dir, f'cooc_matrix_epoch{runner.epoch + 1}.png')
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patheffects as _pe
                plt.figure(figsize=(6, 5))
                _co = self._cooc_counts.clone().float()
                _row = _co.sum(dim=1, keepdim=True).clamp_min(1e-6)
                co_probs = (_co / _row)
                co_probs.fill_diagonal_(1.0)
                _vis = co_probs.numpy()
                im = plt.imshow(co_probs.cpu(), cmap='viridis')
                plt.title(f"Co-occurrence Matrix - Epoch {runner.epoch + 1}")
                plt.colorbar(im)
                plt.xlabel("j (given i→j)")
                plt.ylabel("i")

                # Ajusta os eixos para valores inteiros
                n_classes = co_probs.shape[0]
                plt.xticks(range(n_classes), range(n_classes))
                plt.yticks(range(n_classes), range(n_classes))
                plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

                # === NEW: escreve o número de ocorrências (contagem bruta) em cada célula ===
                _co_np = _co.detach().cpu().numpy()
                for i_tick in range(n_classes):
                    for j_tick in range(n_classes):
                        cnt = int(_co_np[i_tick, j_tick])
                        if cnt <= 0:
                            continue
                        # escolhe cor do texto conforme intensidade para contraste
                        txt_color = 'white' if _vis[i_tick, j_tick] > 0.2 else 'white'
                        plt.text(j_tick, i_tick, str(cnt), ha='center', va='center', fontsize=6,
                                 color=txt_color, path_effects=[_pe.withStroke(linewidth=1, foreground='black')])

                plt.tight_layout()
                plt.savefig(cooc_path)
                plt.close()
            except Exception as e_heat:
                if hasattr(runner, 'logger'):
                    runner.logger.warning(f"[Cooc] Falha ao salvar heatmap: {e_heat}")

            # 2) Forçar log no W&B (imagem, tabela e heatmap interativo)
            if getattr(self, 'use_wandb', False) and (wandb is not None) and getattr(self, '_wandb_ready', False):
                log_dict = {}
                if hasattr(self, '_wandb_imgs') and len(self._wandb_imgs) > 0:
                    log_dict['debug_imgs'] = self._wandb_imgs

                # Recalcula co_probs normalizado (CPU→numpy) para tabela/heatmap
                _co_counts = self._cooc_counts.clone().float()
                _row = _co_counts.sum(dim=1, keepdim=True).clamp_min(1e-6)
                _co_probs = (_co_counts / _row)
                _co_probs.fill_diagonal_(1.0)
                _co_probs_np = _co_probs.detach().cpu().numpy()

                # Tenta obter nomes de classes
                class_names = None
                try:
                    # tenta via datasets concatenados
                    while hasattr(dataset, 'dataset'):
                        dataset_ = dataset.dataset
                    # fallback: tenta pelo primeiro subdataset
                    if 'datasets' in dir(dataset) and len(dataset.datasets) > 0:
                        sub0 = dataset.datasets[0]
                        if hasattr(sub0, 'METAINFO') and isinstance(sub0.METAINFO, dict):
                            class_names = sub0.METAINFO.get('classes', None)
                except Exception:
                    class_names = None

                # Constrói Tabela (i,j,value) para o heatmap
                data_rows = []
                C = int(_co_probs_np.shape[0])
                for i in range(C):
                    for j in range(C):
                        name_i = class_names[i] if (class_names is not None and i < len(class_names)) else str(i)
                        name_j = class_names[j] if (class_names is not None and j < len(class_names)) else str(j)
                        data_rows.append([name_i, name_j, float(_co_probs_np[i, j])])
                cooc_table = wandb.Table(data=data_rows, columns=["i", "j", "value"])  # tabela completa

                # Heatmap interativo no W&B
                try:
                    cooc_heatmap = wandb.plot.heatmap(cooc_table, x="j", y="i", value="value",
                                                      title=f"Co-occurrence (row-normalized) - Epoch {runner.epoch + 1}")
                    log_dict['cooc_heatmap'] = cooc_heatmap
                except Exception as e_hm:
                    if hasattr(runner, 'logger'):
                        runner.logger.warning(f"[W&B] Falha ao criar heatmap: {e_hm}")

                # === NEW: também loga a matriz de contagens brutas ===
                _co_counts_np = _co_counts.detach().cpu().numpy()
                data_rows_cnt = []
                for i_idx in range(C):
                    for j_idx in range(C):
                        name_i = class_names[i_idx] if (class_names is not None and i_idx < len(class_names)) else str(i_idx)
                        name_j = class_names[j_idx] if (class_names is not None and j_idx < len(class_names)) else str(j_idx)
                        data_rows_cnt.append([name_i, name_j, int(_co_counts_np[i_idx, j_idx])])
                cooc_counts_table = wandb.Table(data=data_rows_cnt, columns=["i", "j", "count"])  # contagens
                try:
                    cooc_counts_heatmap = wandb.plot.heatmap(cooc_counts_table, x="j", y="i", value="count",
                                                             title=f"Co-occurrence COUNTS - Epoch {runner.epoch + 1}")
                    log_dict['cooc_counts_heatmap'] = cooc_counts_heatmap
                except Exception as e_hm2:
                    if hasattr(runner, 'logger'):
                        runner.logger.warning(f"[W&B] Falha ao criar heatmap de contagens: {e_hm2}")
                log_dict['cooc_counts_table'] = cooc_counts_table

                # Anexa imagem PNG gerada
                if os.path.exists(cooc_path):
                    log_dict['cooc_matrix'] = wandb.Image(cooc_path)
                # Anexa a tabela numérica
                log_dict['cooc_table'] = cooc_table

                if len(log_dict) > 0:
                    wandb.log(log_dict, commit=True)
                    # esvazia o buffer de imagens para próxima época
                    if 'debug_imgs' in log_dict:
                        self._wandb_imgs.clear()
                    if hasattr(runner, 'logger'):
                        runner.logger.info(f"[W&B] Imagens, cooc_table e cooc_heatmap logados na epoch {runner.epoch + 1}")
        except Exception as e_final:
            if hasattr(runner, 'logger'):
                runner.logger.warning(f"[W&B] Falha no log final da epoch: {e_final}")

        
        # === PHASE 3: GMM-based noise filtering (after relabeling) ===
        # Faz um processo em duas passagens:
        #   Passo 3A) varre toda a época e acumula scores por classe (p_gt)
        #   Passo 3B) ajusta um GMM por classe usando TODOS os scores acumulados e marca low-confidence como noisy
        dataloader = runner.train_loop.dataloader
        dataset = dataloader.dataset

        #reload_dataset = True
        # my_value = 3
        #reload_dataset = getattr(runner.cfg, 'reload_dataset', False)  
        #my_value =  getattr(runner.cfg, 'my_value', 10)  
        # reload_dataset = self.reload_dataset
        # relabel_conf = self.relabel_conf
        
        # # import pdb; pdb.set_trace()
        # if reload_dataset:
        #     runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
        #     runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

        #     runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
        #     runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

        # Desencapsular RepeatDataset até o ConcatDataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        if not hasattr(dataset, 'datasets'):
            raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

        datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

        assigner = MaxIoUAssigner(
            pos_iou_thr=self.iou_assigner,
            neg_iou_thr=self.iou_assigner,
            min_pos_iou=self.iou_assigner,
            match_low_quality=self.low_quality
        )

        # Criar um dicionário para acesso rápido ao índice das imagens no dataset
        dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                        for sub_dataset_idx, sub_dataset in enumerate(datasets)
                        if hasattr(sub_dataset, 'data_list')
                        for data_idx, data_info in enumerate(sub_dataset.data_list)}
        
        # Dicionário para armazenar os scores médios por GT sem assumir sequência
        allbb_preds_map = defaultdict(dict)
        all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
        all_pred_instances_map = defaultdict(dict)
        # all_inputs_map = defaultdict(dict)

        num_classes = 20
        all_classes_preds_map=[]
        all_classes_gt_idx_map=[]
        for i in range(num_classes):
            all_classes_preds_map.append(defaultdict(list))
            all_classes_gt_idx_map.append(defaultdict(list))
        

        count_gts = 0
        sanity_count = 0
        global_index_counter = 0

        
        # Processar o batch de dados
        for batch_idx, data_batch in enumerate(dataloader):
            with torch.no_grad():
                #data = runner.model.data_preprocessor(data_batch, False)
                data = runner.model.data_preprocessor(data_batch, True)
                
                inputs = data['inputs']
                data_samples = data['data_samples']
                
                #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
            
                #temp_filipe

            for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
            # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                img_path = data_sample.img_path
                
                if img_path not in dataset_img_map:
                    print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                    continue

                # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                pred_instances = predictions_pred[i].pred_instances

                # Pegar bounding boxes preditas
                pred_instances.priors = pred_instances.pop('bboxes')

                # Obter os ground truths
                gt_instances = data_sample.gt_instances
                gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                # Garantir que os GTs estão no mesmo dispositivo das predições
                gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                # Garantir que as predições também estão no mesmo dispositivo
                pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                all_pred_instances_map[img_path] = pred_instances
                

                assign_result = assigner.assign(pred_instances, gt_instances)

                updated_labels = gt_labels.clone()

                # Criar lista de GTs para esta imagem
                all_gt_idx_map[img_path] = []
                count_gts += assign_result.num_gts

                
                # **Vetorização do processo de associação**
                for gt_idx in range(assign_result.num_gts):
                    associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    sanity_count += 1
                    
                    if associated_preds.numel() == 0:
                        continue 

                    # max_score = pred_instances.scores[associated_preds].max().item()
                    given_max_score = pred_instances.scores[associated_preds].max().item()
                    # max_logit = pred_instances.logits[associated_preds].max().item()
                    # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                    logits_associated = pred_instances.logits[associated_preds] 
                    myscores = torch.softmax(logits_associated ,dim=-1)

                    

                    # quero pegar o logit_gt da amostra que tem maior logit_pred
                    # import pdb; pdb.set_trace()
                    myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                    mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                    # max_score = myscores_gt.max().item()
                    max_score_val = myscores.max()
                    max_score_idx = myscores.argmax()
                    amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                    score_gt_max_pred = myscores_gt[amostra_id].cpu().item()
                    myious = assign_result.max_overlaps[associated_preds]
                    max_iou_idx = torch.argmax(myious)
                    amostra_id_iou, classe_id_iou = divmod(max_iou_idx.item(), myscores.size(1))
                    score_gt_max_iou = myscores_gt[amostra_id_iou].cpu().item()
                    score_pred_max_iou = myscores[amostra_id_iou, classe_id].cpu().item()

                    # Compute APS
                    row_scores = myscores[amostra_id]  # pega apenas a linha
                    p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                    S_APS_score = -row_scores[row_scores > p_y].sum().item()
                    # p_y = myscores[amostra_id, classe_id].item()
                    # S_APS_score = -myscores[myscores >= p_y].sum().item()


                    max_score_idx_local = torch.argmax(myscores_gt)
                    #----> remover depois do debug
                    #max_logit_idx_gt = associated_preds[max_score_idx_local]
                    max_logit_idx_gt = associated_preds[amostra_id]
                    logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()
                    


    
                    gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]

                    

                    # Pegar idx local do maior IoU
                    max_iou_values = assign_result.max_overlaps[associated_preds]
                    max_iou_idx_local = torch.argmax(max_iou_values)
                    max_iou_idx = associated_preds[max_iou_idx_local]

                    # Vetor contendo ambos (pode repetir se for o mesmo)
                    important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                    important_associated_ids = torch.unique(important_associated_ids)

                    
                    
                    # Armazenar `gt_idx` na ordem correta
                    all_gt_idx_map[img_path].append(gt_idx)

                

                    if self.selcand == 'max':
                        allbb_preds_map[img_path][gt_idx] = {'pred': score_gt_max_pred, 'logit':logit_gt_max_pred , 
                                                            'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                            'aps':S_APS_score,
                                                            'max_pred':max_score_val.cpu().item(), 'pred_label': classe_id,
                                                            'filtered':False}
                    elif self.selcand == 'iou':
                        allbb_preds_map[img_path][gt_idx] = {'pred': score_gt_max_iou, 'logit':logit_gt_max_pred , 
                                                            'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                            'aps':S_APS_score,
                                                            'max_pred':score_pred_max_iou, 'pred_label': classe_id_iou,
                                                            'filtered':False}
                    
                    else:
                        print("Warning: selcand deve ser 'max' ou 'iou'. Usando 'max' por padrão.")
                    global_index_counter += 1


                    

                    confident_preds =  associated_preds[myscores.max(dim=1).values> self.relabel_thr_high]
                    
    

                    if confident_preds.numel() > 0:
                        
                        #---> original
                        # pred_labels_confident = pred_instances.labels[confident_preds]
                        #---> temporario-filipe-debug - remover depois do debug
                        #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                        pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                        most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                    # elif self.group and len(associated_preds) > 1 and (max_score_val > 0.45):
                    #     labels_group = myscores.argmax(dim=1)
                    #     most_common_label, qtd = Counter(labels_group.tolist()).most_common(1)[0]

                    #     scores_most_common = myscores[:,most_common_label]
                    #     confident_most_common =  associated_preds[scores_most_common> 0.45]
                    #     # import pdb; pdb.set_trace()

                    #     # verifica se a quantidade é maior que 50% do total
                    #     if qtd > (ledn(associated_preds) / 2) and len(confident_most_common) > 2:
                            
                    #         most_common_label = most_common_label
                    #     # senao verifica se tem algum elemento com score maior que o threshold
                    #     elif confident_preds.numel() > 0:
                    #         pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)
                    #         most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                    
                    else:
                        continue  


                    updated_labels[gt_idx] = most_common_label
                    #atualiza também no mapeamento
                    allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                    #atualiza max logit-gt
                    gt_logits = logits_associated[:,most_common_label]
                    # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                    max_logit = gt_logits.max().item()

                    ###-->debug -remover depois
                    # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                    # max_score = myscores_gt.max().item()
                    # max_score_val = myscores.max()
                    # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                    myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                    max_score_idx = myscores.argmax()
                    amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                    score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                    # Compute APS
                    # p_y = myscores[amostra_id, classe_id].item()
                    # S_APS_score = -myscores[myscores >= p_y].sum().item()
                    # row_scores = myscores[amostra_id]  # pega apenas a linha
                    # p_y = row_scores[classe_id].item()
                    # S_APS_score = -row_scores[row_scores > p_y].sum().item()
                    row_scores = myscores[amostra_id]  # pega apenas a linha
                    p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                    S_APS_score = -row_scores[row_scores > p_y].sum().item()
                    # if S_APS_score < -1:
                    #     print(S_APS_score)
                    #     print("a")
                    #     import pdb; pdb.set_trace()
                    #     print("ok")
                    allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score

                    #
                    #-->[original]
                    #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                    #-->[temporario-filipe-debug] - remover depois do debug
                    allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                    allbb_preds_map[img_path][gt_idx]['filtered'] = True



                    
                    

                sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                sub_dataset = datasets[sub_dataset_idx]

                # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                # Atualizar os labels corretamente usando o mapeamento
                for gt_idx, valid_idx in enumerate(valid_instance_indices):
                    sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                    # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
        

        # só faz filter depois do warmup
        #if (runner.epoch + 1) >= self.filter_warmup:
        if (runner.epoch + 1) >= 2:

            # filtering
            #Unir todas as probabilidades das imagens e treinar o GMM**
            # all_scores = np.array([
            #     score['pred'] for img_scores in allbb_preds_map.values()
            #     for score in img_scores.values()
            # ]).reshape(-1, 1)
            # import pdb; pdb.set_trace()
            print("[DEBUG1]: Entrou no filtro")
            all_classes_low_confidence_scores_global_idx = []

            for c in range(num_classes):
                # c_class_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values() if score['gt_label'] == c
                # ]).reshape(-1, 1)

                # if runner.epoch + 1 >1:
                #     import pdb; pdb.set_trace()

                scores = np.array([])
                # img_indexes = np.array([])
                scores_dict = {'pred': np.array([]),
                                'logit': np.array([]),
                                'aps': np.array([])}
                low_conf_dict = {'pred': np.array([]),
                                'logit': np.array([]),
                                'aps': np.array([])}
                img_paths = np.array([], dtype=object)
                c_global_indexes = np.array([])

                for img_path, img_info in allbb_preds_map.items():
                    #for pred, gt_label, index  in img_info.values():
                    for temp_gt_idx, values in img_info.items():
                        #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #if gt_label== c:
                        if values['gt_label'] == c:
                            scores_dict['pred'] = np.append(scores_dict['pred'], values['pred'])
                            scores_dict['logit'] = np.append(scores_dict['logit'], values['logit'])
                            scores_dict['aps'] = np.append(scores_dict['aps'], values['aps'])

                            # calculando como pred apenas para ver depois se len(scores) == 0
                            scores = np.append(scores, values['pred'])

                            
                            

                            c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                            if img_path not in img_paths:
                                img_paths = np.append(img_paths, img_path)
                
                if len(scores) == 0:
                    print(f"Aviso: Nenhuma amostra encontrada para a classe {c}.")
                    continue

                
                

                # low_confidence_scores = calculate_gmm(c_class_scores, n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                            
                low_conf_dict['pred'] = calculate_gmm(scores_dict['pred'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                low_conf_dict['logit'] = calculate_gmm(scores_dict['logit'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                low_conf_dict['aps'] = calculate_gmm(scores_dict['aps'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)

                if self.filter_type == 'pred':
                    scores = scores_dict['pred']
                    low_confidence_scores = low_conf_dict['pred']
                elif self.filter_type == 'logit':
                    scores = scores_dict['logit']
                    low_confidence_scores = low_conf_dict['logit']
                elif self.filter_type == 'aps':
                    scores = scores_dict['aps']
                    low_confidence_scores = low_conf_dict['aps']


                # print("[DEBUG1.5]: INICIANDO GMM")

                c_class_scores = c_class_scores = scores.reshape(-1, 1)



                # threshold = self.filter_conf
                
                threshold = self.filter_confgmm
            
                low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                

                # draw pred hist
                save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_pred_ep{runner.epoch + 1}.png"
                low_confidence_indices_temp = np.where(low_conf_dict['pred'] > threshold)[0]
                draw_score_histogram(scores_dict['pred'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_confgmm)

                # draw logit hist
                save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_logit_ep{runner.epoch + 1}.png"
                low_confidence_indices_temp = np.where(low_conf_dict['logit'] > threshold)[0]
                draw_score_histogram(scores_dict['logit'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_confgmm)

                # draw aps hist
                save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_aps_ep{runner.epoch + 1}.png"
                low_confidence_indices_temp = np.where(low_conf_dict['aps'] > threshold)[0]
                draw_score_histogram(scores_dict['aps'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_confgmm)

                # Dentro do seu for c in range(num_classes):
                # import pdb; pdb.set_trace()
                # ... após o fit do GMM
                # plt.figure()
                # plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                # # Adiciona linha vertical que separa as duas classes com base no GMM
                # plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                # # plt.legend()
                # plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                # plt.xlabel('Score')
                # plt.ylabel('Frequência')
                # plt.grid(True)
                # plt.tight_layout()
                # # Criar pasta, se necessário
                # save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                # plt.savefig(save_path)
                # plt.close()
                

            
                

            # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
            # index = 0
            # import pdb; pdb.set_trace()
            my_counter = 0
            for img_path, gt_scores in allbb_preds_map.items():
                sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                sub_dataset = datasets[sub_dataset_idx]


                valid_instance_indices = [
                    idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                    if inst['ignore_flag'] == 0
                ]

                # **Mapear GTs reais na ordem correta**
                gt_idx_list = all_gt_idx_map[img_path]  

                
                

                for gt_idx in gt_idx_list:
                # for gt_idx in valid_instance_indices:

                    related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']  

                    

                    #if index in low_confidence_indices:
                    #if related_global_index in all_classes_low_confidence_scores_global_idx:
                    # if low confidence and not too high confident
                    #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.relabel_conf):
                    #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.5):
                    
                    # if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.3):
                    if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                        # if my_counter<5:

                                                            
                        #     import shutil

                        #     # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                            
                        #     base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                        #     possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg", "_grouped.jpg"]

                        #     for suffix in possible_suffixes:
                                
                        #         base_debug_path = base_prefix + suffix
                        #         if os.path.exists(base_debug_path):
                        #             my_counter+=1 
                        #             filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                        #             # if suffix == "_relabeled.jpg":
                        #             #     import pdb; pdb.set_trace()
                        #             shutil.copy(base_debug_path, filtered_debug_path)
                        #             print(f"[INFO] Cópia criada: {filtered_debug_path}")
                        #             break  # Para no primeiro que encontrar 

                        #     # desenhar_bboxesv3_filtered(all_inputs_map[img_path], sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_filtered.jpg')  
                        # # Encontrar `valid_idx` correspondente ao `gt_idx`
                        # if gt_idx in gt_idx_list:
                        #[ME PARECE ERRADO]
                        # valid_idx = valid_instance_indices[gt_idx_list.index(gt_idx)]
                        #[TESTAR ESSE]
                        # import pdb; pdb.set_trace()
                        valid_idx = valid_instance_indices[gt_idx]

                        # self.double_thr
                        # if allbb_preds_map[img_path][gt_idx]['max_pred'] >= self.double_thr:
                        #     #update
                        #     sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = allbb_preds_map[img_path][gt_idx]['pred_label']
                        # else:    
                            #filtra
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1

                        
                        
                    
                
                            
                            
                        
                    
                    

        print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")

# === Lightweight GNN noise filter hook ===
@HOOKS.register_module()
class MyHookGraphNoiseRelabelFilterGMMSanityDebug(Hook):
    @torch.no_grad()
    def _compute_co_probs(self, device):
        co_counts_cpu = self._cooc_counts.clone().float()
        row_sum = co_counts_cpu.sum(dim=1, keepdim=True).clamp_min(self.cooc_eps)
        co_probs = (co_counts_cpu / row_sum).to(device)
        co_probs.fill_diagonal_(1.0)
        return co_probs
    """
    Antes de cada época, constrói um grafo por imagem usando as detecções atuais,
    executa uma GNN leve para estimar p_ruido por GT e:
      (i) marca instâncias com p_ruido alto como ignore_flag=1, ou
      (ii) opcionalmente relabela para a maioria semântica dos vizinhos confiáveis.

    Parâmetros principais:
      - warmup_epochs: até esse ponto só treina estatísticas, sem filtrar.
      - thr_noise: limiar fixo de p_ruido (0..1) OU usa percentil por classe se use_percentile=True.
      - use_percentile: se True, usa percentil por classe (ex.: 80) como corte dinâmico.
      - k: número de vizinhos no k-NN.
      - do_relabel: se True, tenta relabelar ao invés de ignorar quando houver maioria forte.
      - relabel_min_agree: fração mínima (0..1) de vizinhos confiáveis que concordam.
      - num_classes: nº de classes do seu dataset.
    """
    def __init__(self,
                 warmup_epochs: int = 1,
                 thr_noise: float = 0.7,
                 use_percentile: bool = False,
                 percentile: float = 80.0,
                 k: int = 8,
                 do_relabel: bool = False,
                 relabel_min_agree: float = 0.6,
                 num_classes: int = 20,
                 reload_dataset: bool = True,
                 iou_assigner=0.5,
                 low_quality=False,
                 train_ratio: float = 1.0,
                 k_min: int = 1,
                 k_max: int = 8,
                 k_mode: str = 'sqrt',
                 gnn_lr: float = 1e-3, 
                 gnn_train_steps: int = 50, 
                 tau_gt: float = 0.3, 
                 neighbor_agree: float = 0.6,
                 trust_thr: float = 0.9,
                 corr_thr_low = 0.3,
                 # --- contexto com "gate" ---
                 ctx_conf_beta: float = 2.0,    # peso dos vizinhos por confiança (pmax^beta)
                 ctx_gate_gamma: float = 8.0,   # quão abrupto é o gate sigmoide
                 ctx_dist_sigma: float = 0.75,
                 cong_hidden: int = 128,
                 cong_lr: float = 1e-3,
                 cong_train_steps: int = 100,
                 cong_alpha: float = 0.5,
                 # --- W&B logging params ---
                 use_wandb: bool = True,
                 wandb_project: str = 'noisy-od',
                 wandb_run_name: str = '',
                 wandb_max_images: int = 8,
                 # --- prctx logging knobs ---
                 wandb_log_prctx: bool = True,
                 prctx_topk: int = 5,
                 prctx_max_nodes: int = 3,
                 relabel_thr_ctx: float = 0.7,
                 relabel_thr_high: float = 0.9,
                 # pesos e piso do componente visual no contexto
                 ctx_feat_weight: float = 0.6,
                 ctx_prob_weight: float = 0.4,
                 ctx_sim_min: float = 0.2,
                 filter_thr: float = 0.7,
                 filter_confgmm: float = 0.95
                 ):
        self.warmup_epochs = warmup_epochs
        self.thr_noise = float(thr_noise)
        self.use_percentile = bool(use_percentile)
        self.percentile = float(percentile)
        self.k = int(k)
        self.do_relabel = bool(do_relabel)
        self.relabel_min_agree = float(relabel_min_agree)
        self.num_classes = int(num_classes)
        self.reload_dataset = reload_dataset
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        # será inicializada no primeiro uso para o device correto
        self._gnn = None
        self.train_ratio = float(train_ratio)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.k_mode = str(k_mode)
        self.gnn_lr = float(gnn_lr)
        self.gnn_train_steps = int(gnn_train_steps)
        self.tau_gt = float(tau_gt)
        self.neighbor_agree = float(neighbor_agree)
        self._opt = None  # optimizer da GNN (lazily created)
        self._trained = False  # flag para indicar se a GNN já foi treinada nesta execução
        self.trust_thr = float(trust_thr)
        self.corr_thr_low = corr_thr_low  # limiar para p_ruido só pela co-ocorrência (baixo relacionamento com classes presentes)   
        # knobs de gate de contexto
        self.ctx_conf_beta = float(ctx_conf_beta)
        self.ctx_gate_gamma = float(ctx_gate_gamma)
        self.ctx_dist_sigma = float(ctx_dist_sigma)
        # co-ocorrência (CPU): inicia suavizada para evitar zeros
        self._cooc_counts = torch.ones(self.num_classes, self.num_classes, dtype=torch.float32)
        self.cooc_eps = 1e-3          # suavização mínima ao normalizar (em vez de 1e-6 mencionado)
        self.reset_cooc_each_epoch = True  # se True, zera (para 1s) a cada época
        # ConG (context graph head)
        self.cong_hidden = int(cong_hidden)
        self.cong_lr = float(cong_lr)
        self.cong_train_steps = int(cong_train_steps)
        self.cong_alpha = float(cong_alpha)  # força do reweight: lw = 1 - alpha * p_noise
        self._cong = None
        self._opt_cong = None
        # W&B logging
        self.use_wandb = bool(use_wandb)
        self.wandb_project = str(wandb_project)
        self.wandb_run_name = str(wandb_run_name)
        self.wandb_max_images = int(wandb_max_images)
        self._wandb_ready = False
        self._wandb_img_budget = 0
        self._wandb_imgs = []  # lista de wandb.Image para log único por época
        # --- Phase-1 relabel memory (per epoch, per image) ---
        self._phase1_relabels = {}

        self.wandb_log_prctx = bool(wandb_log_prctx)
        self.prctx_topk = int(prctx_topk)
        self.prctx_max_nodes = int(prctx_max_nodes)

        # W&B extra knobs
        self.wandb_log_if_any_kld = True  # loga imagem se houver qualquer KLD acima do corte, mesmo vetado

        self.relabel_thr_ctx = float(relabel_thr_ctx)
        self.relabel_thr_high = float(relabel_thr_high)
        self.ctx_feat_weight = float(ctx_feat_weight)
        self.ctx_prob_weight = float(ctx_prob_weight)
        self.ctx_sim_min     = float(ctx_sim_min)
        self.filter_thr = float(filter_thr)
        self.filter_confgmm = float(filter_confgmm)
        self.selcand = "max" # max | iou
        self.numGMM = 4
        self.filter_type = "pred" # pred| logit | aps
        

    def _ensure_gnn(self, device):
        if self._gnn is None:
            #self._gnn = GraphNoiseNet(num_classes=self.num_classes, cls_emb_dim=32, prob_dim=64, hidden=128, edge_dim=3).to(device)
            self._gnn = GraphNoiseNet(num_classes=self.num_classes, cls_emb_dim=32, prob_dim=64, hidden=128, edge_dim=2).to(device)
        else:
            self._gnn.to(device)
        self._gnn.eval()  # usamos apenas para scoring no hook
        if self._opt is None:
            self._opt = optim.Adam(self._gnn.parameters(), lr=self.gnn_lr)


    # --- Helper for effective labels from logits and semantic graph ---
    @torch.no_grad()
    def _effective_labels_from_logits(self, node_logits_t: torch.Tensor, gt_labels_t: torch.Tensor, trust_thr: float):
        probs = node_logits_t.softmax(dim=-1)
        pmax, argmax = probs.max(dim=-1)
        eff = gt_labels_t.clone()
        eff[pmax >= trust_thr] = argmax[pmax >= trust_thr]
        return eff, probs, pmax, argmax

    @torch.no_grad()
    def _build_semantic_graph(self, probs: torch.Tensor, eff_labels: torch.Tensor, cooc_probs: torch.Tensor,
                              bboxes_xyxy: torch.Tensor, img_w: int, img_h: int, k: int = 4,
                              features: torch.Tensor = None, feat_weight: float = 0.5, prob_weight: float = 0.5):
        """
        Build a semantic kNN graph using cosine similarity on class-prob vectors and visual features, plus spatial proximity.
        Returns edge_index [2,E] and edge_attr [E,7] = [sim_prob, f_sim, dx, dy, rw, rh, co].
        - sim_prob : similarity between probability distributions (cosine, [0,1])
        - f_sim    : similarity between features (cosine, [0,1])
        - dx,dy    : normalized offsets (src→dst) scaled by dst size
        - rw,rh    : log size ratios (src/dst)
        - co       : co-occurrence prior p(li→lj)
        """
        device = probs.device
        N = probs.size(0)
        if N <= 1:
            return torch.empty(2,0, dtype=torch.long, device=device), torch.empty(0,7, device=device)

        # --- Similaridade entre distribuições de probabilidade (softmax dos logits) ---
        Pn = F.normalize(probs, p=2, dim=-1)
        S_prob = torch.mm(Pn, Pn.t())
        S_prob = (S_prob + 1.0) * 0.5
        S_prob.fill_diagonal_(0.0)

        # --- Similaridade entre features visuais (softmax dos logits ou embeddings) ---
        if features is None:
            # Usa softmax dos logits como features padrão
            features = probs
        Fn = F.normalize(features, p=2, dim=-1)
        S_feat = torch.mm(Fn, Fn.t())
        S_feat = (S_feat + 1.0) * 0.5
        S_feat.fill_diagonal_(0.0)

        # combina do jeito que o chamador pediu (sem renormalizar aqui)
        S = prob_weight * S_prob + feat_weight * S_feat
        S.fill_diagonal_(0.0)

        # top-k by combined similarity
        kk = min(max(1, k), N-1)
        topk = S.topk(kk, dim=1).indices
        src = torch.arange(N, device=device).unsqueeze(1).expand(-1, kk).reshape(-1)
        dst = topk.reshape(-1)

        # spatial attributes from bboxes
        b = bboxes_xyxy
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w  = (x2 - x1).clamp(min=1e-3)
        h  = (y2 - y1).clamp(min=1e-3)

        dx = (cx[src] - cx[dst]) / (w[dst] + 1e-3)
        dy = (cy[src] - cy[dst]) / (h[dst] + 1e-3)
        rw = torch.log(w[src] / w[dst])
        rh = torch.log(h[src] / h[dst])

        sim_prob = S_prob[src, dst].unsqueeze(1)
        f_sim = S_feat[src, dst].unsqueeze(1)
        co  = cooc_probs[eff_labels[src], eff_labels[dst]].unsqueeze(1)

        edge_attr = torch.cat([
            sim_prob,
            f_sim,
            dx.unsqueeze(1), dy.unsqueeze(1),
            rw.unsqueeze(1), rh.unsqueeze(1),
            co
        ], dim=1)
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index, edge_attr

    @torch.no_grad()
    def _make_pseudo_targets(self, node_labels_t, edge_index, node_logits_t, gt_idx_t, cut_prob: float):
        """Constroi y_noise (0/1) por nó.
        Critérios:
            A) p_gt < tau_gt
            B) Fração de vizinhos confiáveis (p_gt >= tau_gt) que concordam com GT é < neighbor_agree
        Marca 1 se (A) OU (B).
        """
        if node_logits_t.numel() == 0:
            return torch.zeros(0, device=node_logits_t.device)
        probs = node_logits_t.softmax(dim=-1)
        idx = torch.arange(probs.size(0), device=probs.device)
        p_gt = probs[idx, gt_idx_t]
        critA = (p_gt < self.tau_gt)

        src, dst = edge_index
        agree_cnt = torch.zeros_like(p_gt)
        total_cnt = torch.zeros_like(p_gt)
        neigh_pgt = p_gt[src]
        neigh_lbl = gt_idx_t[src]
        is_conf = neigh_pgt >= self.tau_gt
        same = (neigh_lbl == gt_idx_t[dst]) & is_conf
        agree_cnt.index_add_(0, dst, same.float())
        total_cnt.index_add_(0, dst, is_conf.float())
        frac_agree = torch.zeros_like(p_gt)
        mask = total_cnt > 0
        frac_agree[mask] = agree_cnt[mask] / total_cnt[mask]
        critB = torch.zeros_like(critA)
        critB[mask] = (frac_agree[mask] < self.neighbor_agree)
        y = (critA | critB).float()
        return y
    
            

    @torch.no_grad()
    def _pnoise_from_cooc(self, eff_labels_t: torch.Tensor, co_probs_dev: torch.Tensor, thr: float):
        """Retorna p_noise em [0,1] por nó baseado APENAS na matriz de co-ocorrência.
        Para cada nó i (classe li), considera as classes dos demais nós j na imagem e
        pega best = max_j co_probs[li, lj]. Se best >= thr → p_noise=0; caso contrário
        p_noise = (thr - best)/thr (quanto menor a co-ocorrência, maior o p_noise).
        """
        N = eff_labels_t.numel()
        if N <= 1:
            return torch.zeros(N, device=eff_labels_t.device)
        li = eff_labels_t                            # [N]
        lj = eff_labels_t                            # [N]
        # matriz cooc para todos pares (i,j)
        C = co_probs_dev[li][:, lj]                  # [N,N]
        # ignora diagonal (relacionar consigo mesmo não conta)
        C = C.masked_fill(torch.eye(N, dtype=torch.bool, device=C.device), 0.0)
        best, _ = C.max(dim=1)                       # [N]
        p = (thr - best).clamp(min=0.0) / max(thr, 1e-6)
        return p
    
    def _ensure_cong(self, device):
        if self._cong is None:
            self._cong = ConG(num_classes=self.num_classes, hidden=self.cong_hidden).to(device)
        else:
            self._cong.to(device)
        if self._opt_cong is None:
            self._opt_cong = optim.AdamW(self._cong.parameters(), lr=self.cong_lr, weight_decay=0.0)

    def before_train_epoch(self, runner):
        if (runner.epoch + 1) <= 0:
            return

        # Reinicia lista de imagens do W&B a cada época
        self._wandb_imgs = []
        dataloader = runner.train_loop.dataloader
        dataset = dataloader.dataset
        # opcional: reiniciar co-ocorrência por época
        if self.reset_cooc_each_epoch:
            self._cooc_counts = torch.ones(self.num_classes, self.num_classes, dtype=torch.float32)

        # --- W&B: init per epoch (lazy) and reset epoch buffers ---
        self._wandb_img_budget = self.wandb_max_images
        self._wandb_imgs = []
        # --- Clear Phase-1 relabel memory for the new epoch ---
        self._phase1_relabels = {}
        if self.use_wandb and (wandb is not None):
            if not self._wandb_ready:
                try:
                    wandb.init(project=self.wandb_project or 'noisy-od',
                               name=self.wandb_run_name or f'run-{os.path.basename(runner.work_dir)}',
                               dir=runner.work_dir,
                               reinit=True)
                    self._wandb_ready = True
                except Exception as _:
                    self._wandb_ready = False

        reload_dataset = self.reload_dataset

        if reload_dataset:
            runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
            runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

            runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
            runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        if not hasattr(dataset, 'datasets'):
            raise ValueError("Esperado um ConcatDataset.")
        datasets = dataset.datasets

        # mapa rápido: img_path -> (sub_dataset_idx, data_idx)
        dataset_img_map = {di['img_path']: (sidx, didx)
                            for sidx, subds in enumerate(datasets)
                            if hasattr(subds, 'data_list')
                            for didx, di in enumerate(subds.data_list)}

        # ----------------- GNN TRAIN/FILTER PHASE (unified pass) -----------------
        # === PHASE 1: build co-occurrence (presence-based) and apply only high-conf relabel ===
        for data_batch in dataloader:
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0 else next(runner.model.parameters()).device)
            self._ensure_cong(device)
            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue
                pred_instances = preds[i].pred_instances.to(device)
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')
                gt = ds.gt_instances.to(device)
                assigner = MaxIoUAssigner(pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner,
                                          min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality)
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue
                logits = pred_instances.logits
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                sub_idx, d_idx = dataset_img_map[img_path]
                subds = datasets[sub_idx]
                valid_instance_indices = [idx for idx, inst in enumerate(subds.data_list[d_idx]['instances']) if inst['ignore_flag'] == 0]

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []
                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    
                    choice = assoc[j]
                    node_labels.append(int(gt.labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)
                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)
                vis_feats = node_logits_t  # usar logits (pré-softmax) como descritor visual
                pmax, pred_cls = pr.max(dim=1)

                # Relabel only by high confidence (no context yet)
                for relabel_idx in range(node_labels_t.shape[0]):
                    if float(pmax[relabel_idx]) >= self.relabel_thr_high:
                        new_lab = int(pred_cls[relabel_idx].item())
                        gt_local_idx = node_img_local_to_valid[relabel_idx]
                        if 0 <= gt_local_idx < len(valid_instance_indices):
                            valid_idx = valid_instance_indices[gt_local_idx]
                            old_lab = int(node_labels_t[relabel_idx].item())
                            # Atualiza tensores/listas deste batch
                            node_labels_t[relabel_idx] = new_lab
                            inst = subds.data_list[d_idx]['instances'][valid_idx]
                            updated = False
                            for key in ['labels','label','bbox_label']:
                                if key in inst:
                                    inst[key] = new_lab
                                    updated = True
                            if not updated:
                                inst['labels'] = new_lab
                            # Memoriza para overlay na Fase 3 (por imagem, indexado pelo GT local)
                            if img_path not in self._phase1_relabels:
                                self._phase1_relabels[img_path] = {}
                            # chave: índice local de GT usado para construir os nós (node_img_local_to_valid)
                            self._phase1_relabels[img_path][int(gt_local_idx)] = int(old_lab)
                        else:
                            if hasattr(runner, 'logger'):
                                runner.logger.warning(f"[Phase1] Skip relabel write: gt_local_idx={gt_local_idx} out of range for {os.path.basename(img_path)}")

                # Presence-based co-occurrence update (after relabel-high)
                eff_labels_t = node_labels_t.clone()
                if eff_labels_t.numel() > 1:
                    uniq = torch.unique(eff_labels_t.detach().to(torch.long))
                    if uniq.numel() > 1:
                        ui = uniq.unsqueeze(1).expand(-1, uniq.numel())
                        uj = uniq.unsqueeze(0).expand(uniq.numel(), -1)
                        mask = (ui != uj)
                        pairs_i = ui[mask].reshape(-1).to('cpu', dtype=torch.long)
                        pairs_j = uj[mask].reshape(-1).to('cpu', dtype=torch.long)
                        self._cooc_counts.index_put_((pairs_i, pairs_j), torch.ones_like(pairs_i, dtype=torch.float32), accumulate=True)

        # Freeze epoch prior
        # epoch_device = runner.model.device
        epoch_device = next(runner.model.parameters()).device
        co_probs_epoch = self._compute_co_probs(epoch_device)

        # === PHASE 2: Train ConG using the fixed co_probs_epoch ===
        steps = 0
        for data_batch in dataloader:
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0 else next(runner.model.parameters()).device)
            self._ensure_cong(device)
            batch_loss = None
            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue
                pred_instances = preds[i].pred_instances.to(device)
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')
                gt = ds.gt_instances.to(device)
                assigner = MaxIoUAssigner(pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner,
                                          min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality)
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue
                logits = pred_instances.logits
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []
                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    choice = assoc[j]
                    node_labels.append(int(gt.labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)
                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)
                # usar logits (pré-softmax) como descritor visual
                vis_feats = node_logits_t

                # Build context inputs
                H_img = inputs.shape[-2]
                W_img = inputs.shape[-1]
                gt_boxes_xyxy = gt.bboxes.tensor
                sel_gt_xyxy = gt_boxes_xyxy[node_img_local_to_valid]
                sp = torch.stack([spatial7_from_xyxy(sel_gt_xyxy[k], W_img, H_img) for k in range(sel_gt_xyxy.size(0))], dim=0)
                # === Consistência de dimensões entre pr/vis_feats/labels/sp/boxes ===
                N_pr   = pr.size(0)
                N_feat = vis_feats.size(0)
                N_lab  = node_labels_t.size(0)
                N_box  = sel_gt_xyxy.size(0)
                N_sp   = sp.size(0)
                min_n = min(N_pr, N_feat, N_lab, N_box, N_sp)
                if not (N_pr == N_feat == N_lab == N_box == N_sp):
                    pr = pr[:min_n]
                    vis_feats = vis_feats[:min_n]
                    node_logits_t = node_logits_t[:min_n]
                    node_labels_t = node_labels_t[:min_n]
                    sel_gt_xyxy = sel_gt_xyxy[:min_n]
                    sp = sp[:min_n]
                Nn = pr.size(0)
                if Nn <= 1:
                    pr_ctx = pr
                else:
                    probs = pr
                    # confidences (neighbors j)
                    pmax = probs.max(dim=-1).values
                    w_conf = (pmax.clamp_min(1e-6) ** self.ctx_conf_beta)
                    # cosine similarity nas probabilidades
                    Pn = F.normalize(pr, p=2, dim=-1)
                    S_prob = torch.mm(Pn, Pn.t())
                    S_prob = (S_prob + 1.0) * 0.5
                    # cosine similarity visual (logits)
                    Fn = F.normalize(vis_feats, p=2, dim=-1)
                    S_feat = torch.mm(Fn, Fn.t())
                    S_feat = (S_feat + 1.0) * 0.5
                    # combinação com pesos configuráveis (com ajuste defensivo de tamanho)
                    if S_prob.shape != S_feat.shape:
                        min_n = min(S_prob.shape[0], S_feat.shape[0])
                        S_prob = S_prob[:min_n, :min_n]
                        S_feat = S_feat[:min_n, :min_n]
                    S = self.ctx_prob_weight * S_prob + self.ctx_feat_weight * S_feat
                    S.fill_diagonal_(0.0)
                    # máscara para suprimir pares visualmente dissimilares
                    S_mask = (S_feat >= self.ctx_sim_min).float()
                    # spatial Gaussian kernel using dest box sizes
                    b = sel_gt_xyxy
                    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
                    cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                    w  = (x2-x1).clamp(min=1e-3); h=(y2-y1).clamp(min=1e-3)
                    DX = (cx.unsqueeze(1) - cx.unsqueeze(0)).abs() / (w.unsqueeze(0) + 1e-3)
                    DY = (cy.unsqueeze(1) - cy.unsqueeze(0)).abs() / (h.unsqueeze(0) + 1e-3)
                    D  = torch.sqrt(DX**2 + DY**2)
                    sigma = self.ctx_dist_sigma
                    Ksp = torch.exp(-D / max(1e-6, sigma))
                    # co-occurrence attenuation between labels
                    li = node_labels_t
                    Cij = co_probs_epoch.to(device)[li][:, li]
                    # final weights W[i,j] (j contributes to i)
                    W = (S * Ksp * Cij) * S_mask
                    W = W * w_conf.unsqueeze(0)
                    W.fill_diagonal_(0.0)
                    den = W.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    pr_ctx = (W @ probs) / den
                pmax = pr.max(dim=-1).values
                alpha_conf = torch.sigmoid(self.ctx_gate_gamma * (pmax - self.trust_thr)).unsqueeze(1)
                alpha = alpha_conf
                pr_mix = alpha * pr + (1.0 - alpha) * pr_ctx
                x_cong = torch.cat([pr_mix, sp], dim=1)

                # Graph with fixed epoch prior
                co_probs = co_probs_epoch.to(device)
                edge_index, edge_attr = self._build_semantic_graph(
                    pr, node_labels_t, co_probs_epoch.to(device), sel_gt_xyxy, W_img, H_img, k=4,
                    features=vis_feats, feat_weight=self.ctx_feat_weight, prob_weight=self.ctx_prob_weight)

                # Train ConG
                self._cong.train()
                logits_cong = self._cong(x_cong, edge_index, edge_attr)
                loss_cong = F.cross_entropy(logits_cong, node_labels_t)
                if batch_loss is None:
                    batch_loss = loss_cong
                else:
                    batch_loss = batch_loss + loss_cong

            if batch_loss is not None:
                self._opt_cong.zero_grad()
                batch_loss.backward()
                self._opt_cong.step()
                steps += 1
                if steps >= self.cong_train_steps:
                    break
        if steps >= self.cong_train_steps and hasattr(runner, 'logger'):
            runner.logger.info(f"[ConG] Trained with fixed epoch co_probs: steps={steps}")

        meu_contador=0
        for data_batch in dataloader:
            # forward do detector SEM grad (apenas extrair preds)
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0 else next(runner.model.parameters()).device)

            self._ensure_cong(device)
            co_probs = self._compute_co_probs(device)
            # ===== por-amostra do batch =====
            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue
                # garanta que TUDO está no mesmo device
                pred_instances = preds[i].pred_instances.to(device)
                # alguns heads expõem 'bboxes'; o assigner aceita 'priors' também
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')

                gt = ds.gt_instances.to(device)

                # referências locais (já no device)
                bboxes = gt.bboxes
                labels = gt.labels
                priors = pred_instances.priors
                logits = pred_instances.logits  # [Np, C]
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                # assign predições -> GTs
                assigner = MaxIoUAssigner(pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner, min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality)
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []

                # mapeamento de instâncias válidas no dataset (para escrita)
                sub_idx, d_idx = dataset_img_map[img_path]
                subds = datasets[sub_idx]
                valid_instance_indices = [idx for idx, inst in enumerate(subds.data_list[d_idx]['instances']) if inst['ignore_flag'] == 0]

                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    choice = assoc[j]
                    node_labels.append(int(labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)
                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)  # [N,C]
                vis_feats = node_logits_t
                # veto: se a classe predita == GT, não considerar como ruído
                pred_cls_t = pr.argmax(dim=1)                  # [N]
                veto_eq_t = (pred_cls_t == node_labels_t)      # [N] bool



                # rótulos efetivos para consultar co-ocorrência (usa pred quando muito confiante)
                eff_labels_t, _, pmax_t, argmax_t = self._effective_labels_from_logits(node_logits_t, node_labels_t, self.trust_thr)

                H_img = inputs.shape[-2]
                W_img = inputs.shape[-1]
                gt_boxes_xyxy = gt.bboxes.tensor
                sel_gt_xyxy = gt_boxes_xyxy[node_img_local_to_valid]
                sp_list = [spatial7_from_xyxy(sel_gt_xyxy[k], W_img, H_img) for k in range(sel_gt_xyxy.size(0))]
                if len(sp_list) == 0:
                    continue
                sp = torch.stack(sp_list, dim=0)  # [N,7]

                # --- contexto ponderado por confiança + gate por confiança e minoritarismo ---
                Nn = pr.size(0)
                # cosine similarities: probs e visual (logits)
                Pn = F.normalize(pr, p=2, dim=-1)
                S_prob = torch.mm(Pn, Pn.t()); S_prob = (S_prob + 1.0) * 0.5
                Fn = F.normalize(vis_feats, p=2, dim=-1)
                S_feat = torch.mm(Fn, Fn.t()); S_feat = (S_feat + 1.0) * 0.5

                # combinação com pesos configuráveis (com ajuste defensivo de tamanho)
                if S_prob.shape != S_feat.shape:
                    min_n = min(S_prob.shape[0], S_feat.shape[0])
                    S_prob = S_prob[:min_n, :min_n]
                    S_feat = S_feat[:min_n, :min_n]
                S = self.ctx_prob_weight * S_prob + self.ctx_feat_weight * S_feat
                S.fill_diagonal_(0.0)
                # gate visual
                S_mask = (S_feat >= self.ctx_sim_min).float()

                # usar logits como features visuais adiante
                features = vis_feats
                if Nn <= 1:
                    pr_ctx = pr
                else:
                    probs = pr
                    pmax = probs.max(dim=-1).values
                    w_conf = (pmax.clamp_min(1e-6) ** self.ctx_conf_beta)
                    b = sel_gt_xyxy
                    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
                    cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                    w  = (x2-x1).clamp(min=1e-3); h=(y2-y1).clamp(min=1e-3)
                    DX = (cx.unsqueeze(1) - cx.unsqueeze(0)).abs() / (w.unsqueeze(0) + 1e-3)
                    DY = (cy.unsqueeze(1) - cy.unsqueeze(0)).abs() / (h.unsqueeze(0) + 1e-3)
                    D  = torch.sqrt(DX**2 + DY**2)
                    sigma = self.ctx_dist_sigma
                    Ksp = torch.exp(-D / max(1e-6, sigma))
                    li = eff_labels_t
                    Cij = co_probs[li][:, li] if (li.numel() == Nn) else torch.ones_like(S)
                    # Peso final: combinação da similaridade de features, probabilidade e co-ocorrência
                    W = (S * Ksp * Cij) * S_mask
                    W = W * w_conf.unsqueeze(0)
                    W.fill_diagonal_(0.0)
                    den = W.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    pr_ctx = (W @ probs) / den
                # gate: self vs contexto condicionado por confiança e minoritarismo
                probs = pr
                pmax = probs.max(dim=-1).values
                alpha_conf = torch.sigmoid(self.ctx_gate_gamma * (pmax - self.trust_thr)).unsqueeze(1)
                alpha = alpha_conf
                pr_mix = alpha * probs + (1.0 - alpha) * pr_ctx
                x_cong = torch.cat([pr_mix, sp], dim=1)
                # --- Build semantic graph with co-occurrence attributes and feature similarity ---
                # edge_attr agora inclui f_sim como segunda coluna
                edge_index, edge_attr = self._build_semantic_graph(
                    pr, eff_labels_t, co_probs, sel_gt_xyxy, W_img, H_img, k=4,
                    features=vis_feats, feat_weight=self.ctx_feat_weight, prob_weight=self.ctx_prob_weight
                )
                self._cong.eval()
                with torch.no_grad():
                    logits_cong = self._cong(x_cong, edge_index, edge_attr)
                    qc = logits_cong.softmax(dim=-1)  # [N,C]

                old_labels_before_agree = node_labels_t.clone()
                relabeled_pairs = []  # lista de tuples (local_idx, old_gt)
                # --- AGREEMENT RELABEL (modelo & contexto concordam) ---
                if qc.shape[0] == pr.shape[0] and pr.shape[0] > 1:
                    pmax_t = pr.max(dim=1).values               # confiança do modelo por nó
                    pred_cls_t = pr.argmax(dim=1)               # classe do modelo por nó
                    ctx_cls_t = qc.argmax(dim=1)                # classe de contexto por nó
                    agree = (pred_cls_t == ctx_cls_t) & (pmax_t >= self.relabel_thr_ctx)
                    agree_idx = torch.nonzero(agree, as_tuple=False).flatten()
                    for _li in agree_idx.tolist():
                        new_lab = int(pred_cls_t[_li].item())
                        old_lab = int(node_labels_t[_li].item())
                        if new_lab == old_lab:
                            # nada a fazer; não marca como relabel
                            continue
                        # atualiza o tensor de labels do batch
                        
                        node_labels_t[_li] = new_lab
                        
                        relabeled_pairs.append((_li, old_lab))
                        # persiste no dataset para próximas épocas
                        if _li < len(node_img_local_to_valid):
                            gt_idx = node_img_local_to_valid[_li]
                            if gt_idx < len(valid_instance_indices):
                                _valid_idx = valid_instance_indices[gt_idx]
                                inst = subds.data_list[d_idx]['instances'][_valid_idx]
                                updated = False
                                for key in ['bbox_label', 'label', 'labels']:
                                    if key in inst:
                                        # inst[key] = new_lab
                                        inst[key] = 99
                                        updated = True
                                        if meu_contador<3:
                                            meu_contador+=1
                                            print(f"sub_idx={sub_idx} d_idx={d_idx} valid_idx={_valid_idx} gt_local_idx={gt_idx} key={key} old_lab={old_lab} new_lab={new_lab} img_path={img_path}")
                                            import pdb; pdb.set_trace()
                                # if not updated:
                                #     inst['labels'] = new_lab
                                #     #debug filipe
                                #     # inst['labels'] = 99
                                #     import pdb; pdb.set_trace()
                    # Como alteramos node_labels_t, recomputamos eff_labels_t e veto_eq_t
                    # eff_labels_t, _, pmax_t, argmax_t = self._effective_labels_from_logits(node_logits_t, node_labels_t, self.trust_thr)
                    veto_eq_t = (pred_cls_t == node_labels_t)
                    disagree_mask = (pred_cls_t != node_labels_t).detach().cpu().numpy().astype(bool)
                # else:
                #     if hasattr(runner, 'logger'):
                #         runner.logger.warning(f"[ConG] Skip agreement relabel: "
                #                               f"N={pr.shape[0]}, qc={qc.shape[0]} pr={pr.shape[0]} on {os.path.basename(img_path)}")

                eps = 1e-7
                p = pr.clamp_min(eps)
                q = qc.clamp_min(eps)
                # --- filtro contextual + co-ocorrência real ---
                kld = (p * (p.log() - q.log())).sum(dim=1)  # [N]
                kld_np = kld.detach().cpu().numpy()
                # usa KLD absoluto como p_noise (sem normalização por imagem)
                p_noise = kld
                p_noise_np = kld_np

                # --- filtro auxiliar de co-ocorrência real ---
                # para cada nó i (classe li), verifica se existe alguma classe w presente na imagem tal que co(li,w) >= corr_thr_low
                Nn = eff_labels_t.numel()
                if Nn > 1:
                    li = eff_labels_t
                    lj = eff_labels_t
                    C = co_probs[li][:, lj]  # [N,N]
                    C = C.masked_fill(torch.eye(Nn, dtype=torch.bool, device=C.device), 0.0)
                    best, _ = C.max(dim=1)   # [N]
                    best_np = best.detach().cpu().numpy()
                    low_corr_mask = (best < float(self.corr_thr_low))  # True quando baixa correlação com classes presentes
                else:
                    low_corr_mask = torch.zeros_like(kld, dtype=torch.bool)
                low_corr_np = low_corr_mask.detach().cpu().numpy().astype(bool)

                # define corte da parte contextual (KLD normalizado)
                if self.use_percentile and p_noise_np.size > 0:
                    cut = float(np.percentile(p_noise_np, self.percentile))
                else:
                    cut = float(self.thr_noise)

                

                # veto adicional: predição igual ao GT nunca é ruído
                veto_eq_np = veto_eq_t.detach().cpu().numpy().astype(bool)

                # --- W&B: loga alguns exemplos visuais ---
                if self.use_wandb and (wandb is not None):
                    # loga até `wandb_max_images` imagens por época com overlay
                    if self._wandb_img_budget > 0:
                        # Log only if at least one bbox had its label actually changed in this image
                        relabeled_pairs = relabeled_pairs if 'relabeled_pairs' in locals() else []
                        has_relabel = len(relabeled_pairs) > 0
                        should_log = has_relabel
                        try:
                            relabeled_pairs = relabeled_pairs if 'relabeled_pairs' in locals() else []
                            relabeled_set = set([int(i) for (i, _) in relabeled_pairs]) if 'relabeled_pairs' in locals() else set()
                            old_gt_map = {int(i): int(old) for (i, old) in relabeled_pairs} if 'relabeled_pairs' in locals() else {}
                            # --- Merge with Phase-1 relabels (pink) ---
                            phase1_map_for_img = self._phase1_relabels.get(img_path, {})  # {gt_local_idx -> old_lab}
                            relabel_high_set = set()
                            old_gt_map_phase1 = {}
                            # Mapear do índice de nó local (local_idx) para info da Fase-1 via gt_local_idx
                            for local_idx, gt_local_idx in enumerate(node_img_local_to_valid[:len(sel_gt_xyxy)]):
                                if int(gt_local_idx) in phase1_map_for_img:
                                    relabel_high_set.add(int(local_idx))
                                    old_gt_map_phase1[int(local_idx)] = int(phase1_map_for_img[int(gt_local_idx)])
                            # noisy se (KLD alto) E (baixa co-ocorrência real)
                            noisy_kld = (p_noise_np >= cut)
                            noisy_mask = np.logical_and(noisy_kld, low_corr_np)
                            # aplica vetos: minoria plausível e pred==GT
                            # if 'veto_np' in locals() and len(veto_np) == len(noisy_mask):
                            #     noisy_mask = np.logical_and(noisy_mask, np.logical_not(veto_np))
                            if 'veto_eq_np' in locals() and len(veto_eq_np) == len(noisy_mask):
                                noisy_mask = np.logical_and(noisy_mask, np.logical_not(veto_eq_np))
                            if should_log:
                                # reconstrói imagem BGR para desenho usando img_norm_cfg real
                                # Carrega a imagem original em BGR e redimensiona para o img_shape atual do batch
                                # (mantém cores corretas para desenho via OpenCV)
                                meta = getattr(data_samples[i], 'metainfo', {})
                                img_shape = tuple(meta.get('img_shape', (inputs.shape[-2], inputs.shape[-1])))  # (H, W, 3)
                                H_v, W_v = int(img_shape[0]), int(img_shape[1])
                                img_np = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
                                if img_np is None:
                                    # fallback: tenta usar o tensor (pode ter cores trocadas)
                                    norm_cfg = meta.get('img_norm_cfg', {})
                                    mean = norm_cfg.get('mean', [123.675, 116.28, 103.53])
                                    std = norm_cfg.get('std', [58.395, 57.12, 57.375])
                                    to_rgb = norm_cfg.get('to_rgb', True)
                                    img_np = tensor_to_numpy_img(inputs[i].cpu(), mean=mean, std=std, to_rgb=to_rgb)
                                else:
                                    img_np = cv2.resize(img_np, (W_v, H_v), interpolation=cv2.INTER_LINEAR)
                                    # aplica o mesmo flip da pipeline para alinhar as boxes
                                    if bool(meta.get('flip', False)):
                                        fd = str(meta.get('flip_direction', 'horizontal'))
                                        if 'diagonal' in fd:
                                            img_np = cv2.flip(img_np, -1)
                                        else:
                                            if 'horizontal' in fd:
                                                img_np = cv2.flip(img_np, 1)
                                            if 'vertical' in fd:
                                                img_np = cv2.flip(img_np, 0)
                                # desenha GTs associados com cor conforme noise
                                sel_gt_xyxy_np = sel_gt_xyxy.detach().cpu().numpy().astype(int)
                                for local_idx in range(min(len(node_img_local_to_valid), len(p_noise_np))):
                                    x1, y1, x2, y2 = sel_gt_xyxy_np[local_idx].tolist()
                                    is_noisy = bool(noisy_mask[local_idx])
                                    is_kld = bool(noisy_kld[local_idx])
                                    # prioridade de cor: relabel (AZUL/Fase-2) > relabel-high (ROSA/Fase-1) > noisy (VERMELHO) > KLD alto vetado (LARANJA) > discordância (AMARELO) > limpo (VERDE)
                                    if local_idx in relabeled_set:
                                        color = (255, 0, 0)            # AZUL (reanotado pela rede de grafos / Phase-2)
                                    elif local_idx in relabel_high_set:
                                        color = (255, 105, 180)        # ROSA (reanotado na Phase-1: alta confiança do modelo)
                                    elif 'disagree_mask' in locals() and local_idx < len(disagree_mask) and bool(disagree_mask[local_idx]):
                                        color = (0, 255, 255)          # AMARELO (pred != label atual)
                                    else:
                                        color = (0, 255, 0)            # VERDE

                                    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
                                    # obter classes GT/pred, KLD e pmax para overlay
                                    _gt_lab = int(node_labels[local_idx])
                                    _pred_lab = int(pr[local_idx].argmax().item())
                                    _kld = float(kld_np[local_idx])
                                    _pmax = float(pr[local_idx].max().item())
                                    _co = float(best_np[local_idx]) if 'best_np' in locals() and local_idx < len(best_np) else 0.0
                                    # Adiciona classe esperada pelo contexto (ctx) usando qc.argmax(dim=1)
                                    _ctx_lab = int(qc[local_idx].argmax().item()) if qc is not None and local_idx < qc.shape[0] else -1
                                    _v_eq = bool(veto_eq_np[local_idx]) if 'veto_eq_np' in locals() and local_idx < len(veto_eq_np) else False
                                    _old2 = old_gt_map.get(local_idx, None)                 # Phase-2 old label (ConG)
                                    _old1 = old_gt_map_phase1.get(local_idx, None)          # Phase-1 old label (high confidence)
                                    # Preferir mostrar o marcador correspondente à cor aplicada
                                    if local_idx in relabeled_set and _old2 is not None:
                                        rx_suffix = f"|R{_old2}"
                                    elif local_idx in relabel_high_set and _old1 is not None:
                                        rx_suffix = f"|R{_old1}"
                                    else:
                                        rx_suffix = ""
                                    _overlay_txt = f"gt{_gt_lab}|p{_pred_lab}|n{float(p_noise_np[local_idx]):.2f}|k{_kld:.2f}|s{_pmax:.2f}|c{_co:.2f}|ct{_ctx_lab}{rx_suffix}"
                                    cv2.putText(img_np,
                                                _overlay_txt,
                                                (max(0, x1), max(15, y1-4)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                # Converte BGR (OpenCV) → RGB antes de enviar ao W&B
                                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                                self._wandb_imgs.append(
                                    wandb.Image(img_rgb, caption=f"epoch={runner.epoch+1} | {os.path.basename(img_path)}")
                                )
                                self._wandb_img_budget -= 1
                                
                                # Extra: logar pr/pr_ctx/qc para casos com discordância modelo x label
                                if getattr(self, 'wandb_log_prctx', False) and 'disagree_mask' in locals():
                                    disagree_idx = np.where(disagree_mask)[0].tolist()
                                    if len(disagree_idx) > 0:
                                        disagree_idx = disagree_idx[:max(1, getattr(self, 'prctx_max_nodes', 8))]
                                        idx_t2 = torch.as_tensor(disagree_idx, device=pr.device, dtype=torch.long)
                                        pr_sel = pr.index_select(0, idx_t2)
                                        pr_ctx_sel = pr_ctx.index_select(0, idx_t2)
                                        qc_sel = qc.index_select(0, idx_t2) if qc is not None else None
                                        labels_sel = node_labels_t.index_select(0, idx_t2)

                                        cls_names = None
                                        try:
                                            cls_names = data_samples[i].metainfo.get('classes', None)
                                        except Exception:
                                            cls_names = None

                                        title2 = f"pr vs pr_ctx vs qc (pred!=label) | {os.path.basename(img_path)}"
                                        prctx_img2 = _make_prctx_figure(pr_sel, pr_ctx_sel, qc_sel, labels_sel,
                                                                        topk=getattr(self, 'prctx_topk', 5),
                                                                        max_nodes=len(disagree_idx),
                                                                        class_names=cls_names,
                                                                        title=title2)
                                        if prctx_img2 is not None:
                                            self._wandb_imgs.append(
                                                wandb.Image(prctx_img2, caption=f"pr_ctx disagree | epoch={runner.epoch+1} | {os.path.basename(img_path)}")
                                            )
                        except Exception as e:
                            if hasattr(runner, 'logger'):
                                runner.logger.warning(f"[W&B] Falha ao montar/registrar imagem: {e}")

                # aplica REWEIGHT (ignore_flag comentado)
                L = min(len(node_img_local_to_valid), len(p_noise_np))
                for local_idx in range(L):
                    gt_idx = node_img_local_to_valid[local_idx]
                    valid_idx = valid_instance_indices[gt_idx] if gt_idx < len(valid_instance_indices) else None
                    if valid_idx is None:
                        continue
                    if (runner.epoch + 1) <= self.warmup_epochs:
                        continue
                    is_lowcorr = bool(low_corr_np[local_idx])
                    # is_veto = bool(veto_np[local_idx]) if 'veto_np' in locals() and local_idx < len(veto_np) else False
                    # is_veto_eq = bool(veto_eq_np[local_idx]) if 'veto_eq_np' in locals() and local_idx < len(veto_eq_np) else False
                    #if (float(p_noise_np[local_idx]) >= cut) and is_lowcorr and (not is_veto) and (not is_veto_eq):
                    # if (float(p_noise_np[local_idx]) >= cut) and is_lowcorr and (not is_veto_eq):
                        # lw = max(0.2, 1.0 - self.cong_alpha * float(p_noise_np[local_idx]))
                        # subds.data_list[d_idx]['instances'][valid_idx]['loss_weight'] = lw
                        # pass
                        # subds.data_list[d_idx]['instances'][valid_idx]['ignore_flag'] = 1  # ignorar apenas quando também houver baixa co-ocorrência
                    
                    

          

        # Salvar matriz de co-ocorrência como imagem de heatmap usando matplotlib
        try:
            # 1) Salvar heatmap da co-ocorrência
            cooc_dir = os.path.join(runner.work_dir, 'debug_cooc')
            os.makedirs(cooc_dir, exist_ok=True)
            cooc_path = os.path.join(cooc_dir, f'cooc_matrix_epoch{runner.epoch + 1}.png')
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patheffects as _pe
                plt.figure(figsize=(6, 5))
                _co = self._cooc_counts.clone().float()
                _row = _co.sum(dim=1, keepdim=True).clamp_min(1e-6)
                co_probs = (_co / _row)
                co_probs.fill_diagonal_(1.0)
                _vis = co_probs.numpy()
                im = plt.imshow(co_probs.cpu(), cmap='viridis')
                plt.title(f"Co-occurrence Matrix - Epoch {runner.epoch + 1}")
                plt.colorbar(im)
                plt.xlabel("j (given i→j)")
                plt.ylabel("i")

                # Ajusta os eixos para valores inteiros
                n_classes = co_probs.shape[0]
                plt.xticks(range(n_classes), range(n_classes))
                plt.yticks(range(n_classes), range(n_classes))
                plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

                # === NEW: escreve o número de ocorrências (contagem bruta) em cada célula ===
                _co_np = _co.detach().cpu().numpy()
                for i_tick in range(n_classes):
                    for j_tick in range(n_classes):
                        cnt = int(_co_np[i_tick, j_tick])
                        if cnt <= 0:
                            continue
                        # escolhe cor do texto conforme intensidade para contraste
                        txt_color = 'white' if _vis[i_tick, j_tick] > 0.2 else 'white'
                        plt.text(j_tick, i_tick, str(cnt), ha='center', va='center', fontsize=6,
                                 color=txt_color, path_effects=[_pe.withStroke(linewidth=1, foreground='black')])

                plt.tight_layout()
                plt.savefig(cooc_path)
                plt.close()
            except Exception as e_heat:
                if hasattr(runner, 'logger'):
                    runner.logger.warning(f"[Cooc] Falha ao salvar heatmap: {e_heat}")

            # 2) Forçar log no W&B (imagem, tabela e heatmap interativo)
            if getattr(self, 'use_wandb', False) and (wandb is not None) and getattr(self, '_wandb_ready', False):
                log_dict = {}
                if hasattr(self, '_wandb_imgs') and len(self._wandb_imgs) > 0:
                    log_dict['debug_imgs'] = self._wandb_imgs

                # Recalcula co_probs normalizado (CPU→numpy) para tabela/heatmap
                _co_counts = self._cooc_counts.clone().float()
                _row = _co_counts.sum(dim=1, keepdim=True).clamp_min(1e-6)
                _co_probs = (_co_counts / _row)
                _co_probs.fill_diagonal_(1.0)
                _co_probs_np = _co_probs.detach().cpu().numpy()

                # Tenta obter nomes de classes
                class_names = None
                try:
                    # tenta via datasets concatenados
                    while hasattr(dataset, 'dataset'):
                        dataset_ = dataset.dataset
                    # fallback: tenta pelo primeiro subdataset
                    if 'datasets' in dir(dataset) and len(dataset.datasets) > 0:
                        sub0 = dataset.datasets[0]
                        if hasattr(sub0, 'METAINFO') and isinstance(sub0.METAINFO, dict):
                            class_names = sub0.METAINFO.get('classes', None)
                except Exception:
                    class_names = None

                # Constrói Tabela (i,j,value) para o heatmap
                data_rows = []
                C = int(_co_probs_np.shape[0])
                for i in range(C):
                    for j in range(C):
                        name_i = class_names[i] if (class_names is not None and i < len(class_names)) else str(i)
                        name_j = class_names[j] if (class_names is not None and j < len(class_names)) else str(j)
                        data_rows.append([name_i, name_j, float(_co_probs_np[i, j])])
                cooc_table = wandb.Table(data=data_rows, columns=["i", "j", "value"])  # tabela completa

                # Heatmap interativo no W&B
                try:
                    cooc_heatmap = wandb.plot.heatmap(cooc_table, x="j", y="i", value="value",
                                                      title=f"Co-occurrence (row-normalized) - Epoch {runner.epoch + 1}")
                    log_dict['cooc_heatmap'] = cooc_heatmap
                except Exception as e_hm:
                    if hasattr(runner, 'logger'):
                        runner.logger.warning(f"[W&B] Falha ao criar heatmap: {e_hm}")

                # === NEW: também loga a matriz de contagens brutas ===
                _co_counts_np = _co_counts.detach().cpu().numpy()
                data_rows_cnt = []
                for i_idx in range(C):
                    for j_idx in range(C):
                        name_i = class_names[i_idx] if (class_names is not None and i_idx < len(class_names)) else str(i_idx)
                        name_j = class_names[j_idx] if (class_names is not None and j_idx < len(class_names)) else str(j_idx)
                        data_rows_cnt.append([name_i, name_j, int(_co_counts_np[i_idx, j_idx])])
                cooc_counts_table = wandb.Table(data=data_rows_cnt, columns=["i", "j", "count"])  # contagens
                try:
                    cooc_counts_heatmap = wandb.plot.heatmap(cooc_counts_table, x="j", y="i", value="count",
                                                             title=f"Co-occurrence COUNTS - Epoch {runner.epoch + 1}")
                    log_dict['cooc_counts_heatmap'] = cooc_counts_heatmap
                except Exception as e_hm2:
                    if hasattr(runner, 'logger'):
                        runner.logger.warning(f"[W&B] Falha ao criar heatmap de contagens: {e_hm2}")
                log_dict['cooc_counts_table'] = cooc_counts_table

                # Anexa imagem PNG gerada
                if os.path.exists(cooc_path):
                    log_dict['cooc_matrix'] = wandb.Image(cooc_path)
                # Anexa a tabela numérica
                log_dict['cooc_table'] = cooc_table

                if len(log_dict) > 0:
                    wandb.log(log_dict, commit=True)
                    # esvazia o buffer de imagens para próxima época
                    if 'debug_imgs' in log_dict:
                        self._wandb_imgs.clear()
                    if hasattr(runner, 'logger'):
                        runner.logger.info(f"[W&B] Imagens, cooc_table e cooc_heatmap logados na epoch {runner.epoch + 1}")
        except Exception as e_final:
            if hasattr(runner, 'logger'):
                runner.logger.warning(f"[W&B] Falha no log final da epoch: {e_final}")

        
        # === PHASE 3: GMM-based noise filtering (after relabeling) ===
        # Faz um processo em duas passagens:
        #   Passo 3A) varre toda a época e acumula scores por classe (p_gt)
        #   Passo 3B) ajusta um GMM por classe usando TODOS os scores acumulados e marca low-confidence como noisy
        dataloader = runner.train_loop.dataloader
        dataset = dataloader.dataset

        #reload_dataset = True
        # my_value = 3
        #reload_dataset = getattr(runner.cfg, 'reload_dataset', False)  
        #my_value =  getattr(runner.cfg, 'my_value', 10)  
        # reload_dataset = self.reload_dataset
        # relabel_conf = self.relabel_conf
        
        # # import pdb; pdb.set_trace()
        # if reload_dataset:
        #     runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
        #     runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

        #     runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
        #     runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

        # Desencapsular RepeatDataset até o ConcatDataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        if not hasattr(dataset, 'datasets'):
            raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

        datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

        assigner = MaxIoUAssigner(
            pos_iou_thr=self.iou_assigner,
            neg_iou_thr=self.iou_assigner,
            min_pos_iou=self.iou_assigner,
            match_low_quality=self.low_quality
        )

        # Criar um dicionário para acesso rápido ao índice das imagens no dataset
        dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                        for sub_dataset_idx, sub_dataset in enumerate(datasets)
                        if hasattr(sub_dataset, 'data_list')
                        for data_idx, data_info in enumerate(sub_dataset.data_list)}
        import pdb; pdb.set_trace()
        # Dicionário para armazenar os scores médios por GT sem assumir sequência
        allbb_preds_map = defaultdict(dict)
        all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
        all_pred_instances_map = defaultdict(dict)
        # all_inputs_map = defaultdict(dict)

        num_classes = 20
        all_classes_preds_map=[]
        all_classes_gt_idx_map=[]
        for i in range(num_classes):
            all_classes_preds_map.append(defaultdict(list))
            all_classes_gt_idx_map.append(defaultdict(list))
        

        count_gts = 0
        sanity_count = 0
        global_index_counter = 0

        
        # Processar o batch de dados
        for batch_idx, data_batch in enumerate(dataloader):
            with torch.no_grad():
                #data = runner.model.data_preprocessor(data_batch, False)
                data = runner.model.data_preprocessor(data_batch, True)
                
                inputs = data['inputs']
                data_samples = data['data_samples']
                
                #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
            
                #temp_filipe

            for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
            # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                img_path = data_sample.img_path
                
                if img_path not in dataset_img_map:
                    print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                    continue

                # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                pred_instances = predictions_pred[i].pred_instances

                # Pegar bounding boxes preditas
                pred_instances.priors = pred_instances.pop('bboxes')

                # Obter os ground truths
                gt_instances = data_sample.gt_instances
                gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                # Garantir que os GTs estão no mesmo dispositivo das predições
                gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                # Garantir que as predições também estão no mesmo dispositivo
                pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                all_pred_instances_map[img_path] = pred_instances
                

                assign_result = assigner.assign(pred_instances, gt_instances)

                updated_labels = gt_labels.clone()

                # Criar lista de GTs para esta imagem
                all_gt_idx_map[img_path] = []
                count_gts += assign_result.num_gts

                
                # **Vetorização do processo de associação**
                for gt_idx in range(assign_result.num_gts):
                    associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    sanity_count += 1
                    
                    if associated_preds.numel() == 0:
                        continue 

                    # max_score = pred_instances.scores[associated_preds].max().item()
                    given_max_score = pred_instances.scores[associated_preds].max().item()
                    # max_logit = pred_instances.logits[associated_preds].max().item()
                    # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                    logits_associated = pred_instances.logits[associated_preds] 
                    myscores = torch.softmax(logits_associated ,dim=-1)

                    

                    # quero pegar o logit_gt da amostra que tem maior logit_pred
                    # import pdb; pdb.set_trace()
                    myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                    mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                    # max_score = myscores_gt.max().item()
                    max_score_val = myscores.max()
                    max_score_idx = myscores.argmax()
                    amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                    score_gt_max_pred = myscores_gt[amostra_id].cpu().item()
                    myious = assign_result.max_overlaps[associated_preds]
                    max_iou_idx = torch.argmax(myious)
                    amostra_id_iou, classe_id_iou = divmod(max_iou_idx.item(), myscores.size(1))
                    score_gt_max_iou = myscores_gt[amostra_id_iou].cpu().item()
                    score_pred_max_iou = myscores[amostra_id_iou, classe_id].cpu().item()

                    # Compute APS
                    row_scores = myscores[amostra_id]  # pega apenas a linha
                    p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                    S_APS_score = -row_scores[row_scores > p_y].sum().item()
                    # p_y = myscores[amostra_id, classe_id].item()
                    # S_APS_score = -myscores[myscores >= p_y].sum().item()


                    max_score_idx_local = torch.argmax(myscores_gt)
                    #----> remover depois do debug
                    #max_logit_idx_gt = associated_preds[max_score_idx_local]
                    max_logit_idx_gt = associated_preds[amostra_id]
                    logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()
                    


    
                    gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]

                    

                    # Pegar idx local do maior IoU
                    max_iou_values = assign_result.max_overlaps[associated_preds]
                    max_iou_idx_local = torch.argmax(max_iou_values)
                    max_iou_idx = associated_preds[max_iou_idx_local]

                    # Vetor contendo ambos (pode repetir se for o mesmo)
                    important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                    important_associated_ids = torch.unique(important_associated_ids)

                    
                    
                    # Armazenar `gt_idx` na ordem correta
                    all_gt_idx_map[img_path].append(gt_idx)

                

                    if self.selcand == 'max':
                        allbb_preds_map[img_path][gt_idx] = {'pred': score_gt_max_pred, 'logit':logit_gt_max_pred , 
                                                            'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                            'aps':S_APS_score,
                                                            'max_pred':max_score_val.cpu().item(), 'pred_label': classe_id,
                                                            'filtered':False}
                    elif self.selcand == 'iou':
                        allbb_preds_map[img_path][gt_idx] = {'pred': score_gt_max_iou, 'logit':logit_gt_max_pred , 
                                                            'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                            'aps':S_APS_score,
                                                            'max_pred':score_pred_max_iou, 'pred_label': classe_id_iou,
                                                            'filtered':False}
                    
                    else:
                        print("Warning: selcand deve ser 'max' ou 'iou'. Usando 'max' por padrão.")
                    global_index_counter += 1


                    

                    confident_preds =  associated_preds[myscores.max(dim=1).values> self.relabel_thr_high]
                    
    

                    if confident_preds.numel() > 0:
                        
                        #---> original
                        # pred_labels_confident = pred_instances.labels[confident_preds]
                        #---> temporario-filipe-debug - remover depois do debug
                        #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                        pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                        most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                    # elif self.group and len(associated_preds) > 1 and (max_score_val > 0.45):
                    #     labels_group = myscores.argmax(dim=1)
                    #     most_common_label, qtd = Counter(labels_group.tolist()).most_common(1)[0]

                    #     scores_most_common = myscores[:,most_common_label]
                    #     confident_most_common =  associated_preds[scores_most_common> 0.45]
                    #     # import pdb; pdb.set_trace()

                    #     # verifica se a quantidade é maior que 50% do total
                    #     if qtd > (ledn(associated_preds) / 2) and len(confident_most_common) > 2:
                            
                    #         most_common_label = most_common_label
                    #     # senao verifica se tem algum elemento com score maior que o threshold
                    #     elif confident_preds.numel() > 0:
                    #         pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)
                    #         most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                    
                    else:
                        continue  


                    updated_labels[gt_idx] = most_common_label
                    #atualiza também no mapeamento
                    allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                    #atualiza max logit-gt
                    gt_logits = logits_associated[:,most_common_label]
                    # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                    max_logit = gt_logits.max().item()

                    ###-->debug -remover depois
                    # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                    # max_score = myscores_gt.max().item()
                    # max_score_val = myscores.max()
                    # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                    myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                    max_score_idx = myscores.argmax()
                    amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                    score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                    # Compute APS
                    # p_y = myscores[amostra_id, classe_id].item()
                    # S_APS_score = -myscores[myscores >= p_y].sum().item()
                    # row_scores = myscores[amostra_id]  # pega apenas a linha
                    # p_y = row_scores[classe_id].item()
                    # S_APS_score = -row_scores[row_scores > p_y].sum().item()
                    row_scores = myscores[amostra_id]  # pega apenas a linha
                    p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                    S_APS_score = -row_scores[row_scores > p_y].sum().item()
                    # if S_APS_score < -1:
                    #     print(S_APS_score)
                    #     print("a")
                    #     import pdb; pdb.set_trace()
                    #     print("ok")
                    allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score

                    #
                    #-->[original]
                    #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                    #-->[temporario-filipe-debug] - remover depois do debug
                    allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                    allbb_preds_map[img_path][gt_idx]['filtered'] = True



                    
                    

                sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                sub_dataset = datasets[sub_dataset_idx]

                # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                # Atualizar os labels corretamente usando o mapeamento
                for gt_idx, valid_idx in enumerate(valid_instance_indices):
                    sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                    # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
        

        # só faz filter depois do warmup
        #if (runner.epoch + 1) >= self.filter_warmup:
        if (runner.epoch + 1) >= 0:

            # filtering
            #Unir todas as probabilidades das imagens e treinar o GMM**
            # all_scores = np.array([
            #     score['pred'] for img_scores in allbb_preds_map.values()
            #     for score in img_scores.values()
            # ]).reshape(-1, 1)
            # import pdb; pdb.set_trace()
            print("[DEBUG1]: Entrou no filtro")
            all_classes_low_confidence_scores_global_idx = []

            for c in range(num_classes):
                # c_class_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values() if score['gt_label'] == c
                # ]).reshape(-1, 1)

                # if runner.epoch + 1 >1:
                #     import pdb; pdb.set_trace()

                scores = np.array([])
                # img_indexes = np.array([])
                scores_dict = {'pred': np.array([]),
                                'logit': np.array([]),
                                'aps': np.array([])}
                low_conf_dict = {'pred': np.array([]),
                                'logit': np.array([]),
                                'aps': np.array([])}
                img_paths = np.array([], dtype=object)
                c_global_indexes = np.array([])

                for img_path, img_info in allbb_preds_map.items():
                    #for pred, gt_label, index  in img_info.values():
                    for temp_gt_idx, values in img_info.items():
                        #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #if gt_label== c:
                        if values['gt_label'] == c:
                            scores_dict['pred'] = np.append(scores_dict['pred'], values['pred'])
                            scores_dict['logit'] = np.append(scores_dict['logit'], values['logit'])
                            scores_dict['aps'] = np.append(scores_dict['aps'], values['aps'])

                            # calculando como pred apenas para ver depois se len(scores) == 0
                            scores = np.append(scores, values['pred'])

                            
                            

                            c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                            if img_path not in img_paths:
                                img_paths = np.append(img_paths, img_path)
                
                if len(scores) == 0:
                    print(f"Aviso: Nenhuma amostra encontrada para a classe {c}.")
                    continue

                
                

                # low_confidence_scores = calculate_gmm(c_class_scores, n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                            
                low_conf_dict['pred'] = calculate_gmm(scores_dict['pred'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                low_conf_dict['logit'] = calculate_gmm(scores_dict['logit'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                low_conf_dict['aps'] = calculate_gmm(scores_dict['aps'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)

                if self.filter_type == 'pred':
                    scores = scores_dict['pred']
                    low_confidence_scores = low_conf_dict['pred']
                elif self.filter_type == 'logit':
                    scores = scores_dict['logit']
                    low_confidence_scores = low_conf_dict['logit']
                elif self.filter_type == 'aps':
                    scores = scores_dict['aps']
                    low_confidence_scores = low_conf_dict['aps']


                # print("[DEBUG1.5]: INICIANDO GMM")

                c_class_scores = c_class_scores = scores.reshape(-1, 1)



                # threshold = self.filter_conf
                
                threshold = self.filter_confgmm
            
                low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                

                # draw pred hist
                save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_pred_ep{runner.epoch + 1}.png"
                low_confidence_indices_temp = np.where(low_conf_dict['pred'] > threshold)[0]
                draw_score_histogram(scores_dict['pred'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_confgmm)

                # draw logit hist
                save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_logit_ep{runner.epoch + 1}.png"
                low_confidence_indices_temp = np.where(low_conf_dict['logit'] > threshold)[0]
                draw_score_histogram(scores_dict['logit'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_confgmm)

                # draw aps hist
                save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_aps_ep{runner.epoch + 1}.png"
                low_confidence_indices_temp = np.where(low_conf_dict['aps'] > threshold)[0]
                draw_score_histogram(scores_dict['aps'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_confgmm)

                # Dentro do seu for c in range(num_classes):
                # import pdb; pdb.set_trace()
                # ... após o fit do GMM
                # plt.figure()
                # plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                # # Adiciona linha vertical que separa as duas classes com base no GMM
                # plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                # # plt.legend()
                # plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                # plt.xlabel('Score')
                # plt.ylabel('Frequência')
                # plt.grid(True)
                # plt.tight_layout()
                # # Criar pasta, se necessário
                # save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                # plt.savefig(save_path)
                # plt.close()
                

            
                

            # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
            # index = 0
            # import pdb; pdb.set_trace()
            my_counter = 0
            for img_path, gt_scores in allbb_preds_map.items():
                sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                sub_dataset = datasets[sub_dataset_idx]


                valid_instance_indices = [
                    idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                    if inst['ignore_flag'] == 0
                ]

                # **Mapear GTs reais na ordem correta**
                gt_idx_list = all_gt_idx_map[img_path]  

                
                

                for gt_idx in gt_idx_list:
                # for gt_idx in valid_instance_indices:

                    related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']  

                    

                    #if index in low_confidence_indices:
                    #if related_global_index in all_classes_low_confidence_scores_global_idx:
                    # if low confidence and not too high confident
                    #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.relabel_conf):
                    #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.5):
                    
                    # if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.3):
                    if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                        # if my_counter<5:

                                                            
                        #     import shutil

                        #     # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                            
                        #     base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                        #     possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg", "_grouped.jpg"]

                        #     for suffix in possible_suffixes:
                                
                        #         base_debug_path = base_prefix + suffix
                        #         if os.path.exists(base_debug_path):
                        #             my_counter+=1 
                        #             filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                        #             # if suffix == "_relabeled.jpg":
                        #             #     import pdb; pdb.set_trace()
                        #             shutil.copy(base_debug_path, filtered_debug_path)
                        #             print(f"[INFO] Cópia criada: {filtered_debug_path}")
                        #             break  # Para no primeiro que encontrar 

                        #     # desenhar_bboxesv3_filtered(all_inputs_map[img_path], sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_filtered.jpg')  
                        # # Encontrar `valid_idx` correspondente ao `gt_idx`
                        # if gt_idx in gt_idx_list:
                        #[ME PARECE ERRADO]
                        # valid_idx = valid_instance_indices[gt_idx_list.index(gt_idx)]
                        #[TESTAR ESSE]
                        # import pdb; pdb.set_trace()
                        valid_idx = valid_instance_indices[gt_idx]

                        # self.double_thr
                        # if allbb_preds_map[img_path][gt_idx]['max_pred'] >= self.double_thr:
                        #     #update
                        #     sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = allbb_preds_map[img_path][gt_idx]['pred_label']
                        # else:    
                            #filtra
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1

                        
                        
                    
                
                            
                            
                        
                    
                    

        print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")

        
        
# === Lightweight GNN noise filter hook ===
@HOOKS.register_module()
class MyHookGraphNoiseRelabelFilterGMMSanity_inverted(Hook):
    @torch.no_grad()
    def _compute_co_probs(self, device):
        co_counts_cpu = self._cooc_counts.clone().float()
        row_sum = co_counts_cpu.sum(dim=1, keepdim=True).clamp_min(self.cooc_eps)
        co_probs = (co_counts_cpu / row_sum).to(device)
        co_probs.fill_diagonal_(1.0)
        return co_probs
    """
    Antes de cada época, constrói um grafo por imagem usando as detecções atuais,
    executa uma GNN leve para estimar p_ruido por GT e:
      (i) marca instâncias com p_ruido alto como ignore_flag=1, ou
      (ii) opcionalmente relabela para a maioria semântica dos vizinhos confiáveis.

    Parâmetros principais:
      - warmup_epochs: até esse ponto só treina estatísticas, sem filtrar.
      - thr_noise: limiar fixo de p_ruido (0..1) OU usa percentil por classe se use_percentile=True.
      - use_percentile: se True, usa percentil por classe (ex.: 80) como corte dinâmico.
      - k: número de vizinhos no k-NN.
      - do_relabel: se True, tenta relabelar ao invés de ignorar quando houver maioria forte.
      - relabel_min_agree: fração mínima (0..1) de vizinhos confiáveis que concordam.
      - num_classes: nº de classes do seu dataset.
    """
    def __init__(self,
                 warmup_epochs: int = 1,
                 thr_noise: float = 0.7,
                 use_percentile: bool = False,
                 percentile: float = 80.0,
                 k: int = 8,
                 do_relabel: bool = False,
                 relabel_min_agree: float = 0.6,
                 num_classes: int = 20,
                 reload_dataset: bool = True,
                 iou_assigner=0.5,
                 low_quality=False,
                 train_ratio: float = 1.0,
                 k_min: int = 1,
                 k_max: int = 8,
                 k_mode: str = 'sqrt',
                 gnn_lr: float = 1e-3, 
                 gnn_train_steps: int = 50, 
                 tau_gt: float = 0.3, 
                 neighbor_agree: float = 0.6,
                 trust_thr: float = 0.9,
                 corr_thr_low = 0.3,
                 # --- contexto com "gate" ---
                 ctx_conf_beta: float = 2.0,    # peso dos vizinhos por confiança (pmax^beta)
                 ctx_gate_gamma: float = 8.0,   # quão abrupto é o gate sigmoide
                 ctx_dist_sigma: float = 0.75,
                 cong_hidden: int = 128,
                 cong_lr: float = 1e-3,
                 cong_train_steps: int = 100,
                 cong_alpha: float = 0.5,
                 # --- W&B logging params ---
                 use_wandb: bool = True,
                 wandb_project: str = 'noisy-od',
                 wandb_run_name: str = '',
                 wandb_max_images: int = 8,
                 # --- prctx logging knobs ---
                 wandb_log_prctx: bool = True,
                 prctx_topk: int = 5,
                 prctx_max_nodes: int = 3,
                 relabel_thr_ctx: float = 0.7,
                 relabel_thr_high: float = 0.9,
                 # pesos e piso do componente visual no contexto
                 ctx_feat_weight: float = 0.6,
                 ctx_prob_weight: float = 0.4,
                 ctx_sim_min: float = 0.2,
                 filter_thr: float = 0.7,
                 filter_confgmm: float = 0.95
                 ):
        self.warmup_epochs = warmup_epochs
        self.thr_noise = float(thr_noise)
        self.use_percentile = bool(use_percentile)
        self.percentile = float(percentile)
        self.k = int(k)
        self.do_relabel = bool(do_relabel)
        self.relabel_min_agree = float(relabel_min_agree)
        self.num_classes = int(num_classes)
        self.reload_dataset = reload_dataset
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        # será inicializada no primeiro uso para o device correto
        self._gnn = None
        self.train_ratio = float(train_ratio)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.k_mode = str(k_mode)
        self.gnn_lr = float(gnn_lr)
        self.gnn_train_steps = int(gnn_train_steps)
        self.tau_gt = float(tau_gt)
        self.neighbor_agree = float(neighbor_agree)
        self._opt = None  # optimizer da GNN (lazily created)
        self._trained = False  # flag para indicar se a GNN já foi treinada nesta execução
        self.trust_thr = float(trust_thr)
        self.corr_thr_low = corr_thr_low  # limiar para p_ruido só pela co-ocorrência (baixo relacionamento com classes presentes)   
        # knobs de gate de contexto
        self.ctx_conf_beta = float(ctx_conf_beta)
        self.ctx_gate_gamma = float(ctx_gate_gamma)
        self.ctx_dist_sigma = float(ctx_dist_sigma)
        # co-ocorrência (CPU): inicia suavizada para evitar zeros
        self._cooc_counts = torch.ones(self.num_classes, self.num_classes, dtype=torch.float32)
        self.cooc_eps = 1e-3          # suavização mínima ao normalizar (em vez de 1e-6 mencionado)
        self.reset_cooc_each_epoch = True  # se True, zera (para 1s) a cada época
        # ConG (context graph head)
        self.cong_hidden = int(cong_hidden)
        self.cong_lr = float(cong_lr)
        self.cong_train_steps = int(cong_train_steps)
        self.cong_alpha = float(cong_alpha)  # força do reweight: lw = 1 - alpha * p_noise
        self._cong = None
        self._opt_cong = None
        # W&B logging
        self.use_wandb = bool(use_wandb)
        self.wandb_project = str(wandb_project)
        self.wandb_run_name = str(wandb_run_name)
        self.wandb_max_images = int(wandb_max_images)
        self._wandb_ready = False
        self._wandb_img_budget = 0
        self._wandb_imgs = []  # lista de wandb.Image para log único por época
        # --- Phase-1 relabel memory (per epoch, per image) ---
        self._phase1_relabels = {}

        self.wandb_log_prctx = bool(wandb_log_prctx)
        self.prctx_topk = int(prctx_topk)
        self.prctx_max_nodes = int(prctx_max_nodes)

        # W&B extra knobs
        self.wandb_log_if_any_kld = True  # loga imagem se houver qualquer KLD acima do corte, mesmo vetado

        self.relabel_thr_ctx = float(relabel_thr_ctx)
        self.relabel_thr_high = float(relabel_thr_high)
        self.ctx_feat_weight = float(ctx_feat_weight)
        self.ctx_prob_weight = float(ctx_prob_weight)
        self.ctx_sim_min     = float(ctx_sim_min)
        self.filter_thr = float(filter_thr)
        self.filter_confgmm = float(filter_confgmm)
        self.selcand = "max" # max | iou
        self.numGMM = 4
        self.filter_type = "pred" # pred| logit | aps
        

    def _ensure_gnn(self, device):
        if self._gnn is None:
            #self._gnn = GraphNoiseNet(num_classes=self.num_classes, cls_emb_dim=32, prob_dim=64, hidden=128, edge_dim=3).to(device)
            self._gnn = GraphNoiseNet(num_classes=self.num_classes, cls_emb_dim=32, prob_dim=64, hidden=128, edge_dim=2).to(device)
        else:
            self._gnn.to(device)
        self._gnn.eval()  # usamos apenas para scoring no hook
        if self._opt is None:
            self._opt = optim.Adam(self._gnn.parameters(), lr=self.gnn_lr)


    # --- Helper for effective labels from logits and semantic graph ---
    @torch.no_grad()
    def _effective_labels_from_logits(self, node_logits_t: torch.Tensor, gt_labels_t: torch.Tensor, trust_thr: float):
        probs = node_logits_t.softmax(dim=-1)
        pmax, argmax = probs.max(dim=-1)
        eff = gt_labels_t.clone()
        eff[pmax >= trust_thr] = argmax[pmax >= trust_thr]
        return eff, probs, pmax, argmax

    @torch.no_grad()
    def _build_semantic_graph(self, probs: torch.Tensor, eff_labels: torch.Tensor, cooc_probs: torch.Tensor,
                              bboxes_xyxy: torch.Tensor, img_w: int, img_h: int, k: int = 4,
                              features: torch.Tensor = None, feat_weight: float = 0.5, prob_weight: float = 0.5):
        """
        Build a semantic kNN graph using cosine similarity on class-prob vectors and visual features, plus spatial proximity.
        Returns edge_index [2,E] and edge_attr [E,7] = [sim_prob, f_sim, dx, dy, rw, rh, co].
        - sim_prob : similarity between probability distributions (cosine, [0,1])
        - f_sim    : similarity between features (cosine, [0,1])
        - dx,dy    : normalized offsets (src→dst) scaled by dst size
        - rw,rh    : log size ratios (src/dst)
        - co       : co-occurrence prior p(li→lj)
        """
        device = probs.device
        N = probs.size(0)
        if N <= 1:
            return torch.empty(2,0, dtype=torch.long, device=device), torch.empty(0,7, device=device)

        # --- Similaridade entre distribuições de probabilidade (softmax dos logits) ---
        Pn = F.normalize(probs, p=2, dim=-1)
        S_prob = torch.mm(Pn, Pn.t())
        S_prob = (S_prob + 1.0) * 0.5
        S_prob.fill_diagonal_(0.0)

        # --- Similaridade entre features visuais (softmax dos logits ou embeddings) ---
        if features is None:
            # Usa softmax dos logits como features padrão
            features = probs
        Fn = F.normalize(features, p=2, dim=-1)
        S_feat = torch.mm(Fn, Fn.t())
        S_feat = (S_feat + 1.0) * 0.5
        S_feat.fill_diagonal_(0.0)

        # combina do jeito que o chamador pediu (sem renormalizar aqui)
        S = prob_weight * S_prob + feat_weight * S_feat
        S.fill_diagonal_(0.0)

        # top-k by combined similarity
        kk = min(max(1, k), N-1)
        topk = S.topk(kk, dim=1).indices
        src = torch.arange(N, device=device).unsqueeze(1).expand(-1, kk).reshape(-1)
        dst = topk.reshape(-1)

        # spatial attributes from bboxes
        b = bboxes_xyxy
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w  = (x2 - x1).clamp(min=1e-3)
        h  = (y2 - y1).clamp(min=1e-3)

        dx = (cx[src] - cx[dst]) / (w[dst] + 1e-3)
        dy = (cy[src] - cy[dst]) / (h[dst] + 1e-3)
        rw = torch.log(w[src] / w[dst])
        rh = torch.log(h[src] / h[dst])

        sim_prob = S_prob[src, dst].unsqueeze(1)
        f_sim = S_feat[src, dst].unsqueeze(1)
        co  = cooc_probs[eff_labels[src], eff_labels[dst]].unsqueeze(1)

        edge_attr = torch.cat([
            sim_prob,
            f_sim,
            dx.unsqueeze(1), dy.unsqueeze(1),
            rw.unsqueeze(1), rh.unsqueeze(1),
            co
        ], dim=1)
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index, edge_attr

    @torch.no_grad()
    def _make_pseudo_targets(self, node_labels_t, edge_index, node_logits_t, gt_idx_t, cut_prob: float):
        """Constroi y_noise (0/1) por nó.
        Critérios:
            A) p_gt < tau_gt
            B) Fração de vizinhos confiáveis (p_gt >= tau_gt) que concordam com GT é < neighbor_agree
        Marca 1 se (A) OU (B).
        """
        if node_logits_t.numel() == 0:
            return torch.zeros(0, device=node_logits_t.device)
        probs = node_logits_t.softmax(dim=-1)
        idx = torch.arange(probs.size(0), device=probs.device)
        p_gt = probs[idx, gt_idx_t]
        critA = (p_gt < self.tau_gt)

        src, dst = edge_index
        agree_cnt = torch.zeros_like(p_gt)
        total_cnt = torch.zeros_like(p_gt)
        neigh_pgt = p_gt[src]
        neigh_lbl = gt_idx_t[src]
        is_conf = neigh_pgt >= self.tau_gt
        same = (neigh_lbl == gt_idx_t[dst]) & is_conf
        agree_cnt.index_add_(0, dst, same.float())
        total_cnt.index_add_(0, dst, is_conf.float())
        frac_agree = torch.zeros_like(p_gt)
        mask = total_cnt > 0
        frac_agree[mask] = agree_cnt[mask] / total_cnt[mask]
        critB = torch.zeros_like(critA)
        critB[mask] = (frac_agree[mask] < self.neighbor_agree)
        y = (critA | critB).float()
        return y
    
            

    @torch.no_grad()
    def _pnoise_from_cooc(self, eff_labels_t: torch.Tensor, co_probs_dev: torch.Tensor, thr: float):
        """Retorna p_noise em [0,1] por nó baseado APENAS na matriz de co-ocorrência.
        Para cada nó i (classe li), considera as classes dos demais nós j na imagem e
        pega best = max_j co_probs[li, lj]. Se best >= thr → p_noise=0; caso contrário
        p_noise = (thr - best)/thr (quanto menor a co-ocorrência, maior o p_noise).
        """
        N = eff_labels_t.numel()
        if N <= 1:
            return torch.zeros(N, device=eff_labels_t.device)
        li = eff_labels_t                            # [N]
        lj = eff_labels_t                            # [N]
        # matriz cooc para todos pares (i,j)
        C = co_probs_dev[li][:, lj]                  # [N,N]
        # ignora diagonal (relacionar consigo mesmo não conta)
        C = C.masked_fill(torch.eye(N, dtype=torch.bool, device=C.device), 0.0)
        best, _ = C.max(dim=1)                       # [N]
        p = (thr - best).clamp(min=0.0) / max(thr, 1e-6)
        return p
    
    def _ensure_cong(self, device):
        if self._cong is None:
            self._cong = ConG(num_classes=self.num_classes, hidden=self.cong_hidden).to(device)
        else:
            self._cong.to(device)
        if self._opt_cong is None:
            self._opt_cong = optim.AdamW(self._cong.parameters(), lr=self.cong_lr, weight_decay=0.0)

    def before_train_epoch(self, runner):
        if (runner.epoch + 1) <= 0:
            return

        # Reinicia lista de imagens do W&B a cada época
        self._wandb_imgs = []
        dataloader = runner.train_loop.dataloader
        dataset = dataloader.dataset
        # opcional: reiniciar co-ocorrência por época
        if self.reset_cooc_each_epoch:
            self._cooc_counts = torch.ones(self.num_classes, self.num_classes, dtype=torch.float32)

        # --- W&B: init per epoch (lazy) and reset epoch buffers ---
        self._wandb_img_budget = self.wandb_max_images
        self._wandb_imgs = []
        # --- Clear Phase-1 relabel memory for the new epoch ---
        self._phase1_relabels = {}
        if self.use_wandb and (wandb is not None):
            if not self._wandb_ready:
                try:
                    wandb.init(project=self.wandb_project or 'noisy-od',
                               name=self.wandb_run_name or f'run-{os.path.basename(runner.work_dir)}',
                               dir=runner.work_dir,
                               reinit=True)
                    self._wandb_ready = True
                except Exception as _:
                    self._wandb_ready = False

        reload_dataset = self.reload_dataset

        if reload_dataset:
            runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
            runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

            runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
            runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        if not hasattr(dataset, 'datasets'):
            raise ValueError("Esperado um ConcatDataset.")
        datasets = dataset.datasets

        # mapa rápido: img_path -> (sub_dataset_idx, data_idx)
        dataset_img_map = {di['img_path']: (sidx, didx)
                            for sidx, subds in enumerate(datasets)
                            if hasattr(subds, 'data_list')
                            for didx, di in enumerate(subds.data_list)}
        
        # === PHASE 3: GMM-based noise filtering (after relabeling) ===
        # Faz um processo em duas passagens:
        #   Passo 3A) varre toda a época e acumula scores por classe (p_gt)
        #   Passo 3B) ajusta um GMM por classe usando TODOS os scores acumulados e marca low-confidence como noisy
        dataloader = runner.train_loop.dataloader
        dataset = dataloader.dataset

        #reload_dataset = True
        # my_value = 3
        #reload_dataset = getattr(runner.cfg, 'reload_dataset', False)  
        #my_value =  getattr(runner.cfg, 'my_value', 10)  
        # reload_dataset = self.reload_dataset
        # relabel_conf = self.relabel_conf
        
        # # import pdb; pdb.set_trace()
        # if reload_dataset:
        #     runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
        #     runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

        #     runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
        #     runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

        # Desencapsular RepeatDataset até o ConcatDataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        if not hasattr(dataset, 'datasets'):
            raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

        datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

        assigner = MaxIoUAssigner(
            pos_iou_thr=self.iou_assigner,
            neg_iou_thr=self.iou_assigner,
            min_pos_iou=self.iou_assigner,
            match_low_quality=self.low_quality
        )

        # Criar um dicionário para acesso rápido ao índice das imagens no dataset
        dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                        for sub_dataset_idx, sub_dataset in enumerate(datasets)
                        if hasattr(sub_dataset, 'data_list')
                        for data_idx, data_info in enumerate(sub_dataset.data_list)}
        
        # Dicionário para armazenar os scores médios por GT sem assumir sequência
        allbb_preds_map = defaultdict(dict)
        all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
        all_pred_instances_map = defaultdict(dict)
        # all_inputs_map = defaultdict(dict)

        num_classes = 20
        all_classes_preds_map=[]
        all_classes_gt_idx_map=[]
        for i in range(num_classes):
            all_classes_preds_map.append(defaultdict(list))
            all_classes_gt_idx_map.append(defaultdict(list))
        

        count_gts = 0
        sanity_count = 0
        global_index_counter = 0

        
        # Processar o batch de dados
        for batch_idx, data_batch in enumerate(dataloader):
            with torch.no_grad():
                #data = runner.model.data_preprocessor(data_batch, False)
                data = runner.model.data_preprocessor(data_batch, True)
                
                inputs = data['inputs']
                data_samples = data['data_samples']
                
                #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
            
                #temp_filipe

            for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
            # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                img_path = data_sample.img_path
                
                if img_path not in dataset_img_map:
                    print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                    continue

                # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                pred_instances = predictions_pred[i].pred_instances

                # Pegar bounding boxes preditas
                pred_instances.priors = pred_instances.pop('bboxes')

                # Obter os ground truths
                gt_instances = data_sample.gt_instances
                gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                # Garantir que os GTs estão no mesmo dispositivo das predições
                gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                # Garantir que as predições também estão no mesmo dispositivo
                pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                all_pred_instances_map[img_path] = pred_instances
                

                assign_result = assigner.assign(pred_instances, gt_instances)

                updated_labels = gt_labels.clone()

                # Criar lista de GTs para esta imagem
                all_gt_idx_map[img_path] = []
                count_gts += assign_result.num_gts

                
                # **Vetorização do processo de associação**
                for gt_idx in range(assign_result.num_gts):
                    associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    sanity_count += 1
                    
                    if associated_preds.numel() == 0:
                        continue 

                    # max_score = pred_instances.scores[associated_preds].max().item()
                    given_max_score = pred_instances.scores[associated_preds].max().item()
                    # max_logit = pred_instances.logits[associated_preds].max().item()
                    # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                    logits_associated = pred_instances.logits[associated_preds] 
                    myscores = torch.softmax(logits_associated ,dim=-1)

                    

                    # quero pegar o logit_gt da amostra que tem maior logit_pred
                    # import pdb; pdb.set_trace()
                    myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                    mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                    # max_score = myscores_gt.max().item()
                    max_score_val = myscores.max()
                    max_score_idx = myscores.argmax()
                    amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                    score_gt_max_pred = myscores_gt[amostra_id].cpu().item()
                    myious = assign_result.max_overlaps[associated_preds]
                    max_iou_idx = torch.argmax(myious)
                    amostra_id_iou, classe_id_iou = divmod(max_iou_idx.item(), myscores.size(1))
                    score_gt_max_iou = myscores_gt[amostra_id_iou].cpu().item()
                    score_pred_max_iou = myscores[amostra_id_iou, classe_id].cpu().item()

                    # Compute APS
                    row_scores = myscores[amostra_id]  # pega apenas a linha
                    p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                    S_APS_score = -row_scores[row_scores > p_y].sum().item()
                    # p_y = myscores[amostra_id, classe_id].item()
                    # S_APS_score = -myscores[myscores >= p_y].sum().item()


                    max_score_idx_local = torch.argmax(myscores_gt)
                    #----> remover depois do debug
                    #max_logit_idx_gt = associated_preds[max_score_idx_local]
                    max_logit_idx_gt = associated_preds[amostra_id]
                    logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()
                    


    
                    gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]

                    

                    # Pegar idx local do maior IoU
                    max_iou_values = assign_result.max_overlaps[associated_preds]
                    max_iou_idx_local = torch.argmax(max_iou_values)
                    max_iou_idx = associated_preds[max_iou_idx_local]

                    # Vetor contendo ambos (pode repetir se for o mesmo)
                    important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                    important_associated_ids = torch.unique(important_associated_ids)

                    
                    
                    # Armazenar `gt_idx` na ordem correta
                    all_gt_idx_map[img_path].append(gt_idx)

                

                    if self.selcand == 'max':
                        allbb_preds_map[img_path][gt_idx] = {'pred': score_gt_max_pred, 'logit':logit_gt_max_pred , 
                                                            'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                            'aps':S_APS_score,
                                                            'max_pred':max_score_val.cpu().item(), 'pred_label': classe_id,
                                                            'filtered':False}
                    elif self.selcand == 'iou':
                        allbb_preds_map[img_path][gt_idx] = {'pred': score_gt_max_iou, 'logit':logit_gt_max_pred , 
                                                            'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                            'aps':S_APS_score,
                                                            'max_pred':score_pred_max_iou, 'pred_label': classe_id_iou,
                                                            'filtered':False}
                    
                    else:
                        print("Warning: selcand deve ser 'max' ou 'iou'. Usando 'max' por padrão.")
                    global_index_counter += 1


                    

                    confident_preds =  associated_preds[myscores.max(dim=1).values> self.relabel_thr_high]
                    
    

                    if confident_preds.numel() > 0:
                        
                        #---> original
                        # pred_labels_confident = pred_instances.labels[confident_preds]
                        #---> temporario-filipe-debug - remover depois do debug
                        #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                        pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                        most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                    # elif self.group and len(associated_preds) > 1 and (max_score_val > 0.45):
                    #     labels_group = myscores.argmax(dim=1)
                    #     most_common_label, qtd = Counter(labels_group.tolist()).most_common(1)[0]

                    #     scores_most_common = myscores[:,most_common_label]
                    #     confident_most_common =  associated_preds[scores_most_common> 0.45]
                    #     # import pdb; pdb.set_trace()

                    #     # verifica se a quantidade é maior que 50% do total
                    #     if qtd > (ledn(associated_preds) / 2) and len(confident_most_common) > 2:
                            
                    #         most_common_label = most_common_label
                    #     # senao verifica se tem algum elemento com score maior que o threshold
                    #     elif confident_preds.numel() > 0:
                    #         pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)
                    #         most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                    
                    else:
                        continue  


                    updated_labels[gt_idx] = most_common_label
                    #atualiza também no mapeamento
                    allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                    #atualiza max logit-gt
                    gt_logits = logits_associated[:,most_common_label]
                    # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                    max_logit = gt_logits.max().item()

                    ###-->debug -remover depois
                    # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                    # max_score = myscores_gt.max().item()
                    # max_score_val = myscores.max()
                    # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                    myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                    max_score_idx = myscores.argmax()
                    amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                    score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                    # Compute APS
                    # p_y = myscores[amostra_id, classe_id].item()
                    # S_APS_score = -myscores[myscores >= p_y].sum().item()
                    # row_scores = myscores[amostra_id]  # pega apenas a linha
                    # p_y = row_scores[classe_id].item()
                    # S_APS_score = -row_scores[row_scores > p_y].sum().item()
                    row_scores = myscores[amostra_id]  # pega apenas a linha
                    p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                    S_APS_score = -row_scores[row_scores > p_y].sum().item()
                    # if S_APS_score < -1:
                    #     print(S_APS_score)
                    #     print("a")
                    #     import pdb; pdb.set_trace()
                    #     print("ok")
                    allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score

                    #
                    #-->[original]
                    #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                    #-->[temporario-filipe-debug] - remover depois do debug
                    allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                    allbb_preds_map[img_path][gt_idx]['filtered'] = True



                    
                    

                sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                sub_dataset = datasets[sub_dataset_idx]

                # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                # Atualizar os labels corretamente usando o mapeamento
                for gt_idx, valid_idx in enumerate(valid_instance_indices):
                    sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                    # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
        

        # só faz filter depois do warmup
        #if (runner.epoch + 1) >= self.filter_warmup:
        if (runner.epoch + 1) >= 1:

            # filtering
            #Unir todas as probabilidades das imagens e treinar o GMM**
            # all_scores = np.array([
            #     score['pred'] for img_scores in allbb_preds_map.values()
            #     for score in img_scores.values()
            # ]).reshape(-1, 1)
            # import pdb; pdb.set_trace()
            print("[DEBUG1]: Entrou no filtro")
            all_classes_low_confidence_scores_global_idx = []

            for c in range(num_classes):
                # c_class_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values() if score['gt_label'] == c
                # ]).reshape(-1, 1)

                # if runner.epoch + 1 >1:
                #     import pdb; pdb.set_trace()

                scores = np.array([])
                # img_indexes = np.array([])
                scores_dict = {'pred': np.array([]),
                                'logit': np.array([]),
                                'aps': np.array([])}
                low_conf_dict = {'pred': np.array([]),
                                'logit': np.array([]),
                                'aps': np.array([])}
                img_paths = np.array([], dtype=object)
                c_global_indexes = np.array([])

                for img_path, img_info in allbb_preds_map.items():
                    #for pred, gt_label, index  in img_info.values():
                    for temp_gt_idx, values in img_info.items():
                        #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #if gt_label== c:
                        if values['gt_label'] == c:
                            scores_dict['pred'] = np.append(scores_dict['pred'], values['pred'])
                            scores_dict['logit'] = np.append(scores_dict['logit'], values['logit'])
                            scores_dict['aps'] = np.append(scores_dict['aps'], values['aps'])

                            # calculando como pred apenas para ver depois se len(scores) == 0
                            scores = np.append(scores, values['pred'])

                            
                            

                            c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                            if img_path not in img_paths:
                                img_paths = np.append(img_paths, img_path)
                
                if len(scores) == 0:
                    print(f"Aviso: Nenhuma amostra encontrada para a classe {c}.")
                    continue

                
                

                # low_confidence_scores = calculate_gmm(c_class_scores, n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                            
                low_conf_dict['pred'] = calculate_gmm(scores_dict['pred'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                low_conf_dict['logit'] = calculate_gmm(scores_dict['logit'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                low_conf_dict['aps'] = calculate_gmm(scores_dict['aps'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)

                if self.filter_type == 'pred':
                    scores = scores_dict['pred']
                    low_confidence_scores = low_conf_dict['pred']
                elif self.filter_type == 'logit':
                    scores = scores_dict['logit']
                    low_confidence_scores = low_conf_dict['logit']
                elif self.filter_type == 'aps':
                    scores = scores_dict['aps']
                    low_confidence_scores = low_conf_dict['aps']


                # print("[DEBUG1.5]: INICIANDO GMM")

                c_class_scores = c_class_scores = scores.reshape(-1, 1)



                # threshold = self.filter_conf
                
                threshold = self.filter_confgmm
            
                low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                

                # draw pred hist
                save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_pred_ep{runner.epoch + 1}.png"
                low_confidence_indices_temp = np.where(low_conf_dict['pred'] > threshold)[0]
                draw_score_histogram(scores_dict['pred'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_confgmm)

                # draw logit hist
                save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_logit_ep{runner.epoch + 1}.png"
                low_confidence_indices_temp = np.where(low_conf_dict['logit'] > threshold)[0]
                draw_score_histogram(scores_dict['logit'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_confgmm)

                # draw aps hist
                save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_aps_ep{runner.epoch + 1}.png"
                low_confidence_indices_temp = np.where(low_conf_dict['aps'] > threshold)[0]
                draw_score_histogram(scores_dict['aps'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_confgmm)

                # Dentro do seu for c in range(num_classes):
                # import pdb; pdb.set_trace()
                # ... após o fit do GMM
                # plt.figure()
                # plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                # # Adiciona linha vertical que separa as duas classes com base no GMM
                # plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                # # plt.legend()
                # plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                # plt.xlabel('Score')
                # plt.ylabel('Frequência')
                # plt.grid(True)
                # plt.tight_layout()
                # # Criar pasta, se necessário
                # save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                # plt.savefig(save_path)
                # plt.close()
                

            
                

            # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
            # index = 0
            # import pdb; pdb.set_trace()
            my_counter = 0
            for img_path, gt_scores in allbb_preds_map.items():
                sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                sub_dataset = datasets[sub_dataset_idx]


                valid_instance_indices = [
                    idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                    if inst['ignore_flag'] == 0
                ]

                # **Mapear GTs reais na ordem correta**
                gt_idx_list = all_gt_idx_map[img_path]  

                
                

                for gt_idx in gt_idx_list:
                # for gt_idx in valid_instance_indices:

                    related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']  

                    

                    #if index in low_confidence_indices:
                    #if related_global_index in all_classes_low_confidence_scores_global_idx:
                    # if low confidence and not too high confident
                    #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.relabel_conf):
                    #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.5):
                    
                    # if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.3):
                    if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                        # if my_counter<5:

                                                            
                        #     import shutil

                        #     # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                            
                        #     base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                        #     possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg", "_grouped.jpg"]

                        #     for suffix in possible_suffixes:
                                
                        #         base_debug_path = base_prefix + suffix
                        #         if os.path.exists(base_debug_path):
                        #             my_counter+=1 
                        #             filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                        #             # if suffix == "_relabeled.jpg":
                        #             #     import pdb; pdb.set_trace()
                        #             shutil.copy(base_debug_path, filtered_debug_path)
                        #             print(f"[INFO] Cópia criada: {filtered_debug_path}")
                        #             break  # Para no primeiro que encontrar 

                        #     # desenhar_bboxesv3_filtered(all_inputs_map[img_path], sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_filtered.jpg')  
                        # # Encontrar `valid_idx` correspondente ao `gt_idx`
                        # if gt_idx in gt_idx_list:
                        #[ME PARECE ERRADO]
                        # valid_idx = valid_instance_indices[gt_idx_list.index(gt_idx)]
                        #[TESTAR ESSE]
                        # import pdb; pdb.set_trace()
                        valid_idx = valid_instance_indices[gt_idx]

                        # self.double_thr
                        # if allbb_preds_map[img_path][gt_idx]['max_pred'] >= self.double_thr:
                        #     #update
                        #     sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = allbb_preds_map[img_path][gt_idx]['pred_label']
                        # else:    
                            #filtra
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1

                        
                        
                    
                
                            
                            
                        
                    
                    

        print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")

        # ----------------- GNN TRAIN/FILTER PHASE (unified pass) -----------------
        # === PHASE 1: build co-occurrence (presence-based) and apply only high-conf relabel ===
        for data_batch in dataloader:
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0 else next(runner.model.parameters()).device)
            self._ensure_cong(device)
            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue
                pred_instances = preds[i].pred_instances.to(device)
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')
                gt = ds.gt_instances.to(device)
                assigner = MaxIoUAssigner(pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner,
                                          min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality)
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue
                logits = pred_instances.logits
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                sub_idx, d_idx = dataset_img_map[img_path]
                subds = datasets[sub_idx]
                valid_instance_indices = [idx for idx, inst in enumerate(subds.data_list[d_idx]['instances']) if inst['ignore_flag'] == 0]

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []
                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    
                    choice = assoc[j]
                    node_labels.append(int(gt.labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)
                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)
                vis_feats = node_logits_t  # usar logits (pré-softmax) como descritor visual
                pmax, pred_cls = pr.max(dim=1)

                # Relabel only by high confidence (no context yet)
                for relabel_idx in range(node_labels_t.shape[0]):
                    if float(pmax[relabel_idx]) >= self.relabel_thr_high:
                        new_lab = int(pred_cls[relabel_idx].item())
                        gt_local_idx = node_img_local_to_valid[relabel_idx]
                        if 0 <= gt_local_idx < len(valid_instance_indices):
                            valid_idx = valid_instance_indices[gt_local_idx]
                            old_lab = int(node_labels_t[relabel_idx].item())
                            # Atualiza tensores/listas deste batch
                            node_labels_t[relabel_idx] = new_lab
                            inst = subds.data_list[d_idx]['instances'][valid_idx]
                            updated = False
                            for key in ['labels','label','bbox_label']:
                                if key in inst:
                                    inst[key] = new_lab
                                    updated = True
                            if not updated:
                                inst['labels'] = new_lab
                            # Memoriza para overlay na Fase 3 (por imagem, indexado pelo GT local)
                            if img_path not in self._phase1_relabels:
                                self._phase1_relabels[img_path] = {}
                            # chave: índice local de GT usado para construir os nós (node_img_local_to_valid)
                            self._phase1_relabels[img_path][int(gt_local_idx)] = int(old_lab)
                        else:
                            if hasattr(runner, 'logger'):
                                runner.logger.warning(f"[Phase1] Skip relabel write: gt_local_idx={gt_local_idx} out of range for {os.path.basename(img_path)}")

                # Presence-based co-occurrence update (after relabel-high)
                eff_labels_t = node_labels_t.clone()
                if eff_labels_t.numel() > 1:
                    uniq = torch.unique(eff_labels_t.detach().to(torch.long))
                    if uniq.numel() > 1:
                        ui = uniq.unsqueeze(1).expand(-1, uniq.numel())
                        uj = uniq.unsqueeze(0).expand(uniq.numel(), -1)
                        mask = (ui != uj)
                        pairs_i = ui[mask].reshape(-1).to('cpu', dtype=torch.long)
                        pairs_j = uj[mask].reshape(-1).to('cpu', dtype=torch.long)
                        self._cooc_counts.index_put_((pairs_i, pairs_j), torch.ones_like(pairs_i, dtype=torch.float32), accumulate=True)

        # Freeze epoch prior
        # epoch_device = runner.model.device
        epoch_device = next(runner.model.parameters()).device
        co_probs_epoch = self._compute_co_probs(epoch_device)

        # === PHASE 2: Train ConG using the fixed co_probs_epoch ===
        steps = 0
        for data_batch in dataloader:
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0 else next(runner.model.parameters()).device)
            self._ensure_cong(device)
            batch_loss = None
            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue
                pred_instances = preds[i].pred_instances.to(device)
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')
                gt = ds.gt_instances.to(device)
                assigner = MaxIoUAssigner(pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner,
                                          min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality)
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue
                logits = pred_instances.logits
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []
                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    choice = assoc[j]
                    node_labels.append(int(gt.labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)
                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)
                # usar logits (pré-softmax) como descritor visual
                vis_feats = node_logits_t

                # Build context inputs
                H_img = inputs.shape[-2]
                W_img = inputs.shape[-1]
                gt_boxes_xyxy = gt.bboxes.tensor
                sel_gt_xyxy = gt_boxes_xyxy[node_img_local_to_valid]
                sp = torch.stack([spatial7_from_xyxy(sel_gt_xyxy[k], W_img, H_img) for k in range(sel_gt_xyxy.size(0))], dim=0)
                # === Consistência de dimensões entre pr/vis_feats/labels/sp/boxes ===
                N_pr   = pr.size(0)
                N_feat = vis_feats.size(0)
                N_lab  = node_labels_t.size(0)
                N_box  = sel_gt_xyxy.size(0)
                N_sp   = sp.size(0)
                min_n = min(N_pr, N_feat, N_lab, N_box, N_sp)
                if not (N_pr == N_feat == N_lab == N_box == N_sp):
                    pr = pr[:min_n]
                    vis_feats = vis_feats[:min_n]
                    node_logits_t = node_logits_t[:min_n]
                    node_labels_t = node_labels_t[:min_n]
                    sel_gt_xyxy = sel_gt_xyxy[:min_n]
                    sp = sp[:min_n]
                Nn = pr.size(0)
                if Nn <= 1:
                    pr_ctx = pr
                else:
                    probs = pr
                    # confidences (neighbors j)
                    pmax = probs.max(dim=-1).values
                    w_conf = (pmax.clamp_min(1e-6) ** self.ctx_conf_beta)
                    # cosine similarity nas probabilidades
                    Pn = F.normalize(pr, p=2, dim=-1)
                    S_prob = torch.mm(Pn, Pn.t())
                    S_prob = (S_prob + 1.0) * 0.5
                    # cosine similarity visual (logits)
                    Fn = F.normalize(vis_feats, p=2, dim=-1)
                    S_feat = torch.mm(Fn, Fn.t())
                    S_feat = (S_feat + 1.0) * 0.5
                    # combinação com pesos configuráveis (com ajuste defensivo de tamanho)
                    if S_prob.shape != S_feat.shape:
                        min_n = min(S_prob.shape[0], S_feat.shape[0])
                        S_prob = S_prob[:min_n, :min_n]
                        S_feat = S_feat[:min_n, :min_n]
                    S = self.ctx_prob_weight * S_prob + self.ctx_feat_weight * S_feat
                    S.fill_diagonal_(0.0)
                    # máscara para suprimir pares visualmente dissimilares
                    S_mask = (S_feat >= self.ctx_sim_min).float()
                    # spatial Gaussian kernel using dest box sizes
                    b = sel_gt_xyxy
                    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
                    cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                    w  = (x2-x1).clamp(min=1e-3); h=(y2-y1).clamp(min=1e-3)
                    DX = (cx.unsqueeze(1) - cx.unsqueeze(0)).abs() / (w.unsqueeze(0) + 1e-3)
                    DY = (cy.unsqueeze(1) - cy.unsqueeze(0)).abs() / (h.unsqueeze(0) + 1e-3)
                    D  = torch.sqrt(DX**2 + DY**2)
                    sigma = self.ctx_dist_sigma
                    Ksp = torch.exp(-D / max(1e-6, sigma))
                    # co-occurrence attenuation between labels
                    li = node_labels_t
                    Cij = co_probs_epoch.to(device)[li][:, li]
                    # final weights W[i,j] (j contributes to i)
                    W = (S * Ksp * Cij) * S_mask
                    W = W * w_conf.unsqueeze(0)
                    W.fill_diagonal_(0.0)
                    den = W.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    pr_ctx = (W @ probs) / den
                pmax = pr.max(dim=-1).values
                alpha_conf = torch.sigmoid(self.ctx_gate_gamma * (pmax - self.trust_thr)).unsqueeze(1)
                alpha = alpha_conf
                pr_mix = alpha * pr + (1.0 - alpha) * pr_ctx
                x_cong = torch.cat([pr_mix, sp], dim=1)

                # Graph with fixed epoch prior
                co_probs = co_probs_epoch.to(device)
                edge_index, edge_attr = self._build_semantic_graph(
                    pr, node_labels_t, co_probs_epoch.to(device), sel_gt_xyxy, W_img, H_img, k=4,
                    features=vis_feats, feat_weight=self.ctx_feat_weight, prob_weight=self.ctx_prob_weight)

                # Train ConG
                self._cong.train()
                logits_cong = self._cong(x_cong, edge_index, edge_attr)
                loss_cong = F.cross_entropy(logits_cong, node_labels_t)
                if batch_loss is None:
                    batch_loss = loss_cong
                else:
                    batch_loss = batch_loss + loss_cong

            if batch_loss is not None:
                self._opt_cong.zero_grad()
                batch_loss.backward()
                self._opt_cong.step()
                steps += 1
                if steps >= self.cong_train_steps:
                    break
        if steps >= self.cong_train_steps and hasattr(runner, 'logger'):
            runner.logger.info(f"[ConG] Trained with fixed epoch co_probs: steps={steps}")

     
        for data_batch in dataloader:
            # forward do detector SEM grad (apenas extrair preds)
            with torch.no_grad():
                data = runner.model.data_preprocessor(data_batch, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                preds = runner.model.my_get_logits(inputs, data_samples, all_logits=True)
                device = (preds[0].pred_instances.scores.device if len(preds) > 0 else next(runner.model.parameters()).device)

            self._ensure_cong(device)
            co_probs = self._compute_co_probs(device)
            # ===== por-amostra do batch =====
            for i, ds in enumerate(data_batch['data_samples']):
                img_path = ds.img_path
                if img_path not in dataset_img_map:
                    continue
                # garanta que TUDO está no mesmo device
                pred_instances = preds[i].pred_instances.to(device)
                # alguns heads expõem 'bboxes'; o assigner aceita 'priors' também
                if 'bboxes' in pred_instances:
                    pred_instances.priors = pred_instances.pop('bboxes')

                gt = ds.gt_instances.to(device)

                # referências locais (já no device)
                bboxes = gt.bboxes
                labels = gt.labels
                priors = pred_instances.priors
                logits = pred_instances.logits  # [Np, C]
                scores_max = logits.softmax(dim=-1).max(dim=-1).values

                # assign predições -> GTs
                assigner = MaxIoUAssigner(pos_iou_thr=self.iou_assigner, neg_iou_thr=self.iou_assigner, min_pos_iou=self.iou_assigner, match_low_quality=self.low_quality)
                assign_result = assigner.assign(pred_instances, gt)
                if assign_result.num_gts == 0:
                    continue

                node_labels = []
                node_logits = []
                node_img_local_to_valid = []

                # mapeamento de instâncias válidas no dataset (para escrita)
                sub_idx, d_idx = dataset_img_map[img_path]
                subds = datasets[sub_idx]
                valid_instance_indices = [idx for idx, inst in enumerate(subds.data_list[d_idx]['instances']) if inst['ignore_flag'] == 0]

                for gt_idx in range(assign_result.num_gts):
                    assoc = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                    if assoc.numel() == 0:
                        continue
                    local_scores = scores_max[assoc]
                    j = int(torch.argmax(local_scores))
                    choice = assoc[j]
                    node_labels.append(int(labels[gt_idx].item()))
                    node_logits.append(logits[choice].view(1, -1))
                    node_img_local_to_valid.append(gt_idx)
                if (len(node_labels) == 0) or (len(node_logits) == 0):
                    continue

                

                node_labels_t = torch.tensor(node_labels, device=device, dtype=torch.long)
                node_logits_t = torch.cat(node_logits, dim=0).to(device)
                pr = node_logits_t.softmax(dim=-1)  # [N,C]
                vis_feats = node_logits_t
                # veto: se a classe predita == GT, não considerar como ruído
                pred_cls_t = pr.argmax(dim=1)                  # [N]
                veto_eq_t = (pred_cls_t == node_labels_t)      # [N] bool



                # rótulos efetivos para consultar co-ocorrência (usa pred quando muito confiante)
                eff_labels_t, _, pmax_t, argmax_t = self._effective_labels_from_logits(node_logits_t, node_labels_t, self.trust_thr)

                H_img = inputs.shape[-2]
                W_img = inputs.shape[-1]
                gt_boxes_xyxy = gt.bboxes.tensor
                sel_gt_xyxy = gt_boxes_xyxy[node_img_local_to_valid]
                sp_list = [spatial7_from_xyxy(sel_gt_xyxy[k], W_img, H_img) for k in range(sel_gt_xyxy.size(0))]
                if len(sp_list) == 0:
                    continue
                sp = torch.stack(sp_list, dim=0)  # [N,7]

                # --- contexto ponderado por confiança + gate por confiança e minoritarismo ---
                Nn = pr.size(0)
                # cosine similarities: probs e visual (logits)
                Pn = F.normalize(pr, p=2, dim=-1)
                S_prob = torch.mm(Pn, Pn.t()); S_prob = (S_prob + 1.0) * 0.5
                Fn = F.normalize(vis_feats, p=2, dim=-1)
                S_feat = torch.mm(Fn, Fn.t()); S_feat = (S_feat + 1.0) * 0.5

                # combinação com pesos configuráveis (com ajuste defensivo de tamanho)
                if S_prob.shape != S_feat.shape:
                    min_n = min(S_prob.shape[0], S_feat.shape[0])
                    S_prob = S_prob[:min_n, :min_n]
                    S_feat = S_feat[:min_n, :min_n]
                S = self.ctx_prob_weight * S_prob + self.ctx_feat_weight * S_feat
                S.fill_diagonal_(0.0)
                # gate visual
                S_mask = (S_feat >= self.ctx_sim_min).float()

                # usar logits como features visuais adiante
                features = vis_feats
                if Nn <= 1:
                    pr_ctx = pr
                else:
                    probs = pr
                    pmax = probs.max(dim=-1).values
                    w_conf = (pmax.clamp_min(1e-6) ** self.ctx_conf_beta)
                    b = sel_gt_xyxy
                    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
                    cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                    w  = (x2-x1).clamp(min=1e-3); h=(y2-y1).clamp(min=1e-3)
                    DX = (cx.unsqueeze(1) - cx.unsqueeze(0)).abs() / (w.unsqueeze(0) + 1e-3)
                    DY = (cy.unsqueeze(1) - cy.unsqueeze(0)).abs() / (h.unsqueeze(0) + 1e-3)
                    D  = torch.sqrt(DX**2 + DY**2)
                    sigma = self.ctx_dist_sigma
                    Ksp = torch.exp(-D / max(1e-6, sigma))
                    li = eff_labels_t
                    Cij = co_probs[li][:, li] if (li.numel() == Nn) else torch.ones_like(S)
                    # Peso final: combinação da similaridade de features, probabilidade e co-ocorrência
                    W = (S * Ksp * Cij) * S_mask
                    W = W * w_conf.unsqueeze(0)
                    W.fill_diagonal_(0.0)
                    den = W.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    pr_ctx = (W @ probs) / den
                # gate: self vs contexto condicionado por confiança e minoritarismo
                probs = pr
                pmax = probs.max(dim=-1).values
                alpha_conf = torch.sigmoid(self.ctx_gate_gamma * (pmax - self.trust_thr)).unsqueeze(1)
                alpha = alpha_conf
                pr_mix = alpha * probs + (1.0 - alpha) * pr_ctx
                x_cong = torch.cat([pr_mix, sp], dim=1)
                # --- Build semantic graph with co-occurrence attributes and feature similarity ---
                # edge_attr agora inclui f_sim como segunda coluna
                edge_index, edge_attr = self._build_semantic_graph(
                    pr, eff_labels_t, co_probs, sel_gt_xyxy, W_img, H_img, k=4,
                    features=vis_feats, feat_weight=self.ctx_feat_weight, prob_weight=self.ctx_prob_weight
                )
                self._cong.eval()
                with torch.no_grad():
                    logits_cong = self._cong(x_cong, edge_index, edge_attr)
                    qc = logits_cong.softmax(dim=-1)  # [N,C]

                old_labels_before_agree = node_labels_t.clone()
                relabeled_pairs = []  # lista de tuples (local_idx, old_gt)
                # --- AGREEMENT RELABEL (modelo & contexto concordam) ---
                if qc.shape[0] == pr.shape[0] and pr.shape[0] > 1:
                    pmax_t = pr.max(dim=1).values               # confiança do modelo por nó
                    pred_cls_t = pr.argmax(dim=1)               # classe do modelo por nó
                    ctx_cls_t = qc.argmax(dim=1)                # classe de contexto por nó
                    agree = (pred_cls_t == ctx_cls_t) & (pmax_t >= self.relabel_thr_ctx)
                    agree_idx = torch.nonzero(agree, as_tuple=False).flatten()
                    for _li in agree_idx.tolist():
                        new_lab = int(pred_cls_t[_li].item())
                        old_lab = int(node_labels_t[_li].item())
                        if new_lab == old_lab:
                            # nada a fazer; não marca como relabel
                            continue
                        # atualiza o tensor de labels do batch
                        node_labels_t[_li] = new_lab
                        relabeled_pairs.append((_li, old_lab))
                        # persiste no dataset para próximas épocas
                        if _li < len(node_img_local_to_valid):
                            gt_idx = node_img_local_to_valid[_li]
                            if gt_idx < len(valid_instance_indices):
                                _valid_idx = valid_instance_indices[gt_idx]
                                inst = subds.data_list[d_idx]['instances'][_valid_idx]
                                updated = False
                                for key in ['bbox_label', 'label', 'labels']:
                                    if key in inst:
                                        inst[key] = new_lab
                                        updated = True
                                if not updated:
                                    inst['labels'] = new_lab
                    # Como alteramos node_labels_t, recomputamos eff_labels_t e veto_eq_t
                    eff_labels_t, _, pmax_t, argmax_t = self._effective_labels_from_logits(node_logits_t, node_labels_t, self.trust_thr)
                    veto_eq_t = (pred_cls_t == node_labels_t)
                    disagree_mask = (pred_cls_t != node_labels_t).detach().cpu().numpy().astype(bool)
                # else:
                #     if hasattr(runner, 'logger'):
                #         runner.logger.warning(f"[ConG] Skip agreement relabel: "
                #                               f"N={pr.shape[0]}, qc={qc.shape[0]} pr={pr.shape[0]} on {os.path.basename(img_path)}")

                eps = 1e-7
                p = pr.clamp_min(eps)
                q = qc.clamp_min(eps)
                # --- filtro contextual + co-ocorrência real ---
                kld = (p * (p.log() - q.log())).sum(dim=1)  # [N]
                kld_np = kld.detach().cpu().numpy()
                # usa KLD absoluto como p_noise (sem normalização por imagem)
                p_noise = kld
                p_noise_np = kld_np

                # --- filtro auxiliar de co-ocorrência real ---
                # para cada nó i (classe li), verifica se existe alguma classe w presente na imagem tal que co(li,w) >= corr_thr_low
                Nn = eff_labels_t.numel()
                if Nn > 1:
                    li = eff_labels_t
                    lj = eff_labels_t
                    C = co_probs[li][:, lj]  # [N,N]
                    C = C.masked_fill(torch.eye(Nn, dtype=torch.bool, device=C.device), 0.0)
                    best, _ = C.max(dim=1)   # [N]
                    best_np = best.detach().cpu().numpy()
                    low_corr_mask = (best < float(self.corr_thr_low))  # True quando baixa correlação com classes presentes
                else:
                    low_corr_mask = torch.zeros_like(kld, dtype=torch.bool)
                low_corr_np = low_corr_mask.detach().cpu().numpy().astype(bool)

                # define corte da parte contextual (KLD normalizado)
                if self.use_percentile and p_noise_np.size > 0:
                    cut = float(np.percentile(p_noise_np, self.percentile))
                else:
                    cut = float(self.thr_noise)

                # --- veto para minoritários plausíveis ---
                # veto_np = np.zeros_like(low_corr_np, dtype=bool)
                # if Nn > 0:
                #     # maioria por frequência na imagem (com rótulos efetivos)
                #     counts = torch.bincount(eff_labels_t, minlength=self.num_classes).float()
                #     majority = int(counts.argmax().item())
                #     # co-ocorrência com a classe majoritária
                #     co_with_major = co_probs[eff_labels_t, majority]  # [N]
                #     co_with_major_np = co_with_major.detach().cpu().numpy()
                #     freq_i = (counts[eff_labels_t] / max(float(eff_labels_t.numel()), 1.0)).detach().cpu().numpy()
                #     minority_np = (freq_i < float(self.rho_min))
                #     strong_cooc_np = (co_with_major_np >= float(self.corr_thr_pos))
                #     veto_np = np.logical_and(minority_np, strong_cooc_np)

                # veto adicional: predição igual ao GT nunca é ruído
                veto_eq_np = veto_eq_t.detach().cpu().numpy().astype(bool)

                # --- W&B: loga alguns exemplos visuais ---
                if self.use_wandb and (wandb is not None):
                    # loga até `wandb_max_images` imagens por época com overlay
                    if self._wandb_img_budget > 0:
                        # Log only if at least one bbox had its label actually changed in this image
                        relabeled_pairs = relabeled_pairs if 'relabeled_pairs' in locals() else []
                        has_relabel = len(relabeled_pairs) > 0
                        should_log = has_relabel
                        try:
                            relabeled_pairs = relabeled_pairs if 'relabeled_pairs' in locals() else []
                            relabeled_set = set([int(i) for (i, _) in relabeled_pairs]) if 'relabeled_pairs' in locals() else set()
                            old_gt_map = {int(i): int(old) for (i, old) in relabeled_pairs} if 'relabeled_pairs' in locals() else {}
                            # --- Merge with Phase-1 relabels (pink) ---
                            phase1_map_for_img = self._phase1_relabels.get(img_path, {})  # {gt_local_idx -> old_lab}
                            relabel_high_set = set()
                            old_gt_map_phase1 = {}
                            # Mapear do índice de nó local (local_idx) para info da Fase-1 via gt_local_idx
                            for local_idx, gt_local_idx in enumerate(node_img_local_to_valid[:len(sel_gt_xyxy)]):
                                if int(gt_local_idx) in phase1_map_for_img:
                                    relabel_high_set.add(int(local_idx))
                                    old_gt_map_phase1[int(local_idx)] = int(phase1_map_for_img[int(gt_local_idx)])
                            # noisy se (KLD alto) E (baixa co-ocorrência real)
                            noisy_kld = (p_noise_np >= cut)
                            noisy_mask = np.logical_and(noisy_kld, low_corr_np)
                            # aplica vetos: minoria plausível e pred==GT
                            # if 'veto_np' in locals() and len(veto_np) == len(noisy_mask):
                            #     noisy_mask = np.logical_and(noisy_mask, np.logical_not(veto_np))
                            if 'veto_eq_np' in locals() and len(veto_eq_np) == len(noisy_mask):
                                noisy_mask = np.logical_and(noisy_mask, np.logical_not(veto_eq_np))
                            if should_log:
                                # reconstrói imagem BGR para desenho usando img_norm_cfg real
                                # Carrega a imagem original em BGR e redimensiona para o img_shape atual do batch
                                # (mantém cores corretas para desenho via OpenCV)
                                meta = getattr(data_samples[i], 'metainfo', {})
                                img_shape = tuple(meta.get('img_shape', (inputs.shape[-2], inputs.shape[-1])))  # (H, W, 3)
                                H_v, W_v = int(img_shape[0]), int(img_shape[1])
                                img_np = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
                                if img_np is None:
                                    # fallback: tenta usar o tensor (pode ter cores trocadas)
                                    norm_cfg = meta.get('img_norm_cfg', {})
                                    mean = norm_cfg.get('mean', [123.675, 116.28, 103.53])
                                    std = norm_cfg.get('std', [58.395, 57.12, 57.375])
                                    to_rgb = norm_cfg.get('to_rgb', True)
                                    img_np = tensor_to_numpy_img(inputs[i].cpu(), mean=mean, std=std, to_rgb=to_rgb)
                                else:
                                    img_np = cv2.resize(img_np, (W_v, H_v), interpolation=cv2.INTER_LINEAR)
                                    # aplica o mesmo flip da pipeline para alinhar as boxes
                                    if bool(meta.get('flip', False)):
                                        fd = str(meta.get('flip_direction', 'horizontal'))
                                        if 'diagonal' in fd:
                                            img_np = cv2.flip(img_np, -1)
                                        else:
                                            if 'horizontal' in fd:
                                                img_np = cv2.flip(img_np, 1)
                                            if 'vertical' in fd:
                                                img_np = cv2.flip(img_np, 0)
                                # desenha GTs associados com cor conforme noise
                                sel_gt_xyxy_np = sel_gt_xyxy.detach().cpu().numpy().astype(int)
                                for local_idx in range(min(len(node_img_local_to_valid), len(p_noise_np))):
                                    x1, y1, x2, y2 = sel_gt_xyxy_np[local_idx].tolist()
                                    is_noisy = bool(noisy_mask[local_idx])
                                    is_kld = bool(noisy_kld[local_idx])
                                    # prioridade de cor: relabel (AZUL/Fase-2) > relabel-high (ROSA/Fase-1) > noisy (VERMELHO) > KLD alto vetado (LARANJA) > discordância (AMARELO) > limpo (VERDE)
                                    if local_idx in relabeled_set:
                                        color = (255, 0, 0)            # AZUL (reanotado pela rede de grafos / Phase-2)
                                    elif local_idx in relabel_high_set:
                                        color = (255, 105, 180)        # ROSA (reanotado na Phase-1: alta confiança do modelo)
                                    elif 'disagree_mask' in locals() and local_idx < len(disagree_mask) and bool(disagree_mask[local_idx]):
                                        color = (0, 255, 255)          # AMARELO (pred != label atual)
                                    else:
                                        color = (0, 255, 0)            # VERDE

                                    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
                                    # obter classes GT/pred, KLD e pmax para overlay
                                    _gt_lab = int(node_labels[local_idx])
                                    _pred_lab = int(pr[local_idx].argmax().item())
                                    _kld = float(kld_np[local_idx])
                                    _pmax = float(pr[local_idx].max().item())
                                    _co = float(best_np[local_idx]) if 'best_np' in locals() and local_idx < len(best_np) else 0.0
                                    # Adiciona classe esperada pelo contexto (ctx) usando qc.argmax(dim=1)
                                    _ctx_lab = int(qc[local_idx].argmax().item()) if qc is not None and local_idx < qc.shape[0] else -1
                                    _v_eq = bool(veto_eq_np[local_idx]) if 'veto_eq_np' in locals() and local_idx < len(veto_eq_np) else False
                                    _old2 = old_gt_map.get(local_idx, None)                 # Phase-2 old label (ConG)
                                    _old1 = old_gt_map_phase1.get(local_idx, None)          # Phase-1 old label (high confidence)
                                    # Preferir mostrar o marcador correspondente à cor aplicada
                                    if local_idx in relabeled_set and _old2 is not None:
                                        rx_suffix = f"|R{_old2}"
                                    elif local_idx in relabel_high_set and _old1 is not None:
                                        rx_suffix = f"|R{_old1}"
                                    else:
                                        rx_suffix = ""
                                    _overlay_txt = f"gt{_gt_lab}|p{_pred_lab}|n{float(p_noise_np[local_idx]):.2f}|k{_kld:.2f}|s{_pmax:.2f}|c{_co:.2f}|ct{_ctx_lab}{rx_suffix}"
                                    cv2.putText(img_np,
                                                _overlay_txt,
                                                (max(0, x1), max(15, y1-4)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                # Converte BGR (OpenCV) → RGB antes de enviar ao W&B
                                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                                self._wandb_imgs.append(
                                    wandb.Image(img_rgb, caption=f"epoch={runner.epoch+1} | {os.path.basename(img_path)}")
                                )
                                self._wandb_img_budget -= 1
                                # Opcional: logar figura pr vs pr_ctx **apenas para nós noisy**
                                # if getattr(self, 'wandb_log_prctx', False):
                                #     noisy_idx = np.where(noisy_mask)[0].tolist()
                                #     if len(noisy_idx) > 0:
                                #         # respeita o limite de nós; seleciona os primeiros noisy
                                #         noisy_idx = noisy_idx[:max(1, getattr(self, 'prctx_max_nodes', 8))]
                                #         idx_t = torch.as_tensor(noisy_idx, device=pr.device, dtype=torch.long)
                                #         pr_sel = pr.index_select(0, idx_t)
                                #         pr_ctx_sel = pr_ctx.index_select(0, idx_t)
                                #         qc_sel = qc.index_select(0, idx_t) if qc is not None else None
                                #         labels_sel = node_labels_t.index_select(0, idx_t)

                                #         cls_names = None
                                #         try:
                                #             cls_names = data_samples[i].metainfo.get('classes', None)
                                #         except Exception:
                                #             cls_names = None
                                #         title = f"pr vs pr_ctx vs qc (noisy only) | {os.path.basename(img_path)}"
                                #         prctx_img = _make_prctx_figure(pr_sel, pr_ctx_sel, qc_sel, labels_sel,
                                #                                        topk=getattr(self, 'prctx_topk', 5),
                                #                                        max_nodes=len(noisy_idx),
                                #                                        class_names=cls_names,
                                #                                        title=title)
                                #         if prctx_img is not None:
                                #             self._wandb_imgs.append(wandb.Image(prctx_img, caption=f"pr_ctx noisy | epoch={runner.epoch+1} | {os.path.basename(img_path)}"))
                                # Extra: logar pr/pr_ctx/qc para casos com discordância modelo x label
                                if getattr(self, 'wandb_log_prctx', False) and 'disagree_mask' in locals():
                                    disagree_idx = np.where(disagree_mask)[0].tolist()
                                    if len(disagree_idx) > 0:
                                        disagree_idx = disagree_idx[:max(1, getattr(self, 'prctx_max_nodes', 8))]
                                        idx_t2 = torch.as_tensor(disagree_idx, device=pr.device, dtype=torch.long)
                                        pr_sel = pr.index_select(0, idx_t2)
                                        pr_ctx_sel = pr_ctx.index_select(0, idx_t2)
                                        qc_sel = qc.index_select(0, idx_t2) if qc is not None else None
                                        labels_sel = node_labels_t.index_select(0, idx_t2)

                                        cls_names = None
                                        try:
                                            cls_names = data_samples[i].metainfo.get('classes', None)
                                        except Exception:
                                            cls_names = None

                                        title2 = f"pr vs pr_ctx vs qc (pred!=label) | {os.path.basename(img_path)}"
                                        prctx_img2 = _make_prctx_figure(pr_sel, pr_ctx_sel, qc_sel, labels_sel,
                                                                        topk=getattr(self, 'prctx_topk', 5),
                                                                        max_nodes=len(disagree_idx),
                                                                        class_names=cls_names,
                                                                        title=title2)
                                        if prctx_img2 is not None:
                                            self._wandb_imgs.append(
                                                wandb.Image(prctx_img2, caption=f"pr_ctx disagree | epoch={runner.epoch+1} | {os.path.basename(img_path)}")
                                            )
                        except Exception as e:
                            if hasattr(runner, 'logger'):
                                runner.logger.warning(f"[W&B] Falha ao montar/registrar imagem: {e}")

                # aplica REWEIGHT (ignore_flag comentado)
                L = min(len(node_img_local_to_valid), len(p_noise_np))
                for local_idx in range(L):
                    gt_idx = node_img_local_to_valid[local_idx]
                    valid_idx = valid_instance_indices[gt_idx] if gt_idx < len(valid_instance_indices) else None
                    if valid_idx is None:
                        continue
                    if (runner.epoch + 1) <= self.warmup_epochs:
                        continue
                    is_lowcorr = bool(low_corr_np[local_idx])
                    # is_veto = bool(veto_np[local_idx]) if 'veto_np' in locals() and local_idx < len(veto_np) else False
                    is_veto_eq = bool(veto_eq_np[local_idx]) if 'veto_eq_np' in locals() and local_idx < len(veto_eq_np) else False
                    #if (float(p_noise_np[local_idx]) >= cut) and is_lowcorr and (not is_veto) and (not is_veto_eq):
                    if (float(p_noise_np[local_idx]) >= cut) and is_lowcorr and (not is_veto_eq):
                        lw = max(0.2, 1.0 - self.cong_alpha * float(p_noise_np[local_idx]))
                        # subds.data_list[d_idx]['instances'][valid_idx]['loss_weight'] = lw
                        pass
                        # subds.data_list[d_idx]['instances'][valid_idx]['ignore_flag'] = 1  # ignorar apenas quando também houver baixa co-ocorrência
                    
                    # pred_label = int(node_labels_t[local_idx].item())
                    # pred_labelv2 = int(eff_labels_t[local_idx].item())
                    # import pdb; pdb.set_trace()
                    # subds.data_list[d_idx]['instances'][valid_idx]['bbox_label'] = pred_label

          

        # Salvar matriz de co-ocorrência como imagem de heatmap usando matplotlib
        try:
            # 1) Salvar heatmap da co-ocorrência
            cooc_dir = os.path.join(runner.work_dir, 'debug_cooc')
            os.makedirs(cooc_dir, exist_ok=True)
            cooc_path = os.path.join(cooc_dir, f'cooc_matrix_epoch{runner.epoch + 1}.png')
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patheffects as _pe
                plt.figure(figsize=(6, 5))
                _co = self._cooc_counts.clone().float()
                _row = _co.sum(dim=1, keepdim=True).clamp_min(1e-6)
                co_probs = (_co / _row)
                co_probs.fill_diagonal_(1.0)
                _vis = co_probs.numpy()
                im = plt.imshow(co_probs.cpu(), cmap='viridis')
                plt.title(f"Co-occurrence Matrix - Epoch {runner.epoch + 1}")
                plt.colorbar(im)
                plt.xlabel("j (given i→j)")
                plt.ylabel("i")

                # Ajusta os eixos para valores inteiros
                n_classes = co_probs.shape[0]
                plt.xticks(range(n_classes), range(n_classes))
                plt.yticks(range(n_classes), range(n_classes))
                plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

                # === NEW: escreve o número de ocorrências (contagem bruta) em cada célula ===
                _co_np = _co.detach().cpu().numpy()
                for i_tick in range(n_classes):
                    for j_tick in range(n_classes):
                        cnt = int(_co_np[i_tick, j_tick])
                        if cnt <= 0:
                            continue
                        # escolhe cor do texto conforme intensidade para contraste
                        txt_color = 'white' if _vis[i_tick, j_tick] > 0.2 else 'white'
                        plt.text(j_tick, i_tick, str(cnt), ha='center', va='center', fontsize=6,
                                 color=txt_color, path_effects=[_pe.withStroke(linewidth=1, foreground='black')])

                plt.tight_layout()
                plt.savefig(cooc_path)
                plt.close()
            except Exception as e_heat:
                if hasattr(runner, 'logger'):
                    runner.logger.warning(f"[Cooc] Falha ao salvar heatmap: {e_heat}")

            # 2) Forçar log no W&B (imagem, tabela e heatmap interativo)
            if getattr(self, 'use_wandb', False) and (wandb is not None) and getattr(self, '_wandb_ready', False):
                log_dict = {}
                if hasattr(self, '_wandb_imgs') and len(self._wandb_imgs) > 0:
                    log_dict['debug_imgs'] = self._wandb_imgs

                # Recalcula co_probs normalizado (CPU→numpy) para tabela/heatmap
                _co_counts = self._cooc_counts.clone().float()
                _row = _co_counts.sum(dim=1, keepdim=True).clamp_min(1e-6)
                _co_probs = (_co_counts / _row)
                _co_probs.fill_diagonal_(1.0)
                _co_probs_np = _co_probs.detach().cpu().numpy()

                # Tenta obter nomes de classes
                class_names = None
                try:
                    # tenta via datasets concatenados
                    while hasattr(dataset, 'dataset'):
                        dataset_ = dataset.dataset
                    # fallback: tenta pelo primeiro subdataset
                    if 'datasets' in dir(dataset) and len(dataset.datasets) > 0:
                        sub0 = dataset.datasets[0]
                        if hasattr(sub0, 'METAINFO') and isinstance(sub0.METAINFO, dict):
                            class_names = sub0.METAINFO.get('classes', None)
                except Exception:
                    class_names = None

                # Constrói Tabela (i,j,value) para o heatmap
                data_rows = []
                C = int(_co_probs_np.shape[0])
                for i in range(C):
                    for j in range(C):
                        name_i = class_names[i] if (class_names is not None and i < len(class_names)) else str(i)
                        name_j = class_names[j] if (class_names is not None and j < len(class_names)) else str(j)
                        data_rows.append([name_i, name_j, float(_co_probs_np[i, j])])
                cooc_table = wandb.Table(data=data_rows, columns=["i", "j", "value"])  # tabela completa

                # Heatmap interativo no W&B
                try:
                    cooc_heatmap = wandb.plot.heatmap(cooc_table, x="j", y="i", value="value",
                                                      title=f"Co-occurrence (row-normalized) - Epoch {runner.epoch + 1}")
                    log_dict['cooc_heatmap'] = cooc_heatmap
                except Exception as e_hm:
                    if hasattr(runner, 'logger'):
                        runner.logger.warning(f"[W&B] Falha ao criar heatmap: {e_hm}")

                # === NEW: também loga a matriz de contagens brutas ===
                _co_counts_np = _co_counts.detach().cpu().numpy()
                data_rows_cnt = []
                for i_idx in range(C):
                    for j_idx in range(C):
                        name_i = class_names[i_idx] if (class_names is not None and i_idx < len(class_names)) else str(i_idx)
                        name_j = class_names[j_idx] if (class_names is not None and j_idx < len(class_names)) else str(j_idx)
                        data_rows_cnt.append([name_i, name_j, int(_co_counts_np[i_idx, j_idx])])
                cooc_counts_table = wandb.Table(data=data_rows_cnt, columns=["i", "j", "count"])  # contagens
                try:
                    cooc_counts_heatmap = wandb.plot.heatmap(cooc_counts_table, x="j", y="i", value="count",
                                                             title=f"Co-occurrence COUNTS - Epoch {runner.epoch + 1}")
                    log_dict['cooc_counts_heatmap'] = cooc_counts_heatmap
                except Exception as e_hm2:
                    if hasattr(runner, 'logger'):
                        runner.logger.warning(f"[W&B] Falha ao criar heatmap de contagens: {e_hm2}")
                log_dict['cooc_counts_table'] = cooc_counts_table

                # Anexa imagem PNG gerada
                if os.path.exists(cooc_path):
                    log_dict['cooc_matrix'] = wandb.Image(cooc_path)
                # Anexa a tabela numérica
                log_dict['cooc_table'] = cooc_table

                if len(log_dict) > 0:
                    wandb.log(log_dict, commit=True)
                    # esvazia o buffer de imagens para próxima época
                    if 'debug_imgs' in log_dict:
                        self._wandb_imgs.clear()
                    if hasattr(runner, 'logger'):
                        runner.logger.info(f"[W&B] Imagens, cooc_table e cooc_heatmap logados na epoch {runner.epoch + 1}")
        except Exception as e_final:
            if hasattr(runner, 'logger'):
                runner.logger.warning(f"[W&B] Falha no log final da epoch: {e_final}")

        
        




@HOOKS.register_module()
class MyHookFilterPredGT_Class_Relabel(Hook):

    def __init__(self, reload_dataset=False, relabel_conf=0.95, double_thr = 2, filter_conf=0.8, filter_warmup=0, iou_assigner=0.5, low_quality=False, filter_thr=0.7, numGMM=2, filter_type = 'pred', group = False, selcand='max'):
        """
        Hook personalizado para relabeling de amostras.

        Args:
            reload_dataset (bool): Indica se o dataset deve ser recarregado a cada época.
            my_value (int): Um valor de exemplo para demonstrar parâmetros personalizados.
        """
        self.reload_dataset = reload_dataset
        self.relabel_conf = relabel_conf
        self.double_thr = double_thr
        self.filter_conf = filter_conf
        self.filter_warmup = filter_warmup
        self.iou_assigner = iou_assigner
        self.low_quality = low_quality
        self.filter_thr = filter_thr
        self.numGMM = numGMM
        self.filter_type = filter_type
        self.group = group
        self.selcand = selcand
        
        
        

    def before_train_epoch(self, runner, *args, **kwargs):
        
        """
        Atualiza os labels de classe (sem alterar os bounding boxes) a cada 100 iterações.
        """
        # import pdb; pdb.set_trace()
        #if (runner.epoch + 1) % 1 != 0:  
        if (runner.epoch + 1) >0 :  
        # if (runner.epoch + 1) >=5 :  

            print(f"[EPOCH]- runner.epoch: {runner.epoch}")
            print(f"[DEBUG] Antes do hook - iter: {runner.iter}")
            dataloader = runner.train_loop.dataloader
            dataset = dataloader.dataset

            #reload_dataset = True
            # my_value = 3
            #reload_dataset = getattr(runner.cfg, 'reload_dataset', False)  
            #my_value =  getattr(runner.cfg, 'my_value', 10)  
            reload_dataset = self.reload_dataset
            relabel_conf = self.relabel_conf
            
            # import pdb; pdb.set_trace()
            if reload_dataset:
                runner.train_loop.dataloader.dataset.dataset.datasets[0]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[0].full_init()

                runner.train_loop.dataloader.dataset.dataset.datasets[1]._fully_initialized = False
                runner.train_loop.dataloader.dataset.dataset.datasets[1].full_init()

            # Desencapsular RepeatDataset até o ConcatDataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset

            if not hasattr(dataset, 'datasets'):
                raise ValueError("Esperado um ConcatDataset, mas dataset não tem atributo 'datasets'.")

            datasets = dataset.datasets  # Lista de datasets dentro do ConcatDataset

            assigner = MaxIoUAssigner(
                pos_iou_thr=self.iou_assigner,
                neg_iou_thr=self.iou_assigner,
                min_pos_iou=self.iou_assigner,
                match_low_quality=self.low_quality
            )

            # Criar um dicionário para acesso rápido ao índice das imagens no dataset
            dataset_img_map = {data_info['img_path']: (sub_dataset_idx, data_idx)
                            for sub_dataset_idx, sub_dataset in enumerate(datasets)
                            if hasattr(sub_dataset, 'data_list')
                            for data_idx, data_info in enumerate(sub_dataset.data_list)}
            
            # Dicionário para armazenar os scores médios por GT sem assumir sequência
            allbb_preds_map = defaultdict(dict)
            all_gt_idx_map = defaultdict(list)  # Mapeamento dos GTs reais
            all_pred_instances_map = defaultdict(dict)
            # all_inputs_map = defaultdict(dict)

            num_classes = 20
            all_classes_preds_map=[]
            all_classes_gt_idx_map=[]
            for i in range(num_classes):
                all_classes_preds_map.append(defaultdict(list))
                all_classes_gt_idx_map.append(defaultdict(list))
            

            count_gts = 0
            sanity_count = 0
            global_index_counter = 0

            
            
            # Processar o batch de dados
            for batch_idx, data_batch in enumerate(dataloader):
                with torch.no_grad():
                    #data = runner.model.data_preprocessor(data_batch, False)
                    data = runner.model.data_preprocessor(data_batch, True)
                    
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                    
                    #predictions_pred = runner.model.my_get_preds(inputs, data_samples)
                    predictions_pred = runner.model.my_get_logits(inputs, data_samples,all_logits=True)
                
                    #temp_filipe

                for i, data_sample in enumerate(data_batch['data_samples']): #before pre_process
                # for i, data_sample in enumerate(data['data_samples']):   #after pre_process
                    img_path = data_sample.img_path
                    
                    if img_path not in dataset_img_map:
                        print(f"[WARNING] Não foi possível localizar a amostra no dataset: {img_path}")
                        continue

                    # Criar `pred_instances` diretamente a partir das predições do `mode='predict'`
                    pred_instances = predictions_pred[i].pred_instances

                    # Pegar bounding boxes preditas
                    pred_instances.priors = pred_instances.pop('bboxes')

                    # Obter os ground truths
                    gt_instances = data_sample.gt_instances
                    gt_labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que os GTs estão no mesmo dispositivo das predições
                    gt_instances.bboxes = gt_instances.bboxes.to(pred_instances.priors.device)
                    gt_instances.labels = gt_instances.labels.to(pred_instances.priors.device)

                    # Garantir que as predições também estão no mesmo dispositivo
                    pred_instances.priors = pred_instances.priors.to(pred_instances.priors.device)
                    pred_instances.labels = pred_instances.labels.to(pred_instances.priors.device)
                    pred_instances.scores = pred_instances.scores.to(pred_instances.priors.device)
                    pred_instances.logits = pred_instances.logits.to(pred_instances.priors.device)

                    all_pred_instances_map[img_path] = pred_instances
                    

                    assign_result = assigner.assign(pred_instances, gt_instances)

                    updated_labels = gt_labels.clone()

                    # Criar lista de GTs para esta imagem
                    all_gt_idx_map[img_path] = []
                    count_gts += assign_result.num_gts

                    
                    
                    
                    # sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    # sub_dataset = datasets[sub_dataset_idx]
                    # import pdb; pdb.set_trace()
                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/sub_{os.path.basename(img_path)}')  
                    # desenhar_bboxesv3(img_path_real,inputs[i], gt_instances, save_path=f'debug_imgs/gt_{os.path.basename(img_path)}')  
                    
                    # if  len(data_samples[i].ignored_instances.labels)>0:
                    #     import pdb; pdb.set_trace()
                    

                    # **Vetorização do processo de associação**
                    for gt_idx in range(assign_result.num_gts):
                        associated_preds = assign_result.gt_inds.eq(gt_idx + 1).nonzero(as_tuple=True)[0]
                        sanity_count += 1
                        
                        if associated_preds.numel() == 0:
                            continue 

                        # Calcular a média dos scores das bounding boxes associadas a este `gt_idx`
                        # mean_score = pred_instances.scores[associated_preds].mean().item()
                        

                        # max_score = pred_instances.scores[associated_preds].max().item()
                        given_max_score = pred_instances.scores[associated_preds].max().item()
                        # max_logit = pred_instances.logits[associated_preds].max().item()
                        # gt_logit = pred_instances.logits[associated_preds][updated_labels[gt_idx]]
                        logits_associated = pred_instances.logits[associated_preds] 
                        myscores = torch.softmax(logits_associated ,dim=-1)

                        
                        # import pdb;pdb.set_trace()
                        # all_pred_instances_map[img_path]['priors']=pred_instances.priors[associated_preds]
                        # all_pred_instances_map[img_path]['scores']=pred_instances.scores[associated_preds]
                        # all_pred_instances_map[img_path]['labels']=pred_instances.labels[associated_preds]
                        # import pdb;pdb.set_trace()
                        # myscores_pred =  myscores.max(dim=1).values.item()

                        # quero pegar o logit_gt da amostra que tem maior logit_pred
                        # import pdb; pdb.set_trace()
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        mylogits_gt = logits_associated[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        max_score_val = myscores.max()
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()
                        myious = assign_result.max_overlaps[associated_preds]
                        max_iou_idx = torch.argmax(myious)
                        amostra_id_iou, classe_id_iou = divmod(max_iou_idx.item(), myscores.size(1))
                        score_gt_max_iou = myscores_gt[amostra_id_iou].cpu().item()
                        score_pred_max_iou = myscores[amostra_id_iou, classe_id].cpu().item()

                        # Compute APS
                        row_scores = myscores[amostra_id]  # pega apenas a linha
                        p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                        S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        # p_y = myscores[amostra_id, classe_id].item()
                        # S_APS_score = -myscores[myscores >= p_y].sum().item()


                            

                        # import pdb; pdb.set_trace()
                        # allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score
                         
                        max_score_idx_local = torch.argmax(myscores_gt)
                        #----> remover depois do debug
                        #max_logit_idx_gt = associated_preds[max_score_idx_local]
                        max_logit_idx_gt = associated_preds[amostra_id]
                        logit_gt_max_pred = mylogits_gt[amostra_id].cpu().item()
                        


        
                        gt_logits = logits_associated[:,updated_labels[gt_idx].cpu().item()]

                        #---> logit com maior valor [OFICIAL]-voltar depois do debug
                        # max_logit = gt_logits.max().item()
                        # max_logit_idx_local = torch.argmax(gt_logits)
                        # max_logit_idx = associated_preds[max_logit_idx_local]

                        


                        #logit com maior IoU
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()

                        # Pegar idx local do maior IoU
                        max_iou_values = assign_result.max_overlaps[associated_preds]
                        max_iou_idx_local = torch.argmax(max_iou_values)
                        max_iou_idx = associated_preds[max_iou_idx_local]

                        # Vetor contendo ambos (pode repetir se for o mesmo)
                        important_associated_ids = torch.tensor([max_logit_idx_gt.item(), max_iou_idx.item()])
                        important_associated_ids = torch.unique(important_associated_ids)

                        # import pdb; pdb.set_trace()


                        # gt_logits = [gt_logit[updated_labels[gt_idx]] for gt_logit in logits_associated]
                        
                        # Armazenar `gt_idx` na ordem correta
                        all_gt_idx_map[img_path].append(gt_idx)

                        # Armazenar `mean_score` corretamente
                        #---->[original]
                        # allbb_preds_map[img_path][gt_idx] = {'pred':max_logit, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                        #---->[temporario-filipe-debug] - remover depois do debug

                        if self.selcand == 'max':
                            allbb_preds_map[img_path][gt_idx] = {'pred': score_gt_max_pred, 'logit':logit_gt_max_pred , 
                                                             'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                             'aps':S_APS_score,
                                                              'max_pred':max_score_val.cpu().item(), 'pred_label': classe_id,
                                                                'filtered':False}
                        elif self.selcand == 'iou':
                            allbb_preds_map[img_path][gt_idx] = {'pred': score_gt_max_iou, 'logit':logit_gt_max_pred , 
                                                             'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter, 
                                                             'aps':S_APS_score,
                                                              'max_pred':score_pred_max_iou, 'pred_label': classe_id_iou,
                                                                'filtered':False}
                        
                        else:
                            print("Warning: selcand deve ser 'max' ou 'iou'. Usando 'max' por padrão.")
                        global_index_counter += 1



                        

                        #-->original   
                        #confident_preds =  associated_preds[pred_instances.scores[associated_preds] > relabel_conf]
                        #-->temporario-filipe-debug - remover depois
                        # import pdb; pdb.set_trace()

                        

                        confident_preds =  associated_preds[myscores.max(dim=1).values> relabel_conf]
                        

                        
                        if (batch_idx <2):

                            if confident_preds.numel() > 0:
                                # save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_relabeled.jpg'
                                # desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                pass
                            
                            
                                
                            elif len(associated_preds) > 1 and (max_score_val > 0.45) :
                                labels_group = myscores.argmax(dim=1)
                            
                                # _, qtd = Counter(labels_group.tolist()).most_common(1)[0]
                                most_common_label, qtd = Counter(labels_group.tolist()).most_common(1)[0]

                                scores_most_common = myscores[:,most_common_label]
                                confident_most_common =  associated_preds[scores_most_common> 0.45]
                                
                                # import pdb; pdb.set_trace()
                                # verifica se a quantidade é maior que 50% do total
                                if qtd > (len(associated_preds) / 2)  and len(confident_most_common) > 2:
                                    save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_grouped.jpg'
                                    #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                    desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, associated_preds,  updated_labels[gt_idx].cpu().item(), save_path=save_path)  

                            else:
                                                                
                                save_path = runner.work_dir + f'/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg'
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_not_relabel.jpg')  
                                #desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, important_associated_ids, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                                desenhar_bboxesv3_pred(inputs[i], gt_instances,pred_instances,assign_result.max_overlaps, associated_preds, updated_labels[gt_idx].cpu().item(), save_path=save_path)  
                            

                        
                        
                                  

                        if confident_preds.numel() > 0:
                            
                            #---> original
                            # pred_labels_confident = pred_instances.labels[confident_preds]
                            #---> temporario-filipe-debug - remover depois do debug
                            #pred_labels_confident= pred_instances['logits'][important_associated_ids].argmax(dim=1)
                            pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)


                            most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        elif self.group and len(associated_preds) > 1 and (max_score_val > 0.45):
                            labels_group = myscores.argmax(dim=1)
                            most_common_label, qtd = Counter(labels_group.tolist()).most_common(1)[0]

                            scores_most_common = myscores[:,most_common_label]
                            confident_most_common =  associated_preds[scores_most_common> 0.45]
                            # import pdb; pdb.set_trace()

                            # verifica se a quantidade é maior que 50% do total
                            if qtd > (len(associated_preds) / 2) and len(confident_most_common) > 2:
                                
                                most_common_label = most_common_label
                            # senao verifica se tem algum elemento com score maior que o threshold
                            elif confident_preds.numel() > 0:
                                pred_labels_confident= pred_instances['logits'][confident_preds].argmax(dim=1)
                                most_common_label = Counter(pred_labels_confident.tolist()).most_common(1)[0][0]

                        # elif confident_preds.numel() == 0:
                        else:
                            continue  


                        updated_labels[gt_idx] = most_common_label
                        #atualiza também no mapeamento
                        allbb_preds_map[img_path][gt_idx]['gt_label'] =  most_common_label
                        #atualiza max logit-gt
                        gt_logits = logits_associated[:,most_common_label]
                        # max_logit = gt_logits[assign_result.max_overlaps[associated_preds].argmax()].item()
                        max_logit = gt_logits.max().item()

                        ###-->debug -remover depois
                        # myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        # max_score = myscores_gt.max().item()
                        # max_score_val = myscores.max()
                        # Depois do relabel, o max score vai ser o da própria predição. Como está atualizado, vai coincidir com o gt
                        myscores_gt = myscores[:,updated_labels[gt_idx].cpu().item()]
                        max_score_idx = myscores.argmax()
                        amostra_id, classe_id = divmod(max_score_idx.item(), myscores.size(1))
                        score_gt_max_pred = myscores_gt[amostra_id].cpu().item()

                        # Compute APS
                        # p_y = myscores[amostra_id, classe_id].item()
                        # S_APS_score = -myscores[myscores >= p_y].sum().item()
                        # row_scores = myscores[amostra_id]  # pega apenas a linha
                        # p_y = row_scores[classe_id].item()
                        # S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        row_scores = myscores[amostra_id]  # pega apenas a linha
                        p_y = row_scores[updated_labels[gt_idx].cpu().item()].item()
                        S_APS_score = -row_scores[row_scores > p_y].sum().item()
                        # if S_APS_score < -1:
                        #     print(S_APS_score)
                        #     print("a")
                        #     import pdb; pdb.set_trace()
                        #     print("ok")
                        allbb_preds_map[img_path][gt_idx]['aps'] = S_APS_score

                        #
                        #-->[original]
                        #allbb_preds_map[img_path][gt_idx]['pred'] = max_logit
                        #-->[temporario-filipe-debug] - remover depois do debug
                        allbb_preds_map[img_path][gt_idx]['pred'] = score_gt_max_pred
                        allbb_preds_map[img_path][gt_idx]['filtered'] = True



                        
                        

                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    # Criar uma lista para mapear os índices corretos das instâncias que NÃO são ignoradas
                    valid_instance_indices = [idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances']) if inst['ignore_flag'] == 0]

                    # Atualizar os labels corretamente usando o mapeamento
                    for gt_idx, valid_idx in enumerate(valid_instance_indices):
                        sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = updated_labels[gt_idx].item()
                        # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = 10
            

            # só faz filter depois do warmup
            if (runner.epoch + 1) >= self.filter_warmup:

                # filtering
                #Unir todas as probabilidades das imagens e treinar o GMM**
                # all_scores = np.array([
                #     score['pred'] for img_scores in allbb_preds_map.values()
                #     for score in img_scores.values()
                # ]).reshape(-1, 1)
                # import pdb; pdb.set_trace()
                print("[DEBUG1]: Entrou no filtro")
                all_classes_low_confidence_scores_global_idx = []

                for c in range(num_classes):
                    # c_class_scores = np.array([
                    #     score['pred'] for img_scores in allbb_preds_map.values()
                    #     for score in img_scores.values() if score['gt_label'] == c
                    # ]).reshape(-1, 1)

                    # if runner.epoch + 1 >1:
                    #     import pdb; pdb.set_trace()

                    scores = np.array([])
                    # img_indexes = np.array([])
                    scores_dict = {'pred': np.array([]),
                                   'logit': np.array([]),
                                   'aps': np.array([])}
                    low_conf_dict = {'pred': np.array([]),
                                   'logit': np.array([]),
                                   'aps': np.array([])}
                    img_paths = np.array([], dtype=object)
                    c_global_indexes = np.array([])

                    for img_path, img_info in allbb_preds_map.items():
                        #for pred, gt_label, index  in img_info.values():
                        for temp_gt_idx, values in img_info.items():
                            #{'pred':max_score, 'gt_label':gt_labels[gt_idx].item(), 'global_index_counter':global_index_counter}
                            #if gt_label== c:
                            if values['gt_label'] == c:
                                scores_dict['pred'] = np.append(scores_dict['pred'], values['pred'])
                                scores_dict['logit'] = np.append(scores_dict['logit'], values['logit'])
                                scores_dict['aps'] = np.append(scores_dict['aps'], values['aps'])

                                # calculando como pred apenas para ver depois se len(scores) == 0
                                scores = np.append(scores, values['pred'])

                                
                                # if self.filter_type == 'pred':
                                #     scores = np.append(scores, values['pred'])
                                # elif self.filter_type == 'logit':
                                #     scores = np.append(scores, values['logit'])
                                # elif self.filter_type == 'aps':
                                #     scores = np.append(scores, values['aps'])

                                c_global_indexes = np.append(c_global_indexes, values['global_index_counter'])
                                if img_path not in img_paths:
                                    img_paths = np.append(img_paths, img_path)
                    
                    if len(scores) == 0:
                        print(f"Aviso: Nenhuma amostra encontrada para a classe {c}.")
                        continue

                    
                    

                    # low_confidence_scores = calculate_gmm(c_class_scores, n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                                
                    low_conf_dict['pred'] = calculate_gmm(scores_dict['pred'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    low_conf_dict['logit'] = calculate_gmm(scores_dict['logit'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    low_conf_dict['aps'] = calculate_gmm(scores_dict['aps'].reshape(-1, 1), n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)

                    if self.filter_type == 'pred':
                        scores = scores_dict['pred']
                        low_confidence_scores = low_conf_dict['pred']
                    elif self.filter_type == 'logit':
                        scores = scores_dict['logit']
                        low_confidence_scores = low_conf_dict['logit']
                    elif self.filter_type == 'aps':
                        scores = scores_dict['aps']
                        low_confidence_scores = low_conf_dict['aps']


                    # print("[DEBUG1.5]: INICIANDO GMM")

                    c_class_scores = c_class_scores = scores.reshape(-1, 1)


                    #gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    # gmm = GaussianMixture(n_components=self.numGMM, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
                    # gmm.fit(c_class_scores)

                    # # Identificar o cluster com menor média (baixa confiança)
                    # low_confidence_component = np.argmin(gmm.means_)
                    # low_confidence_scores = gmm.predict_proba(c_class_scores)[:, low_confidence_component]


                    threshold = self.filter_conf
                
                    low_confidence_indices = np.where(low_confidence_scores > threshold)[0]
                    all_classes_low_confidence_scores_global_idx.extend(c_global_indexes[low_confidence_indices])

                    

                    

                    # draw pred hist
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_pred_ep{runner.epoch + 1}.png"
                    low_confidence_indices_temp = np.where(low_conf_dict['pred'] > threshold)[0]
                    draw_score_histogram(scores_dict['pred'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_conf)

                    # draw logit hist
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_logit_ep{runner.epoch + 1}.png"
                    low_confidence_indices_temp = np.where(low_conf_dict['logit'] > threshold)[0]
                    draw_score_histogram(scores_dict['logit'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_conf)

                    # draw aps hist
                    save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_aps_ep{runner.epoch + 1}.png"
                    low_confidence_indices_temp = np.where(low_conf_dict['aps'] > threshold)[0]
                    draw_score_histogram(scores_dict['aps'].reshape(-1, 1), low_confidence_indices_temp, save_path, runner.epoch + 1, c , threshold=self.filter_conf)

                    # Dentro do seu for c in range(num_classes):
                    # import pdb; pdb.set_trace()
                    # ... após o fit do GMM
                    # plt.figure()
                    # plt.hist(c_class_scores, bins=50, color='blue', alpha=0.7)
                    # # Adiciona linha vertical que separa as duas classes com base no GMM
                    # plt.axvline(x=score_cutoff, color='red', linestyle='--', linewidth=2,label=f'GMM Cutoff: {score_cutoff:.2f}')
                    # # plt.legend()
                    # plt.title(f'Classe {c} - Histograma de scores (época {runner.epoch + 1} - {len(low_confidence_indices)/len(c_class_scores):.2f} amostras)')
                    # plt.xlabel('Score')
                    # plt.ylabel('Frequência')
                    # plt.grid(True)
                    # plt.tight_layout()
                    # # Criar pasta, se necessário
                    # save_path = runner.work_dir+f"/debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png"
                    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    # #plt.savefig(f"debug_hist/class_{c}_hist_ep{runner.epoch + 1}.png")
                    # plt.savefig(save_path)
                    # plt.close()
                   

                # print("[DEBUG2]: SAIU do GMM")
                # if len(all_scores) > 0:
                    

                # **3️⃣ Atualizar `ignore_flag=1` nas bounding boxes com baixa confiança**
                # index = 0
                # import pdb; pdb.set_trace()
                my_counter = 0
                for img_path, gt_scores in allbb_preds_map.items():
                    sub_dataset_idx, dataset_data_idx = dataset_img_map[img_path]
                    sub_dataset = datasets[sub_dataset_idx]

                    

                    valid_instance_indices = [
                        idx for idx, inst in enumerate(sub_dataset.data_list[dataset_data_idx]['instances'])
                        if inst['ignore_flag'] == 0
                    ]

                    # **Mapear GTs reais na ordem correta**
                    gt_idx_list = all_gt_idx_map[img_path]  

                    # import pdb; pdb.set_trace()

                    # img_path_real = img_path
                    # desenhar_bboxes(img_path_real, sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path)}')  
                    # save_path = f'debug_imgs/epoch{runner.epoch+1}_{os.path.basename(img_path)}'
                    # import pdb; pdb.set_trace()
                    

                    for gt_idx in gt_idx_list:
                    # for gt_idx in valid_instance_indices:

                        related_global_index = allbb_preds_map[img_path][gt_idx]['global_index_counter']  

                        

                        #if index in low_confidence_indices:
                        #if related_global_index in all_classes_low_confidence_scores_global_idx:
                        # if low confidence and not too high confident
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.relabel_conf):
                        #if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.5):
                        
                        # if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < 0.3):
                        if (related_global_index in all_classes_low_confidence_scores_global_idx) and (allbb_preds_map[img_path][gt_idx]['pred'] < self.filter_thr):
                            if my_counter<5:

                                                              
                                import shutil

                                # Tenta encontrar o arquivo com sufixo '_relabel.jpg' ou '_not_relabel.jpg'
                                
                                base_prefix = f"{runner.work_dir}/debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}"
                                possible_suffixes = ["_relabeled.jpg", "_not_relabel.jpg", "_grouped.jpg"]

                                for suffix in possible_suffixes:
                                    
                                    base_debug_path = base_prefix + suffix
                                    if os.path.exists(base_debug_path):
                                        my_counter+=1 
                                        filtered_debug_path = base_debug_path[:-4] + "_filtered.jpg"
                                        # if suffix == "_relabeled.jpg":
                                        #     import pdb; pdb.set_trace()
                                        shutil.copy(base_debug_path, filtered_debug_path)
                                        print(f"[INFO] Cópia criada: {filtered_debug_path}")
                                        break  # Para no primeiro que encontrar 

                                # desenhar_bboxesv3_filtered(all_inputs_map[img_path], sub_dataset.data_list[dataset_data_idx]['instances'], save_path=f'debug_imgs/{os.path.basename(img_path[:-4])}_ep{runner.epoch + 1}_gt{gt_idx}_filtered.jpg')  
                            # Encontrar `valid_idx` correspondente ao `gt_idx`
                            # if gt_idx in gt_idx_list:
                            #[ME PARECE ERRADO]
                            # valid_idx = valid_instance_indices[gt_idx_list.index(gt_idx)]
                            #[TESTAR ESSE]
                            # import pdb; pdb.set_trace()
                            valid_idx = valid_instance_indices[gt_idx]

                            # self.double_thr
                            if allbb_preds_map[img_path][gt_idx]['max_pred'] >= self.double_thr:
                                #update
                                sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox_label'] = allbb_preds_map[img_path][gt_idx]['pred_label']
                            else:    
                                #filtra
                                sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1

                            
                            # import pdb; pdb.set_trace()
                            # sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['ignore_flag'] = 1
                            # sub_dataset.data_list[dataset_data_idx]['instances'][gt_idx]['ignore_flag'] = 1
                                #print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, GT: {gt_idx}")
                                #bbox = sub_dataset.data_list[dataset_data_idx]['instances'][valid_idx]['bbox']
                                # print(f"[UPDATE] ignore_flag=1 atualizado para img: {img_path}, BBOX: {bbox} GT: {gt_idx}")
                        
                    
                                
                                
                            
                        # index += 1
                        

            print(f"[DEBUG] Atualização finalizada para a época {runner.epoch + 1}")

# ==========================================
# AB vs APBT（嚴謹 Group Split；內建自動分組；單一骨架）
# 用父資料夾名稱作為 group。若每類群組不足，自動複製成 K 組新資料夾後再評估。
# ==========================================

from PIL import Image
import os, glob, re, time, hashlib, shutil
import numpy as np
import torch, torchvision
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ===== 你只要改這裡（骨架名稱）======
BACKBONE_NAME = "efficientnet_b3"
# 可選："efficientnet_b0", "efficientnet_b3", "mobilenet_v3_large", "resnet50", "vit_b_16"
# ===================================

# 其他可調參數
SRC_DIR = "/content/my_images"     # 原始資料夾（可含子資料夾）
TEST_SIZE = 0.3
RANDOM_SEED = 42
K_GROUPS_PER_CLASS = 5            # 若群組不足，重組時每類分成 K 個 group（ab_g0.., apbt_g0..）
torch.set_grad_enabled(False)

# ---------- 工具：讀圖（遞迴 + 大小寫副檔名） ----------
EXISTS_EXTS = ["jpg","jpeg","png","webp","bmp","JPG","JPEG","PNG","WEBP","BMP"]
def list_images_recursively(base_dir):
    files_list = []
    for ext in EXISTS_EXTS:
        files_list.extend(glob.glob(os.path.join(base_dir, f"**/*.{ext}"), recursive=True))
    return sorted(files_list)

# ---------- 類別貼標：AB=0 / APBT=1 ----------
def is_ab(name):
    n = name.lower()
    if "american_bulldog" in n or "am_bulldog" in n:
        return True
    # 若資料很乾淨也可放寬 'bulldog'，此處保守起見不放寬以避免誤收
    return False

def is_apbt(name):
    n = name.lower()
    return ("american_pit_bull_terrier" in n) or ("apbt" in n) or ("pitbull" in n) or ("pit_bull" in n)

def label_or_none(path):
    b = os.path.basename(path)
    if is_apbt(b): return 1
    if is_ab(b):   return 0
    return None

# ---------- 來源群組：父資料夾名 ----------
def folder_group(path):
    parent = os.path.basename(os.path.dirname(path))
    return parent if parent else "root"

def count_groups_per_class(paths, labels):
    """回傳每類的群組數（以父資料夾為 group）"""
    groups = np.array([folder_group(p) for p in paths])
    g_ab  = set(groups[np.where(labels==0)[0]])
    g_apb = set(groups[np.where(labels==1)[0]])
    return len(g_ab), len(g_apb)

# ---------- 若群組不足，自動重組到新資料夾 ----------
def hash_bucket(stem, k):
    h = hashlib.sha1(stem.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % k

def ensure_grouped_folder(src_dir, k_groups=K_GROUPS_PER_CLASS):
    paths = list_images_recursively(src_dir)
    if len(paths) == 0:
        raise RuntimeError(f"⚠️ 在 {src_dir} 找不到影像")
    # 先貼標，濾掉無法貼標的
    usable = []
    y = []
    for p in paths:
        lbl = label_or_none(p)
        if lbl is not None:
            usable.append(p)
            y.append(lbl)
    if len(usable) == 0:
        raise RuntimeError("⚠️ 無可用影像（檔名需包含 american_bulldog / american_pit_bull_terrier / apbt / pitbull / pit_bull）")
    y = np.array(y)
    # 檢查目前父資料夾群組數
    ab_groups, apbt_groups = count_groups_per_class(usable, y)
    if ab_groups >= 2 and apbt_groups >= 2:
        # 直接用原資料夾
        return src_dir, "資料夾群組模式（原樣）"
    # 否則自動重組
    dst_dir = f"{src_dir.rstrip('/')}_grouped_k{k_groups}"
    os.makedirs(dst_dir, exist_ok=True)
    for g in range(k_groups):
        os.makedirs(os.path.join(dst_dir, f"ab_g{g}"), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, f"apbt_g{g}"), exist_ok=True)

    moved = {"ab":0, "apbt":0}
    for p, lbl in zip(usable, y):
        base = os.path.basename(p)
        g = hash_bucket(base, k_groups)
        sub = f"ab_g{g}" if lbl==0 else f"apbt_g{g}"
        shutil.copy2(p, os.path.join(dst_dir, sub, base))
        moved["ab" if lbl==0 else "apbt"] += 1

    mode = f"資料夾群組模式（自動重組，K={k_groups}）"
    return dst_dir, mode

# ---------- 載入骨幹 ----------
def load_backbone(name):
    if name == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1"); m.classifier = torch.nn.Identity()
        img_size = 224
    elif name == "efficientnet_b3":
        m = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1"); m.classifier = torch.nn.Identity()
        img_size = 300  # 重要：B3 用 300
    elif name == "mobilenet_v3_large":
        m = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2"); m.classifier = torch.nn.Identity()
        img_size = 224
    elif name == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2"); m.fc = torch.nn.Identity()
        img_size = 224
    elif name == "vit_b_16":
        m = torchvision.models.vit_b_16(weights="IMAGENET1K_V1"); m.heads.head = torch.nn.Identity()
        img_size = 224
    else:
        raise ValueError(f"未知骨幹: {name}")

    pre = transforms.Compose([
        transforms.Resize(img_size + 32),  # 略放大再中心裁切，或直接用 Resize((img_size, img_size))
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    m.eval()
    return m, pre


# ---------- 主流程 ----------
# (A) 準備資料夾（若不足自動重組）
BASE_DIR, group_mode_desc = ensure_grouped_folder(SRC_DIR)

# (B) 讀取最終要用的檔案、貼標、群組
file_paths = list_images_recursively(BASE_DIR)
labels = []
for p in file_paths:
    lbl = label_or_none(p)
    if lbl is not None:
        labels.append(lbl)
y = np.array(labels)
groups = np.array([folder_group(p) for p in file_paths])

# (C) 載入骨幹並抽特徵
backbone, preprocess = load_backbone(BACKBONE_NAME)
X, latencies = [], []
for p in file_paths:
    img = Image.open(p).convert("RGB")
    t0 = time.time()
    with torch.no_grad():
        v = preprocess(img).unsqueeze(0)
        f = backbone(v).squeeze(0).cpu().numpy()
    latencies.append((time.time()-t0)*1000.0)
    X.append(f)
X = np.stack(X)
avg_ms = float(np.mean(latencies))

# (D) Group Split（若不幸切出單一類別就重試 seed）
def split_with_retry(X, y, groups, test_size=TEST_SIZE, seed=RANDOM_SEED, max_tries=500):
    for rs in range(seed, seed+max_tries):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr_idx, te_idx = next(gss.split(X, y, groups=groups))
        if len(np.unique(y[tr_idx])) >= 2 and len(np.unique(y[te_idx])) >= 2:
            return tr_idx, te_idx, rs
    raise RuntimeError("⚠️ 無法切出兩側都有兩類，請檢查資料或提高 K_GROUPS_PER_CLASS")

tr_idx, te_idx, used_seed = split_with_retry(X, y, groups)

X_tr, X_te, y_tr, y_te = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

# (E) 訓練 + 舊版輸出
clf = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)

print(f"[INFO] 總影像數: {len(y)}, 特徵維度: {X.shape[1]}, 平均單張延遲: {avg_ms:.1f} ms")
print(f"[INFO] 使用骨幹: {BACKBONE_NAME}, 群組模式: {group_mode_desc}, seed={used_seed}")

print("\n=== 嚴謹切分成績（AB=0, APBT=1）===")
print(classification_report(y_te, y_pred, digits=4))

print("混淆矩陣：")
print(confusion_matrix(y_te, y_pred))

try:
    y_prob = clf.predict_proba(X_te)[:,1]
    auc = roc_auc_score(y_te, y_prob)
    print(f"\nROC AUC: {auc:.4f}")
except:
    print("\nROC AUC: 無法計算")

print(f"\n平均骨幹推論延遲（ms/張）: {avg_ms:.1f}")

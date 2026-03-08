"""
SAM 3 Demo — Segment Anything Model 3
======================================
Demonstrates 3 prompt modes:
  1. Text prompt  — find objects by name (e.g., "egg")
  2. Point prompt — click a point to segment one object
  3. Automatic Mask Generation (AMG) — discover all objects

Usage:
  python demo_sam3.py --mode text --text "egg"
  python demo_sam3.py --mode point --point 200 150
  python demo_sam3.py --mode amg
  python demo_sam3.py --mode all
  python demo_sam3.py --image path/to/image.jpg --mode text --text "cat"
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pcs_model(device):
    """Load SAM 3 PCS model (text + detection)."""
    from transformers import Sam3Processor, Sam3Model
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    model.eval()
    print(f"[OK] SAM 3 PCS model loaded on {device}")
    return processor, model


def load_tracker_model(device):
    """Load SAM 3 Tracker model (point/box prompts + AMG)."""
    from transformers import Sam3TrackerProcessor, Sam3TrackerModel
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
    model.eval()
    print(f"[OK] SAM 3 Tracker model loaded on {device}")
    return processor, model


# ---------------------------------------------------------------------------
# 1. Text prompt segmentation
# ---------------------------------------------------------------------------

def segment_by_text(image, text_prompt, processor, model, device,
                    threshold=0.3, mask_threshold=0.5):
    """
    Find all instances of `text_prompt` in the image using SAM 3 PCS.

    Returns list of dicts with keys: mask, score, bbox
    """
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=[list(image.size[::-1])],  # (H, W)
    )[0]

    segments = []
    masks = results.get("masks", [])
    scores = results.get("scores", [])

    for mask, score in zip(masks, scores):
        mask_np = mask.cpu().numpy().astype(bool)
        score_val = score.item() if torch.is_tensor(score) else float(score)
        ys, xs = np.where(mask_np)
        if len(xs) == 0:
            continue
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        segments.append({"mask": mask_np, "score": score_val, "bbox": bbox})

    return segments


def demo_text_prompt(image, text_prompt, device):
    """Run and visualize text prompt segmentation."""
    processor, model = load_pcs_model(device)
    print(f'\nSearching for "{text_prompt}" ...')

    segments = segment_by_text(image, text_prompt, processor, model, device)
    print(f"Found {len(segments)} instance(s)")

    img_w, img_h = image.size
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(image)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(segments), 1)))
    for i, seg in enumerate(segments):
        overlay = np.zeros((img_h, img_w, 4))
        overlay[seg["mask"]] = [*colors[i % len(colors)][:3], 0.5]
        axes[1].imshow(overlay)

        x1, y1, x2, y2 = seg["bbox"]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=colors[i % len(colors)],
                                  facecolor="none")
        axes[1].add_patch(rect)
        axes[1].text(x1, y1 - 5, f'{text_prompt} ({seg["score"]:.2f})',
                     fontsize=10, color="white", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor=colors[i % len(colors)][:3], alpha=0.85))

    axes[1].set_title(f'Text Prompt: "{text_prompt}" — {len(segments)} found')
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("demo_sam3_text.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: demo_sam3_text.png")


# ---------------------------------------------------------------------------
# 2. Point prompt segmentation
# ---------------------------------------------------------------------------

def segment_by_point(image, point_xy, tracker_processor, tracker_model, device):
    """
    Segment the object at (x, y) using SAM 3 Tracker with a single point prompt.

    Returns the best mask (numpy bool array) and its IoU score.
    """
    input_points = [[[[point_xy[0], point_xy[1]]]]]
    input_labels = [[[1]]]  # 1 = foreground

    inputs = tracker_processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = tracker_model(**inputs)

    masks = outputs.pred_masks.squeeze().cpu().numpy()
    scores = outputs.iou_scores.squeeze().cpu().numpy()

    best_idx = scores.argmax()
    best_mask = masks[best_idx]
    best_score = float(scores[best_idx])

    img_w, img_h = image.size
    best_mask_resized = np.array(
        PILImage.fromarray(best_mask.astype(np.float32)).resize((img_w, img_h))
    ) > 0

    return best_mask_resized, best_score


def demo_point_prompt(image, point_xy, device):
    """Run and visualize point prompt segmentation."""
    tracker_processor, tracker_model = load_tracker_model(device)
    print(f"\nSegmenting object at point ({point_xy[0]}, {point_xy[1]}) ...")

    mask, score = segment_by_point(image, point_xy, tracker_processor, tracker_model, device)
    img_w, img_h = image.size

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(image)
    axes[0].scatter([point_xy[0]], [point_xy[1]], c="red", s=200, marker="*", zorder=5)
    axes[0].set_title(f"Input Point ({point_xy[0]}, {point_xy[1]})")
    axes[0].axis("off")

    axes[1].imshow(image)
    overlay = np.zeros((img_h, img_w, 4))
    overlay[mask] = [0.0, 0.8, 0.2, 0.5]
    axes[1].imshow(overlay)
    axes[1].scatter([point_xy[0]], [point_xy[1]], c="red", s=200, marker="*", zorder=5)
    axes[1].set_title(f"Segmented Object (IoU score: {score:.3f})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("demo_sam3_point.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: demo_sam3_point.png")


# ---------------------------------------------------------------------------
# 3. Automatic Mask Generation (AMG)
# ---------------------------------------------------------------------------

def compute_mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def automatic_mask_generation(image, tracker_processor, tracker_model, device,
                               points_per_side=16, pred_iou_thresh=0.80,
                               min_mask_region_area=500, nms_iou_thresh=0.5):
    """
    Automatic Mask Generation: sample a grid of points, collect masks, apply NMS.

    Returns list of dicts with keys: mask, score, bbox, area
    """
    img_w, img_h = image.size

    # Generate uniform grid
    xs = np.linspace(0, img_w - 1, points_per_side).astype(int)
    ys = np.linspace(0, img_h - 1, points_per_side).astype(int)
    grid_points = [[int(x), int(y)] for y in ys for x in xs]

    print(f"AMG: Sampling {len(grid_points)} grid points ({points_per_side}x{points_per_side}) ...")

    candidates = []
    for pt in grid_points:
        input_points = [[[[pt[0], pt[1]]]]]
        input_labels = [[[1]]]

        inputs = tracker_processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = tracker_model(**inputs)

        masks = outputs.pred_masks.squeeze().cpu().numpy()
        scores = outputs.iou_scores.squeeze().cpu().numpy()

        best_idx = scores.argmax()
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx])

        best_mask_resized = np.array(
            PILImage.fromarray(best_mask.astype(np.float32)).resize((img_w, img_h))
        ) > 0

        area = int(best_mask_resized.sum())

        if best_score >= pred_iou_thresh and area >= min_mask_region_area:
            ys_m, xs_m = np.where(best_mask_resized)
            bbox = [int(xs_m.min()), int(ys_m.min()), int(xs_m.max()), int(ys_m.max())]
            candidates.append({"mask": best_mask_resized, "score": best_score,
                                "bbox": bbox, "area": area})

    print(f"  Candidates after filtering: {len(candidates)}")

    # NMS
    candidates.sort(key=lambda x: x["score"], reverse=True)
    keep = []
    used = [False] * len(candidates)
    for i in range(len(candidates)):
        if used[i]:
            continue
        keep.append(candidates[i])
        used[i] = True
        for j in range(i + 1, len(candidates)):
            if not used[j] and compute_mask_iou(candidates[i]["mask"], candidates[j]["mask"]) > nms_iou_thresh:
                used[j] = True

    print(f"  Final masks after NMS: {len(keep)}")
    return keep


def demo_amg(image, device):
    """Run and visualize Automatic Mask Generation."""
    tracker_processor, tracker_model = load_tracker_model(device)
    print()

    results = automatic_mask_generation(
        image, tracker_processor, tracker_model, device,
        points_per_side=16, pred_iou_thresh=0.80,
        min_mask_region_area=500, nms_iou_thresh=0.5,
    )

    img_w, img_h = image.size
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(image)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(results), 1)))
    for i, obj in enumerate(results):
        overlay = np.zeros((img_h, img_w, 4))
        overlay[obj["mask"]] = [*colors[i % len(colors)][:3], 0.45]
        axes[1].imshow(overlay)

        x1, y1, x2, y2 = obj["bbox"]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=colors[i % len(colors)],
                                  facecolor="none")
        axes[1].add_patch(rect)
        axes[1].text(x1, y1 - 5, f'Obj {i+1} ({obj["score"]:.2f})',
                     fontsize=9, color="white", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor=colors[i % len(colors)][:3], alpha=0.8))

    axes[1].set_title(f"AMG: {len(results)} Objects Discovered")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("demo_sam3_amg.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: demo_sam3_amg.png")

    print(f"\n{'Obj':<5} {'Score':<8} {'Area (px)':<12} {'BBox'}")
    print("-" * 50)
    for i, r in enumerate(results):
        print(f"{i+1:<5} {r['score']:<8.4f} {r['area']:<12,} {r['bbox']}")


# ---------------------------------------------------------------------------
# Load a sample image (from FoodSeg103 or local file)
# ---------------------------------------------------------------------------

def load_sample_image(image_path=None, dataset_index=2629):
    """Load image from file path or FoodSeg103 dataset."""
    if image_path:
        image = PILImage.open(image_path).convert("RGB")
        print(f"Loaded image: {image_path} ({image.size[0]}x{image.size[1]})")
        return image

    from datasets import load_dataset
    ds = load_dataset("EduardoPacheco/FoodSeg103", split="train")
    sample = ds[dataset_index]
    image = sample["image"].convert("RGB")
    gt_classes = sample.get("classes_on_image", [])

    FOODSEG103_LABELS = {
        0: "background", 1: "candy", 2: "egg tart", 3: "french fries",
        4: "chocolate", 5: "biscuit", 24: "egg", 25: "apple", 29: "banana",
        37: "lemon", 46: "steak", 58: "bread", 59: "corn", 66: "rice",
        73: "asparagus", 74: "broccoli", 84: "lettuce", 91: "tomato",
    }
    gt_names = [FOODSEG103_LABELS.get(c, f"class_{c}") for c in gt_classes]
    print(f"Loaded FoodSeg103[{dataset_index}] ({image.size[0]}x{image.size[1]})")
    print(f"  Ground truth classes: {gt_names}")
    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM 3 Demo")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image (default: FoodSeg103 sample)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["text", "point", "amg", "all"],
                        help="Prompt mode: text, point, amg, or all")
    parser.add_argument("--text", type=str, default="egg",
                        help="Text prompt for text mode (default: egg)")
    parser.add_argument("--point", type=int, nargs=2, default=[200, 150],
                        metavar=("X", "Y"),
                        help="Point coordinates for point mode (default: 200 150)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    image = load_sample_image(args.image)

    if args.mode in ("text", "all"):
        demo_text_prompt(image, args.text, device)

    if args.mode in ("point", "all"):
        demo_point_prompt(image, args.point, device)

    if args.mode in ("amg", "all"):
        demo_amg(image, device)

    print("\nDone!")


if __name__ == "__main__":
    main()

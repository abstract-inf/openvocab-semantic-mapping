import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
import matplotlib.patches as patches

# --- CONFIGURATION ---
RESULTS_CSV = "model_comparison_results.csv"
DETAILED_CSV = "detailed_predictions.csv"
IMAGE_DIR = "./val2017"
OUTPUT_DIR = "./paper_plots"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Set style for "beautiful plots"
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = {"YOLOv10n": "#d62728", "RT-DETR": "#1f77b4", "YOLO11n": "#2ca02c", "YOLO-Worldv2": "#9467bd"}

def plot_bar_metrics(df):
    """Generates bar charts for key accuracy metrics."""
    metrics = ["mAP@.75", "CIoU", "F1-Score"]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x="Model", y=metric, palette=PALETTE, ax=axes[i], hue="Model", legend=False)
        axes[i].set_title(f"{metric} Comparison", fontweight='bold', fontsize=20)
        axes[i].set_ylim(0, 1.0)
        axes[i].set_xlabel("").set_fontsize(20)
        axes[i].set_ylabel(metric).set_fontsize(20)
        axes[i].grid(axis='y', linestyle="--", alpha=0.9)
        axes[i].tick_params(axis='x', labelsize=15)
        axes[i].tick_params(axis='y', labelsize=20)
        # Add values on top of bars
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt='%.2f', padding=3, fontsize=25)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/metric_comparison.png", dpi=300)
    print(f"Saved metric_comparison.png")
    plt.close()

def plot_efficiency(df):
    """Scatter plot for Latency vs Accuracy (Pareto-style)."""
    plt.figure(figsize=(11, 7))
    
    # Scatter plot
    sns.scatterplot(
        data=df, 
        x="Inference Latency (ms)", 
        y="mAP@.75", 
        hue="Model", 
        style="Model", 
        palette=PALETTE, 
        s=200, # Marker size
        alpha=0.9
    )
    plt.legend(fontsize=15, title="Model", title_fontsize=15, frameon=True, fancybox=True, shadow=True, prop={'weight': 'bold'})
    
    # Add text labels
    for i, row in df.iterrows():
        if row["Model"] == "YOLO-Worldv2":
            plt.text(
                row["Inference Latency (ms)"] + 1, 
                row["mAP@.75"] - 0.03, 
                f"{row['Model']}\n({row['FPS']} FPS)", 
                fontsize=20,
                weight='bold'
            )
        elif row["Model"] == "RT-DETR":
            plt.text(
                row["Inference Latency (ms)"] - 8, 
                row["mAP@.75"] + 0.01, 
                f"{row['Model']}\n({row['FPS']} FPS)", 
                fontsize=20,
                weight='bold'
            )
        else:
            plt.text(
                row["Inference Latency (ms)"] + 1, 
                row["mAP@.75"] + 0.01, 
                f"{row['Model']}\n({row['FPS']} FPS)", 
                fontsize=20,
                weight='bold'
            )

    plt.title("Efficiency vs. Accuracy Trade-off", fontweight='bold', fontsize=25)
    plt.xlabel("Inference Latency (ms)", fontsize=25)
    plt.ylabel("mAP@.75", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(True, linestyle="--", alpha=0.75)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/efficiency_tradeoff.png", dpi=300)
    print(f"Saved efficiency_tradeoff.png")
    plt.close()

def plot_visual_case_studies(detailed_df, image_dir):
    """
    Creates a 2x3 grid of images. 
    Each cell shows ONE image with bounding boxes from ALL models.
    """
    # 1. Find images where at least 3 models made a detection (for interesting overlap)
    img_counts = detailed_df.groupby("Filename")["Model"].nunique()
    interesting_files = img_counts[img_counts >= 3].index.tolist()
    
    if len(interesting_files) < 6:
        print("Not enough 'interesting' images (images where 3+ models detected something). Sampling random ones.")
        interesting_files = detailed_df["Filename"].unique().tolist()
    
    # Sample 6 distinct images
    selected_files = random.sample(interesting_files, min(6, len(interesting_files)))
    
    # Create Figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, filename in enumerate(selected_files):
        ax = axes[i]
        img_path = os.path.join(image_dir, filename)
        
        if not os.path.exists(img_path):
            ax.text(0.5, 0.5, "Image Not Found", ha='center')
            continue

        # Load Image (RGB)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get predictions for this image
        preds = detailed_df[detailed_df["Filename"] == filename]
        
        # Display Image
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Sample {i+1}", fontsize=12, fontweight='bold')

        # Draw Boxes
        legend_handles = []
        seen_models = set()
        
        for _, row in preds.iterrows():
            model = row["Model"]
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            ciou = row["CIoU"]
            
            color = PALETTE.get(model, "#333333")
            
            # Draw Rectangle
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, 
                linewidth=2, 
                edgecolor=color, 
                facecolor='none', 
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add tiny text label inside box
            ax.text(
                x1, y1 - 5, 
                f"{model[:4]} {ciou:.2f}", 
                color="white", 
                fontsize=7, 
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1)
            )

            # For Legend
            if model not in seen_models:
                legend_handles.append(patches.Patch(color=color, label=model))
                seen_models.add(model)
        
        if i == 0: # Add legend only to first plot to avoid clutter
            ax.legend(handles=legend_handles, loc='upper right', fontsize='small', framealpha=0.9)

    plt.suptitle(f"Qualitative Comparison: Bounding Box Alignments (CIoU)", fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/visual_case_studies.png", dpi=300)
    print(f"Saved visual_case_studies.png")
    plt.close()

def main():
    print("Loading data...")
    if not os.path.exists(RESULTS_CSV) or not os.path.exists(DETAILED_CSV):
        print(f"Error: CSV files not found. Please run compare_models.py first.")
        return

    summary_df = pd.read_csv(RESULTS_CSV)
    detailed_df = pd.read_csv(DETAILED_CSV)

    # 1. Metric Bars
    plot_bar_metrics(summary_df)

    # 2. Efficiency Scatter
    plot_efficiency(summary_df)

    # 3. Visual Grid
    plot_visual_case_studies(detailed_df, IMAGE_DIR)
    
    print("\nAll plots generated in './paper_plots/' folder.")

if __name__ == "__main__":
    main()
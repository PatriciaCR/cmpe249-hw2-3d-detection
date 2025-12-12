#!/usr/bin/env python3
"""
Open3D viewer for PLY files saved by mmdet3d_inference2.py.
Adds offline PNG screenshot saving with --save_png.
"""

import argparse
import os
import sys

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is not installed. Install with `pip install open3d`.\n")
    sys.exit(1)


def load_if_exists(path: str, loader, name: str):
    """Load a geometry with the given loader if the path exists."""
    if os.path.exists(path):
        try:
            obj = loader(path)
            print(f"[LOAD] {name}: {path}")
            return obj
        except Exception as e:
            print(f"[WARN] Failed to load {name} ({path}): {e}")
    else:
        print(f"[SKIP] {name} not found: {path}")
    return None


def save_png(geoms, width, height, out_path):
    """Render the scene offscreen and save a PNG."""
    print(f"[INFO] Saving PNG to: {out_path}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)

    for g in geoms:
        vis.add_geometry(g)

    vis.poll_events()
    vis.update_renderer()

    success = vis.capture_screen_image(out_path)
    vis.destroy_window()

    if success:
        print(f"[OK] Saved screenshot: {out_path}")
    else:
        print("[ERROR] Failed to save screenshot.")


def main():
    parser = argparse.ArgumentParser(description="Open3D viewer for saved PLY outputs")
    parser.add_argument("--dir", required=True,
                        help="Folder containing PLY files")
    parser.add_argument("--basename", required=True,
                        help="Base name of the files")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--save_png", action="store_true",
                        help="Save PNG instead of opening viewer")

    args = parser.parse_args()

    base_dir = os.path.expanduser(args.dir)
    base = args.basename

    # File paths
    points_path = os.path.join(base_dir, f"{base}_points.ply")
    axes_path = os.path.join(base_dir, f"{base}_axes.ply")
    pred_bbox_path = os.path.join(base_dir, f"{base}_pred_bboxes.ply")
    pred_label_path = os.path.join(base_dir, f"{base}_pred_labels.ply")
    gt_bbox_path = os.path.join(base_dir, f"{base}_gt_bboxes.ply")

    geoms = []

    # Load geometries
    pcd = load_if_exists(points_path, o3d.io.read_point_cloud, "Point cloud")
    if pcd is not None:
        geoms.append(pcd)

    axes = load_if_exists(axes_path, o3d.io.read_triangle_mesh, "Coordinate axes")
    if axes is not None:
        geoms.append(axes)

    pred_bboxes = load_if_exists(pred_bbox_path, o3d.io.read_line_set, "Predicted bboxes")
    if pred_bboxes is not None:
        geoms.append(pred_bboxes)

    pred_labels = load_if_exists(pred_label_path, o3d.io.read_line_set, "Predicted labels")
    if pred_labels is not None:
        geoms.append(pred_labels)

    gt_bboxes = load_if_exists(gt_bbox_path, o3d.io.read_line_set, "Ground truth bboxes")
    if gt_bboxes is not None:
        geoms.append(gt_bboxes)

    if not geoms:
        print("\nNo geometries loaded. Check --dir and --basename.")
        return

    # Offline PNG saving
    if args.save_png:
        out_path = os.path.join(base_dir, f"{base}.png")
        save_png(geoms, args.width, args.height, out_path)
        return

    # Interactive viewer
    print("\n[INFO] Opening viewer. Controls: mouse to rotate, scroll to zoom, 'Q' to exit.")
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"PLY Viewer: {base}",
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()

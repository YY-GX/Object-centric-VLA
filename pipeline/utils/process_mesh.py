#!/usr/bin/env python3
"""
Mesh Processing Tool for FoundationPose

Processes mesh files to ensure compatibility with FoundationPose:
- Removes material/texture references that can cause errors
- Optionally re-orients mesh to Z-up
- Optionally re-centers mesh origin to bottom center

Usage:
    python process_mesh.py [mesh_path] [options]

Examples:
    python process_mesh.py  # Process default plate mesh
    python process_mesh.py /path/to/mesh.obj --reorient --recenter
    python process_mesh.py /path/to/mesh.obj -o /path/to/output.obj
"""

import argparse
import numpy as np
import trimesh
from pathlib import Path


def process_mesh(
    input_path: str,
    output_path: str = None,
    reorient_z_up: bool = False,
    recenter: str = None,
    target_height: float = None,
    remove_materials: bool = True,
    verbose: bool = True
) -> str:
    """
    Process mesh for FoundationPose compatibility.

    Args:
        input_path: Path to input mesh file
        output_path: Path to output mesh file (default: overwrites input)
        reorient_z_up: Rotate mesh so Z is up (from Y-up)
        recenter: Origin placement - "center" (centroid), "bottom" (bottom center), or None
        target_height: Target height in meters to scale mesh to (based on Z after reorient)
        remove_materials: Remove material/texture references
        verbose: Print processing info

    Returns:
        Path to processed mesh file
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    if verbose:
        print(f"Loading mesh: {input_path}")

    # Load mesh
    mesh = trimesh.load(str(input_path), force='mesh')

    if verbose:
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        bounds = mesh.bounds
        print(f"  Bounds X: [{bounds[0][0]:.4f}, {bounds[1][0]:.4f}]")
        print(f"  Bounds Y: [{bounds[0][1]:.4f}, {bounds[1][1]:.4f}]")
        print(f"  Bounds Z: [{bounds[0][2]:.4f}, {bounds[1][2]:.4f}]")

    # Re-orient to Z-up (rotate -90 degrees around X axis)
    if reorient_z_up:
        if verbose:
            print("  Reorienting: Y-up -> Z-up (rotating -90° around X)")
        # Rotation matrix: -90 degrees around X axis
        # This transforms Y -> Z and Z -> -Y
        rotation = trimesh.transformations.rotation_matrix(
            -np.pi / 2, [1, 0, 0], point=[0, 0, 0]
        )
        mesh.apply_transform(rotation)

        if verbose:
            bounds = mesh.bounds
            print(f"  New bounds X: [{bounds[0][0]:.4f}, {bounds[1][0]:.4f}]")
            print(f"  New bounds Y: [{bounds[0][1]:.4f}, {bounds[1][1]:.4f}]")
            print(f"  New bounds Z: [{bounds[0][2]:.4f}, {bounds[1][2]:.4f}]")

    # Scale mesh to target height (based on Z dimension)
    if target_height is not None:
        bounds = mesh.bounds
        current_height = bounds[1][2] - bounds[0][2]  # Z dimension
        if current_height > 0:
            scale_factor = target_height / current_height
            if verbose:
                print(f"  Scaling: {current_height:.4f}m -> {target_height:.4f}m (factor: {scale_factor:.4f})")
            mesh.apply_scale(scale_factor)

            if verbose:
                bounds = mesh.bounds
                sizes = bounds[1] - bounds[0]
                print(f"  New size X: {sizes[0]:.4f}m ({sizes[0]*100:.1f}cm)")
                print(f"  New size Y: {sizes[1]:.4f}m ({sizes[1]*100:.1f}cm)")
                print(f"  New size Z: {sizes[2]:.4f}m ({sizes[2]*100:.1f}cm)")

    # Re-center origin
    if recenter == "center":
        if verbose:
            print("  Recentering: moving origin to centroid (center of object)")
        centroid = mesh.centroid
        translation = [-centroid[0], -centroid[1], -centroid[2]]
        mesh.apply_translation(translation)

        if verbose:
            bounds = mesh.bounds
            print(f"  New bounds X: [{bounds[0][0]:.4f}, {bounds[1][0]:.4f}]")
            print(f"  New bounds Y: [{bounds[0][1]:.4f}, {bounds[1][1]:.4f}]")
            print(f"  New bounds Z: [{bounds[0][2]:.4f}, {bounds[1][2]:.4f}]")

    elif recenter == "bottom":
        if verbose:
            print("  Recentering: moving origin to bottom center")
        bounds = mesh.bounds
        # Center in X and Y, bottom in Z
        center_x = (bounds[0][0] + bounds[1][0]) / 2
        center_y = (bounds[0][1] + bounds[1][1]) / 2
        bottom_z = bounds[0][2]

        translation = [-center_x, -center_y, -bottom_z]
        mesh.apply_translation(translation)

        if verbose:
            bounds = mesh.bounds
            print(f"  New bounds X: [{bounds[0][0]:.4f}, {bounds[1][0]:.4f}]")
            print(f"  New bounds Y: [{bounds[0][1]:.4f}, {bounds[1][1]:.4f}]")
            print(f"  New bounds Z: [{bounds[0][2]:.4f}, {bounds[1][2]:.4f}]")

    # Remove materials/textures to avoid FoundationPose errors
    if remove_materials:
        if verbose:
            print("  Removing materials/textures for FoundationPose compatibility")
        # Create a simple visual without materials
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export as OBJ without materials
    if verbose:
        print(f"Saving processed mesh: {output_path}")

    # Export with minimal OBJ (no MTL reference)
    obj_content = trimesh.exchange.obj.export_obj(mesh, include_normals=True, include_texture=False)

    # Remove any mtllib lines that might have been added
    lines = obj_content.split('\n')
    clean_lines = [line for line in lines if not line.startswith('mtllib') and not line.startswith('usemtl')]
    clean_content = '\n'.join(clean_lines)

    with open(output_path, 'w') as f:
        f.write(clean_content)

    if verbose:
        print("Done!")
        print(f"\nMesh info after processing:")
        # Reload to verify
        mesh_check = trimesh.load(str(output_path), force='mesh')
        bounds = mesh_check.bounds
        print(f"  Vertices: {len(mesh_check.vertices)}")
        print(f"  Faces: {len(mesh_check.faces)}")
        print(f"  Bounds X: [{bounds[0][0]:.4f}, {bounds[1][0]:.4f}] (width: {bounds[1][0]-bounds[0][0]:.4f})")
        print(f"  Bounds Y: [{bounds[0][1]:.4f}, {bounds[1][1]:.4f}] (depth: {bounds[1][1]-bounds[0][1]:.4f})")
        print(f"  Bounds Z: [{bounds[0][2]:.4f}, {bounds[1][2]:.4f}] (height: {bounds[1][2]-bounds[0][2]:.4f})")
        if recenter == "center":
            print(f"  Origin at: (0, 0, 0) -> center of mesh")
        elif recenter == "bottom":
            print(f"  Origin at: (0, 0, 0) -> bottom center of mesh")

    return str(output_path)


def check_mesh(input_path: str):
    """
    Print key information about a mesh file.

    Args:
        input_path: Path to mesh file
    """
    print(f"=== Mesh Info: {input_path} ===\n")

    mesh = trimesh.load(str(input_path), force='mesh')
    bounds = mesh.bounds
    sizes = bounds[1] - bounds[0]
    centroid = mesh.centroid

    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")

    print(f"\nBounds:")
    print(f"  X: [{bounds[0][0]:.4f}, {bounds[1][0]:.4f}] (size: {sizes[0]:.4f}m = {sizes[0]*100:.1f}cm)")
    print(f"  Y: [{bounds[0][1]:.4f}, {bounds[1][1]:.4f}] (size: {sizes[1]:.4f}m = {sizes[1]*100:.1f}cm)")
    print(f"  Z: [{bounds[0][2]:.4f}, {bounds[1][2]:.4f}] (size: {sizes[2]:.4f}m = {sizes[2]*100:.1f}cm)")

    print(f"\nCentroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")

    # Determine orientation
    axis_names = ['X', 'Y', 'Z']
    min_axis = axis_names[np.argmin(sizes)]
    max_axis = axis_names[np.argmax(sizes)]

    print(f"\nOrientation analysis:")
    print(f"  Thinnest axis: {min_axis} ({np.min(sizes)*100:.1f}cm) - likely object height")
    print(f"  Largest axis:  {max_axis} ({np.max(sizes)*100:.1f}cm) - likely object width/diameter")

    if min_axis == 'Z':
        print(f"  Status: Z-up orientation (good)")
    else:
        print(f"  Status: {min_axis}-up orientation (may need --reorient)")

    # Check origin placement
    print(f"\nOrigin analysis:")
    if abs(bounds[0][2]) < 0.001:
        print(f"  Z min = 0: Origin at bottom (good)")
    else:
        print(f"  Z min = {bounds[0][2]:.4f}: Origin NOT at bottom (may need --recenter)")

    if abs(centroid[0]) < 0.01 and abs(centroid[1]) < 0.01:
        print(f"  XY centered (good)")
    else:
        print(f"  XY offset: [{centroid[0]:.4f}, {centroid[1]:.4f}] (may need --recenter)")

    # Material/texture check
    print(f"\nMaterial/texture:")
    has_material = hasattr(mesh.visual, 'material') and mesh.visual.material is not None
    if has_material:
        has_texture = hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None
        if has_texture:
            print(f"  Has texture image (may cause FoundationPose errors, run without --check to fix)")
        else:
            print(f"  Has material but no texture")
    else:
        print(f"  No materials (good)")

    print(f"\n=== End ===")


def main():
    parser = argparse.ArgumentParser(
        description='Process mesh for FoundationPose compatibility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check mesh info (no modifications)
  python process_mesh.py --check
  python process_mesh.py --check --input_path /path/to/mesh.obj

  # Standard processing: reorient + recenter + scale to real height (recommended)
  python process_mesh.py --input_path /path/to/mesh.obj --reorient --recenter center --height 0.10

  # Scale to 10cm height without reorienting
  python process_mesh.py --input_path /path/to/mesh.obj --recenter center --height 0.10

  # Reorient + recenter to bottom center + scale
  python process_mesh.py --input_path /path/to/mesh.obj --reorient --recenter bottom --height 0.10

  # Save to different output file
  python process_mesh.py --input_path /path/to/mesh.obj -o /path/to/processed.obj --reorient --recenter center --height 0.10
        """
    )

    parser.add_argument(
        '--input_path',
        nargs='?',
        default='/home/yygx/PANDA/pipeline/data/meshes/plate/mesh.obj',
        help='Path to input mesh file (default: plate mesh)'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output path (default: overwrite input)'
    )
    parser.add_argument(
        '--reorient',
        action='store_true',
        help='Reorient mesh from Y-up to Z-up'
    )
    parser.add_argument(
        '--recenter',
        choices=['center', 'bottom'],
        default=None,
        help='Recenter origin: "center" (centroid) or "bottom" (bottom center)'
    )
    parser.add_argument(
        '--height',
        type=float,
        default=None,
        help='Target height in METERS to scale mesh to (e.g., 0.10 for 10cm)'
    )
    parser.add_argument(
        '--keep-materials',
        action='store_true',
        help='Keep materials/textures (not recommended for FoundationPose)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress output'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check mesh info only (no modifications)'
    )

    args = parser.parse_args()

    if args.check:
        check_mesh(args.input_path)
        return

    process_mesh(
        input_path=args.input_path,
        output_path=args.output,
        reorient_z_up=args.reorient,
        recenter=args.recenter,
        target_height=args.height,
        remove_materials=not args.keep_materials,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Fix Mesh Files for FoundationPose

Repairs common mesh issues:
- Inverted/inconsistent normals
- Non-watertight meshes (holes)

Usage:
    python fix_mesh.py --mesh /path/to/mesh.obj
    python fix_mesh.py --mesh /path/to/mesh.obj --output /path/to/fixed.obj
    python fix_mesh.py --all  # Fix all known problematic meshes
"""

import argparse
import numpy as np
import trimesh
from pathlib import Path


def analyze_mesh(mesh, name="mesh"):
    """Analyze mesh quality and print report."""
    print(f"\n{'='*50}")
    print(f"Analyzing: {name}")
    print(f"{'='*50}")

    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Is watertight: {mesh.is_watertight}")
    print(f"Is convex: {mesh.is_convex}")

    # Check dimensions
    vertices = np.array(mesh.vertices)
    dims = vertices.max(axis=0) - vertices.min(axis=0)
    print(f"Dimensions (m): {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f}")

    # Check normals orientation
    face_normals = mesh.face_normals
    centroids = mesh.triangles_center
    center = mesh.centroid

    directions = centroids - center
    dots = np.sum(directions * face_normals, axis=1)
    outward_pct = (dots > 0).sum() / len(dots) * 100
    print(f"Outward-facing normals: {outward_pct:.1f}%")

    # Check for NaN
    if np.isnan(mesh.vertices).any():
        print("WARNING: Mesh has NaN vertices!")
    if np.isnan(mesh.vertex_normals).any():
        print("WARNING: Mesh has NaN normals!")

    return {
        'watertight': mesh.is_watertight,
        'outward_pct': outward_pct,
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces)
    }


def fix_mesh(mesh_path, output_path=None, backup=True):
    """
    Fix mesh issues and save.

    Args:
        mesh_path: Path to input mesh
        output_path: Path to save fixed mesh (default: overwrite original)
        backup: Whether to create backup of original

    Returns:
        True if fixes were applied, False if mesh was already OK
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        print(f"Error: Mesh not found: {mesh_path}")
        return False

    # Load mesh
    print(f"\nLoading: {mesh_path}")
    mesh = trimesh.load(str(mesh_path))

    # Analyze before
    print("\n--- BEFORE FIXES ---")
    before = analyze_mesh(mesh, mesh_path.name)

    needs_fix = False

    # Check if fixes needed
    if before['outward_pct'] < 90:
        print(f"\n⚠️  Normals need fixing ({before['outward_pct']:.1f}% outward)")
        needs_fix = True

    if not before['watertight']:
        print(f"\n⚠️  Mesh is not watertight")
        needs_fix = True

    if not needs_fix:
        print(f"\n✓ Mesh looks OK, no fixes needed")
        return False

    # Apply fixes
    print(f"\n--- APPLYING FIXES ---")

    # Fix normals (make consistent and outward-facing)
    print("Fixing normals...")
    trimesh.repair.fix_normals(mesh)

    # Fix winding order
    print("Fixing face winding...")
    trimesh.repair.fix_winding(mesh)

    # Fill holes (try to make watertight)
    print("Filling holes...")
    trimesh.repair.fill_holes(mesh)

    # Analyze after
    print("\n--- AFTER FIXES ---")
    after = analyze_mesh(mesh, f"{mesh_path.name} (fixed)")

    # Determine output path
    if output_path is None:
        output_path = mesh_path
    else:
        output_path = Path(output_path)

    # Create backup if overwriting
    if backup and output_path == mesh_path:
        backup_path = mesh_path.with_suffix('.obj.backup')
        if not backup_path.exists():
            import shutil
            shutil.copy(mesh_path, backup_path)
            print(f"\n✓ Created backup: {backup_path}")

    # Save fixed mesh
    print(f"\nSaving fixed mesh to: {output_path}")
    mesh.export(str(output_path))

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Outward normals: {before['outward_pct']:.1f}% -> {after['outward_pct']:.1f}%")
    print(f"Watertight: {before['watertight']} -> {after['watertight']}")
    print(f"✓ Mesh fixed and saved!")

    return True


def fix_all_problematic():
    """Fix all known problematic meshes."""
    pipeline_root = Path(__file__).parent.parent
    meshes_dir = pipeline_root / "data" / "meshes"

    # Known problematic meshes
    problematic = ['wooden_tray', 'rubik_cube']

    print(f"\n{'='*60}")
    print("FIXING ALL PROBLEMATIC MESHES")
    print(f"{'='*60}")

    fixed_count = 0
    for obj_name in problematic:
        mesh_path = meshes_dir / obj_name / "mesh.obj"
        if mesh_path.exists():
            if fix_mesh(mesh_path):
                fixed_count += 1
        else:
            print(f"\n⚠️  Mesh not found: {mesh_path}")

    print(f"\n{'='*60}")
    print(f"Done! Fixed {fixed_count}/{len(problematic)} meshes")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Fix mesh files for FoundationPose")
    parser.add_argument('--mesh', type=str, help='Path to mesh file to fix')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: overwrite original)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup when overwriting')
    parser.add_argument('--all', action='store_true',
                        help='Fix all known problematic meshes (wooden_tray, rubik_cube)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze mesh, do not fix')

    args = parser.parse_args()

    if args.all:
        fix_all_problematic()
    elif args.mesh:
        if args.analyze_only:
            mesh = trimesh.load(args.mesh)
            analyze_mesh(mesh, args.mesh)
        else:
            fix_mesh(args.mesh, args.output, backup=not args.no_backup)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python fix_mesh.py --all")
        print("  python fix_mesh.py --mesh /path/to/mesh.obj")
        print("  python fix_mesh.py --mesh /path/to/mesh.obj --analyze-only")


if __name__ == '__main__':
    main()

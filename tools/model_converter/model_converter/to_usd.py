import argparse
import bpy


def convert(input_file, output_usd):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Autodetect file extension and try to import
    ext = input_file.split('.')[-1].lower()
    if ext == "sdf":
        raise NotImplementedError("SDF import is not supported by Blender's bpy.ops. Please use OBJ, FBX, or DAE formats.")
    elif ext == "obj":
        bpy.ops.wm.obj_import(filepath=input_file)
    elif ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=input_file)
    elif ext == "dae":
        bpy.ops.wm.collada_import(filepath=input_file)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Manually correct the scene orientation for USD's Y-Up standard.
    if bpy.data.objects:
        bpy.ops.object.select_all(action='SELECT')
        # Rotate 90 degrees around the X-axis to convert Z-up to Y-up
        bpy.ops.transform.rotate(value=1.570796, orient_axis='X')
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        bpy.ops.object.select_all(action='DESELECT')

    bpy.ops.wm.usd_export(filepath=output_usd)


def main():
    parser = argparse.ArgumentParser(description="Convert SDF to USD")
    parser.add_argument("input_file", help="Path to the input SDF file")
    parser.add_argument("output_usd", help="Path to the output USD file")
    args = parser.parse_args()

    convert(args.input_file, args.output_usd)


if __name__ == "__main__":
    main()

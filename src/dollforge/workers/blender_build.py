"""Executed only by Blender's isolated Python. Input is JSON, never executable metadata."""
import json
import sys

import bpy


def main():
    source, destination = sys.argv[sys.argv.index("--") + 1:]
    with open(source, encoding="utf-8") as stream:
        request = json.load(stream)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    root = bpy.data.collections.new("DOLL_ROOT")
    bpy.context.scene.collection.children.link(root)
    collections = {}
    for name in ("00_REFERENCES", "10_BODY", "20_CLOTHING", "30_CONNECTORS", "40_GUIDES",
                 "50_BOOLEAN_CUTTERS", "60_PRINT_EXPORTS", "90_DEBUG"):
        collection = bpy.data.collections.new(name)
        root.children.link(collection)
        collections[name] = collection
    colors = [(0.25, .65, .58, 1), (.81, .58, .32, 1), (.44, .51, .73, 1)]
    for index, item in enumerate(request["meshes"]):
        mesh = bpy.data.meshes.new(item["name"] + "__source")
        mesh.from_pydata(item["vertices"], [], item["faces"])
        mesh.update()
        obj = bpy.data.objects.new(item["name"], mesh)
        group = "20_CLOTHING" if item["name"].startswith("footwear") else "10_BODY"
        collections[group].objects.link(obj)
        obj["part_instance_id"] = item["part_instance_id"]
        obj["provenance"] = json.dumps(item["provenance"])
        obj["confidence"] = item["confidence"]
        obj["canonical_transform"] = item["transform"]
        obj["vertices_space"] = "canonical_world_baked"
        material = bpy.data.materials.new(item["name"] + "__material")
        material.diffuse_color = colors[index % len(colors)]
        obj.data.materials.append(material)
        for face in mesh.polygons:
            face.use_smooth = True
    scene = bpy.context.scene
    scene["scale_mode"] = "absolute" if request["unit"] == "mm" else "relative"
    scene["manufacturable"] = False
    scene["canonical_coordinates"] = "RH_Z_UP; forward=-Y; origin=pelvis"
    if request["unit"] == "mm":
        scene.unit_settings.system = "METRIC"
        scene.unit_settings.scale_length = .001
        scene.unit_settings.length_unit = "MILLIMETERS"
    else:
        scene.unit_settings.system = "NONE"
    bpy.ops.wm.save_as_mainfile(filepath=destination)


if __name__ == "__main__":
    main()

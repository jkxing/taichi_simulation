import bpy 
import os 

# folder path to save rendered image
in_dir = "E:\\"
lst = os.listdir(in_dir)

# folder path for importing data files
in_dir_ply = "F:\\code\\taichi_simulation\\output"
lst_ply = os.listdir(in_dir_ply)

# folder path to save rendered animation
out_dir = "F:\\code\\taichi_simulation"

# Filter file list by valid file types.
candidates = []
candidates_name = []
c = 0
for item in lst_ply:
    fileName, fileExtension = os.path.splitext(lst_ply[c])
    if fileExtension == ".obj":
        candidates.append(item)
        candidates_name.append(fileName)
    c = c + 1

file = [{"name":i} for i in candidates]   
st = 0
ed = 360
file = file[st:ed]
candidates=candidates[st:ed]
candidates_name=candidates_name[st:ed]
n = len(file)

mat_name = "mat1"
bpy.context.scene.render.engine = 'CYCLES'
newmat = bpy.data.materials.new(mat_name)
newmat.use_nodes = True
node_tree = newmat.node_tree
nodes = node_tree.nodes
bsdf = nodes.get("Principled BSDF") 
assert(bsdf) 
bsdf.inputs[0].default_value=(0.0,0.5,0.5,0.5)
bsdf.inputs[5].default_value=(1.0)
bsdf.inputs[7].default_value=(0.0)
bsdf.inputs[15].default_value=(1.0)
bsdf.inputs[14].default_value=(1.333)


from math import radians
bpy.data.objects["Cube"].hide_viewport = True
bpy.data.objects["Cube"].hide_render = True


#bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value=(0.6,0.8,0.2, 1.0)

tree_nodes = bpy.data.worlds["World"].node_tree.nodes
node_background = bpy.data.worlds["World"].node_tree.nodes["Background"]
node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
node_environment.image = bpy.data.images.load("C:\\Program Files\\Blender Foundation\\Blender 2.83\\2.83\\datafiles\\studiolights\\world\\sunset.exr") # Relative path
links = bpy.data.worlds["World"].node_tree.links
link = links.new(node_environment.outputs[0], node_background.inputs[0])


bpy.data.cameras["Camera"].lens = 14
# To import mesh.ply in batches
for i in range (0,n):
    print(i)
    bpy.ops.import_scene.obj(filepath=os.path.join(in_dir_ply,candidates[i]), filter_glob="*.obj")
    bpy.data.objects[candidates_name[i]].hide_viewport = True
    bpy.data.objects[candidates_name[i]].hide_render = True
    bpy.data.objects[candidates_name[i]].select_set(True)
    bpy.data.objects[candidates_name[i]].rotation_euler = (0, 0, radians(90))
    bpy.data.objects[candidates_name[i]].scale =  (12,12,12)
    bpy.data.objects[candidates_name[i]].location =  (0,4,-6)
    bpy.data.objects[candidates_name[i]].select_set(False)

# Set file_format for render images
bpy.data.scenes["Scene"].render.image_settings.file_format = 'PNG'

#bpy.data.materials[mat_name].node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value=(0,0,255,0.5)
#bpy.data.materials[mat_name].node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value=(0,0,255,0.5)


# To render and save rendered images
for i in range (0,n):
    bpy.data.objects[candidates_name[i]].hide_viewport = False    #objects must be visible to use modifier
    bpy.data.objects[candidates_name[i]].hide_render = False    #objects must be renderable to export render image
    bpy.context.view_layer.objects.active = bpy.data.objects[candidates_name[i]] #get object
    #bpy.ops.object.modifier_add(type='PARTICLE_SYSTEM')    #add modifier as particle_system
    #bpy.data.objects[candidates_name[i]].particle_systems['ParticleSystem'].settings = bpy.data.particles['ParticleSettings'] #assign particle settings to object's particle system
    bpy.data.objects[candidates_name[i]].data.materials[0]=newmat    #assign existed material to active object.
    bpy.data.scenes["Scene"].render.filepath = "E:/%03d"%(i)    #set save filepath
    bpy.ops.render.render( write_still=True )    #render and save
    bpy.data.objects[candidates_name[i]].hide_viewport = True    #hide again for next image rendering
    bpy.data.objects[candidates_name[i]].hide_render = True    #hide again for next image rendering

# Active VSE to generate rendering animation
bpy.data.scenes["Scene"].render.use_sequencer = True

# Filter file list by valid file types.
re_image = []
re_image_name = []
c = 0
for item in lst:
    fileName, fileExtension = os.path.splitext(lst[c])
    if fileExtension == ".png":
        re_image.append(item)
        re_image_name.append(fileName)
    c = c + 1

# Create the sequencer data
bpy.context.scene.sequence_editor_create()

# Add strip into VSE by importing new image
for i in range (0,n):
    bpy.context.scene.sequence_editor.sequences.new_image(
        name="%03d.png"%(i),
        filepath=os.path.join(in_dir, "%03d.png"%(i)),
        channel=1, frame_start=i)

# Resolution settings for animation
resx = 1920 
resy = 1080 
bpy.data.scenes["Scene"].render.resolution_x = resx 
bpy.data.scenes["Scene"].render.resolution_y = resy 
bpy.data.scenes["Scene"].render.resolution_percentage = 100 

bpy.data.scenes["Scene"].frame_end = n 
bpy.data.scenes["Scene"].render.image_settings.file_format = 'AVI_JPEG' 
bpy.data.scenes["Scene"].render.filepath = out_dir
bpy.ops.render.render( animation=True )
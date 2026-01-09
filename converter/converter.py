#!/usr/bin/env python3
"""
IMD to NSBMD Converter
======================

This script converts Intermediate Model Data (IMD) XML files, typically generated
by Pokemon DS Map Studio, into Nintendo System Binary Model Data (NSBMD) files.
It mimics the interface of the original `g3dcvtr.exe` tool to allow seamless
integration with existing workflows.

Format Details:
---------------
- Input: IMD (XML based format for NDS models).
- Output: NSBMD (Binary format specific to Nintendo DS games).

Technical Context:
------------------
The NDS 3D engine (G3D) uses a Display List architecture. Commands are sent to a
FIFO to draw primitives. The NSBMD format essentially wraps these display list
commands along with model metadata (dictionaries, textures, palettes).

This script parses the high-level XML description of primitives and packs them
into the raw binary display list commands required by the hardware (e.g.,
unpacking 'pos_xyz' nodes into VTX_16 commands).
"""

import sys
import struct
import xml.etree.ElementTree as ET
import argparse
import os

class BinaryWriter:
    """
    Utility for assembling binary data with Little Endian packing.
    Extends bytearray with struct-based writing and random-access seek/overwrite capability,
    essential for filling in size headers and relative offsets after data is written.
    """
    def __init__(self):
        self.data = bytearray()
        self.pos = 0

    def write(self, fmt, *args):
        """
        Packs arguments according to format string and writes to current position.
        Uses standard struct.pack (e.g., '<I' for unsigned 32-bit little-endian).
        """
        data = struct.pack(fmt, *args)
        self.write_bytes(data)

    def write_bytes(self, b):
        """
        Writes raw bytes at the current position. 
        If at end of buffer, extends. If in middle, overwrites.
        Automatically advances the internal position counter.
        """
        if self.pos == len(self.data):
             self.data.extend(b)
        else:
             # Overwrite mode: handles mid-stream updates for offsets/sizes
             end_pos = self.pos + len(b)
             if end_pos > len(self.data):
                 # Extend if writing past current end
                 self.data[self.pos:] = b[:len(self.data)-self.pos]
                 self.data.extend(b[len(self.data)-self.pos:])
             else:
                 self.data[self.pos:end_pos] = b
        self.pos += len(b)

    def tell(self):
        """Returns the current write position."""
        return self.pos
        
    def seek(self, pos, whence=0):
        """
        Adjusts the internal write position.
        whence: 0=absolute, 1=relative to current, 2=relative to end.
        """
        if whence == 0:
            self.pos = pos
        elif whence == 1:
            self.pos += pos
        elif whence == 2:
            self.pos = len(self.data) + pos


class NSBMDConverter:
    """
    Orchestrates the conversion of IMD (XML) to NSBMD (Binary).
    
    This class implements the heavy lifting of NDS G3D format generation,
    including Display List command packing, Bounding Box calculation,
    and the hierarchical MDL0/TEX0 block assembly.
    """
    def __init__(self):
        self.writer = BinaryWriter()
        
        # --- NDS G3D Hardware Commands (FIFO Opcodes) ---
        # These constants represent the raw opcodes processed by the
        # Nintendo DS 3D Geometry Engine. 
        # Source: http://problemkaputt.de/gbatek.htm#ds3dvideo
        
        self.CMD_MTX_RESTORE = 0x14  # [1 word] Pops a matrix from the stack (Index 0-30)
        self.CMD_COLOR       = 0x20  # [1 word] Sets RGB 555 color for downstream vertices
        self.CMD_NORMAL      = 0x21  # [1 word] Sets the normal vector for lighting (10-bit components)
        self.CMD_TEXCOORD    = 0x22  # [1 word] Sets S,T texture coordinates (16-bit FX12)
        self.CMD_VTX_16      = 0x23  # [2 words] Sets X,Y,Z coords (16-bit FX12)
        self.CMD_VTX_10      = 0x24  # [1 word] Sets X,Y,Z coords (10-bit FX6)
        self.CMD_VTX_XY      = 0x25  # [1 word] Updates X,Y coords (16-bit word)
        self.CMD_VTX_XZ      = 0x26  # [1 word] Updates X,Z coords (16-bit word)
        self.CMD_VTX_YZ      = 0x27  # [1 word] Updates Y,Z coords (16-bit word)
        self.CMD_VTX_DIFF    = 0x28  # [1 word] Relative X,Y,Z update (10-bit components)
        self.CMD_POLYGON_ATTR= 0x29  # [1 word] Configures Cull, Alpha, Light, ID, Fog
        self.CMD_BEGIN_VTXS  = 0x40  # [1 word] Defines primitive type (Tri, Quad, Strip)
        self.CMD_END_VTXS    = 0x41  # [0 word] Terminates a primitive stream

    def build_nitro_dictionary(self, w, count):
        """
        Builds a Nitro resource dictionary header.
        
        The Pokemon-DS-Rom-Editor loader (NSBMD.cs line 434) skips:
          stream.Skip(10 + 4 + (num * 4))
        
        This means it expects:
          - 10 bytes: Dictionary tree structure  
          - 4 bytes: Padding/unknown
          - (num * 4) bytes: Offset table (handled separately)
        
        Total: 14 bytes before offset table
        
        We write exactly 14 bytes to match this skip pattern.
        """
        # Write 16 bytes of dictionary header (matching loader's skip of 1+1+14)
        # Byte 0: Dummy
        w.write("<B", 0)
        # Byte 1: Count
        w.write("<B", count)
        # Bytes 2-15: Padding (14 bytes to make total 16)
        w.write_bytes(b'\x00' * 14)
        
        return w.tell()
    def parse_imd(self, input_path):
        """Parses the IMD XML file and returns the root element."""
        print(f"Parsing IMD: {input_path}")
        try:
            tree = ET.parse(input_path)
            root = tree.getroot()
            return root
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            sys.exit(1)
        except FileNotFoundError:
             print(f"Error: File not found {input_path}")
             sys.exit(1)

    def build_mdl0_block(self, xml_root):
        """
        Constructs the MDL0 (Model 0) section of the NSBMD file.
        This block contains all geometric and material data for the model.
        
        The NDS format requires a nested hierarchy of dictionaries and blocks
        to allow the hardware/engine to quickly index into specific sections.
        
        Hierarchy:
        1. MDL0 Header (Magic, Size)
        2. Model Dictionary: Maps name (e.g., "model0") to the actual Model Header.
        3. Model Header: Contains global counts (vertices, tris), Bounding Box, and
           internal offsets to Code, Material, and Polygon sub-blocks.
        """
        w = BinaryWriter()
        w.write("<4s", b"MDL0")
        size_offset = w.tell()
        w.write("<I", 0) # Placeholder [Size]: Filled at the end of this method.
        
        # We assume 1 Model per file for now (standard for Map Studio IMD exports).
        num_models = 1
        
        # --- MDL0 Block Header (Dictionary for Models) ---
        # Nitro blocks require a Resource Dictionary even if only one model exists.
        # This matches the skip/read logic in LibNDSFormats (NSBMD.cs)
        
        self.build_nitro_dictionary(w, num_models)
        
        # Loader skips 14 + num*4 after count (NSBMD.cs:434)
        # Write first offset table (skipped)
        w.write("<I", 0)
        
        # Write second offset table (read at line 439)
        offset_model_offsets = w.tell()
        w.write("<I", 0)           # Offset to Model 0 Data (relative to MDL0 start)
        
        # Model Name (16 bytes)
        model_name = "model0".encode('ascii')
        w.write_bytes(model_name.ljust(16, b'\x00'))
        
        # --- Model Data Starts Here ---
        model_data_start = w.tell()
        w.seek(offset_model_offsets)
        w.write("<I", model_data_start)
        w.seek(model_data_start)
        
        model_start = w.tell() # Beginning of the actual model data structure (totalsize, etc)
        w.write("<I", 0) # [0x00] Total Size: Size of this entire Model block
        w.write("<I", 0) # [0x04] Code Offset (backfilled later)
        w.write("<I", 0) # [0x08] Material Offset (backfilled later)
        w.write("<I", 0) # [0x0C] Polygon Offset (backfilled later)
        w.write("<I", 0) # [0x10] Polygon End (backfilled later)
        w.write("<I", 0) # [0x14] Padding
        
        # Counts: Essential for GPU memory allocation on the NDS hardware.
        counts = self.get_model_counts(xml_root)
        w.write("<B", counts['mat'])  # [0x18] Number of materials
        w.write("<B", counts['poly']) # [0x19] Number of polygon groups
        w.write("<B", 0)              # [0x1A] LastStackId: Used for matrix stack state persistence
        w.write("<B", 1)              # [0x1B] ScaleMode: 0=Standard, 1=Maya, 2=Max (影响变换计算)
        
        # Scale & BBox: Stored as FX32 (Fixed point 1.19.12)
        # Bounding boxes are crucial for frustum culling on the DS.
        bbox = self.get_bbox(xml_root)
        w.write("<I", int(bbox['scale'] * 4096))
        w.write("<I", int(bbox['inv_scale'] * 4096))
        w.write("<h", bbox['xmin'])
        w.write("<h", bbox['ymin'])
        w.write("<h", bbox['zmin'])
        w.write("<h", bbox['xmax'])
        w.write("<h", bbox['ymax'])
        w.write("<h", bbox['zmax'])
        
        w.write("<H", counts['vertex']) # Total Vertex Count
        w.write("<H", 0)               # Surface Count (Used for specific collision/picking logic)
        w.write("<H", counts['tri'])    # Triangle count
        w.write("<H", counts['quad'])   # Quad count
        
        # Padding to 0x40 (64 bytes) for the Model Header structure
        current = w.tell() - model_start
        if current < 64:
             w.write_bytes(b'\x00' * (64 - current))
        
        # --- SUB-BLOCKS ---
        # Each block's start position is recorded as a relative offset from model_start.
        
        # 1. Object Block (Code/Bones)
        code_offset = w.tell() - model_start
        self.build_object_block(w, xml_root)
        
        # 2. Material Block (Textures/Palettes)
        mat_offset = w.tell() - model_start
        self.build_material_block(w, xml_root)
        
        # 3. Polygon Block (Display Lists)
        poly_offset = w.tell() - model_start
        self.build_polygon_block(w, xml_root)
        
        # Final Block Alignment (4 bytes) - MUST be done before calculating poly_end!
        w.seek(0, 2)
        align_pad = (4 - (w.tell() % 4)) % 4
        w.write_bytes(b'\x00' * align_pad)
        
        # Now calculate poly_end AFTER alignment
        poly_end = w.tell() - model_start
        
        # --- Backfill Offsets ---
        # Now that we know where sub-blocks ended up, we overwrite the placeholders.
        total_size = w.tell() - model_start
        w.seek(model_start)
        w.write("<I", total_size) # Size relative to this header
        w.write("<I", code_offset)
        w.write("<I", mat_offset)
        w.write("<I", poly_offset)
        w.write("<I", poly_end)
        
        
        # Final MDL0 Section Size
        final_size = w.tell()
        struct.pack_into("<I", w.data, size_offset, final_size)
        return w.data

    def get_model_counts(self, xml_root):
        """Extracts model counts (materials, polygons, vertices, etc.) from IMD XML."""
        counts = {'mat': 0, 'poly': 0, 'vertex': 0, 'tri': 0, 'quad': 0}
        
        # Get primitive counts from output_info (these are triangle/quad counts, NOT object counts)
        output_info = xml_root.find(".//output_info")
        if output_info is not None:
             counts['vertex'] = int(output_info.get("vertex_size", 0))
             counts['tri'] = int(output_info.get("triangle_size", 0))
             counts['quad'] = int(output_info.get("quad_size", 0))
             
        # Material count from material_array
        mat_array = xml_root.find(".//material_array")
        if mat_array is not None:
            mat_count = int(mat_array.get("size", 0))
            if mat_count > 255:
                raise ValueError(f"Material count ({mat_count}) exceeds NSBMD limit of 255. Please reduce materials in your map.")
            counts['mat'] = mat_count
        
        # Polygon count from polygon_array (number of polygon objects, NOT primitives)
        poly_array = xml_root.find(".//polygon_array")
        if poly_array is not None:
            poly_count = int(poly_array.get("size", 0))
            if poly_count > 255:
                raise ValueError(f"Polygon count ({poly_count}) exceeds NSBMD limit of 255. Please split your map into smaller chunks.")
            counts['poly'] = poly_count
            
        return counts

    def get_bbox(self, xml_root):
        """Extracts bounding box and scale info from box_test node."""
        bbox = {'scale': 1.0, 'inv_scale': 1.0, 'xmin': 0, 'ymin': 0, 'zmin': 0, 'xmax': 0, 'ymax': 0, 'zmax': 0}
        
        box_test = xml_root.find(".//box_test")
        if box_test is not None:
            # Parse pos_scale
            # ImdModel.java: scaleFactor = 0.25 * 2^(default - posScale)
            # But XML contains 'pos_scale' attribute as integer?
            # ImdModel lines 772+: posScale is calculated.
            # In XML, box_test has 'pos_scale', 'xyz', 'whd'.
            
            # 'xyz' seems to be Min coords?
            # 'whd' seems to be Width/Height/Depth (Size).
            
            xyz = [float(x) for x in box_test.get("xyz", "0 0 0").split()]
            whd = [float(x) for x in box_test.get("whd", "0 0 0").split()]
            
            bbox['xmin'] = int(xyz[0] * 4096)
            bbox['ymin'] = int(xyz[1] * 4096)
            bbox['zmin'] = int(xyz[2] * 4096)
            
            bbox['xmax'] = int((xyz[0] + whd[0]) * 4096)
            bbox['ymax'] = int((xyz[1] + whd[1]) * 4096)
            bbox['zmax'] = int((xyz[2] + whd[2]) * 4096)
            
            # Scale?
            # NSBMD.cs Reads 'modelscale' (int / 4096f) and 'boundscale'.
            # Usually 1.0 (4096) for standard models.
            bbox['scale'] = 1.0
            bbox['inv_scale'] = 1.0
            
        return bbox

    def build_object_block(self, w, xml_root):
        """
        Constructs the Object Group (Bone/Scene definition) using node_array.
        Each "Object" in NSBMD corresponds to a Node in the IMD XML. These nodes
        form the transformation hierarchy (Scene Graph) of the model.
        """
        base_offset = w.tell()
        
        objects = []
        node_arr = xml_root.find(".//node_array")
        if node_arr is not None:
             objects = node_arr.findall("node")
        
        # Ensure we have at least one object (root node)
        if not objects:
            num_objects = 1
            is_dummy = True
        else:
            num_objects = len(objects)
            is_dummy = False
            
        # --- Dictionary: Maps object names to data ---
        # Pokemon-DS-Rom-Editor expects (NSBMD.cs line 512):
        #   stream.Skip(14 + (objnum * 4));
        # This skips: 14-byte dictionary + (objnum *4) offset table #1
        # Then it READS another (objnum * 4) offset table #2 at lines 522-524
        # So we must write TWO offset tables!
        
        self.build_nitro_dictionary(w, num_objects)
        
        # First offset table (will be skipped by loader - part of dictionary structure)
        w.write_bytes(b'\x00' * (4 * num_objects))

        # Second offset table (will be READ by loader - actual offsets)
        offsets_start = w.tell()
        w.write_bytes(b'\x00' * (4 * num_objects))

        
        # --- Names: 16-byte fixed-length ASCII strings ---
        if is_dummy:
             w.write_bytes(b"node0".ljust(16, b'\x00')[:16])
        else:
             for obj in objects:
                  name = obj.get("name", "node").encode('ascii', 'ignore')[:15]
                  w.write_bytes(name.ljust(16, b'\x00'))
                  
        # --- Object Data: Transform payloads ---
        obj_data_offsets = []
        
        if is_dummy:
             # Default root object with identity transforms
             pos = w.tell()
             obj_data_offsets.append(pos)
             w.write("<H", 0x0007) # Bit 0,1,2 set = No Trans, No Rot, No Scale
             w.write("<h", 0)      # Padding
             w.write_bytes(b'\x00' * 40) 
             
        else:
             for obj in objects:
                  pos = w.tell()
                  obj_data_offsets.append(pos)
                  
                  # Parse Transforms from XML (IMD uses standard floats)
                  scale = [float(x) for x in obj.get("scale", "1 1 1").split()]
                  rot = [float(x) for x in obj.get("rotate", "0 0 0").split()]
                  trans = [float(x) for x in obj.get("translate", "0 0 0").split()]
                  
                  # Flags: Determine which transform components follow.
                  # Bit 0: Translation disabled if set to 1
                  # Bit 1: Rotation disabled if set to 1
                  # Bit 2: Scale disabled if set to 1
                  # Bit 3: Visibility/Rendering (0=Show, 1=Hide)
                  
                  flag = 0
                  is_trans_ident = (trans == [0.0, 0.0, 0.0])
                  is_rot_ident = (rot == [0.0, 0.0, 0.0])
                  is_scale_ident = (scale == [1.0, 1.0, 1.0])
                  
                  if is_trans_ident: flag |= 1
                  if is_rot_ident:   flag |= 2
                  if is_scale_ident: flag |= 4
                  
                  w.write("<H", flag)
                  w.write("<h", 0) # Padding to 32-bit boundary
                  
                  # Payload: Translation -> Rotation -> Scale (Order defined by flag bits)
                  # NDS uses FX32 (Fixed Point 20.12) for position and scale.
                  
                  if not is_trans_ident:
                       w.write("<I", int(trans[0] * 4096))
                       w.write("<I", int(trans[1] * 4096))
                       w.write("<I", int(trans[2] * 4096))
                       
                  if not is_rot_ident:
                       # Note: NDS 3x3 Matrices use 9 fixed-point values.
                       # Conversion from Euler (XML) to Matrix is required for complex animations.
                       # Map Studio map nodes are mostly identity at this level (offsets handled in meshes).
                       # We write an Identity Matrix if non-identity rotation is detected as a fail-safe.
                       # TODO: Implement Euler -> Matrix if non-identity rotations are required for bones.
                       # Row 1-3 (Identity Matrix 4096 = 1.0)
                       w.write("<I", 4096); w.write("<I", 0); w.write("<I", 0)
                       w.write("<I", 0); w.write("<I", 4096); w.write("<I", 0)
                       w.write("<I", 0); w.write("<I", 0); w.write("<I", 4096)
                       
                  if not is_scale_ident:
                       w.write("<I", int(scale[0] * 4096))
                       w.write("<I", int(scale[1] * 4096))
                       w.write("<I", int(scale[2] * 4096))
                       
        # Post-process: Update the offsets table in the object block header
        curr = w.tell()
        w.seek(offsets_start)
        for off in obj_data_offsets:
             w.write("<I", off - base_offset)
        w.seek(curr)
        
        # Byte Alignment
        pad = (4 - (w.tell() % 4)) % 4
        w.write_bytes(b'\x00' * pad) 
        
    def build_material_block(self, w, xml_root):
        """
        Constructs the Material Group (Textures & Palettes linkage).
        """
        base_offset = w.tell()
        
        # Header: Pointers to texture and palette linkage blocks (4 bytes)
        # Read at NSBMD.cs:555-556
        tex_off_ptr = w.tell()
        w.write("<H", 0) # texoffset placeholder
        pal_off_ptr = w.tell()
        w.write("<H", 0) # paloffset placeholder

        
        # Parse Materials from XML
        materials = []
        mat_node = xml_root.find(".//material_array")
        if mat_node is not None:
             for m in mat_node.findall("material"):
                  materials.append(m)
        
        num_materials = len(materials)
        
        # --- Dictionary for Materials ---
        self.build_nitro_dictionary(w, num_materials)
        
        # Loader skips 16 + matnum*4 (NSBMD.cs:566)
        # The 16 includes the 4 byte tex/pal offsets + 12 additional? 
        # Actually, NSBMD.cs reads tex/pal (4 bytes) THEN skips 16.
        # So we need 16 bytes of "dictionary" even after tex/pal headers.
        
        # SKIPPED OFFSET TABLE
        w.write_bytes(b'\x00' * (4 * num_materials))
        
        # READ OFFSET TABLE (Read at NSBMD.cs:571)
        offsets_start = w.tell()
        w.write_bytes(b'\x00' * (4 * num_materials))
        
        # --- Names ---
        for m in materials:
             name = m.get("name", "mat").encode('ascii', 'ignore')[:15]
             w.write_bytes(name.ljust(16, b'\x00'))
             
        # --- Material Definitions: Binary payloads for each material ---
        mat_def_offsets = []
        for i, m in enumerate(materials):
            mat_def_start = w.tell()
            mat_def_offsets.append(mat_def_start - base_offset)
            
            # 1. Component Colors (Packed BGR555)
            diff = [int(x) for x in m.get("diffuse", "192 192 192").split()]
            amb = [int(x) for x in m.get("ambient", "64 64 64").split()]
            spec = [int(x) for x in m.get("specular", "255 255 255").split()]
            emi = [int(x) for x in m.get("emission", "0 0 0").split()]
            alpha = int(m.get("alpha", "31"))
            
            diff_bgr = self.rgb_to_bgr15(*diff)
            amb_bgr = self.rgb_to_bgr15(*amb)
            spec_bgr = self.rgb_to_bgr15(*spec)
            emi_bgr = self.rgb_to_bgr15(*emi)
            
            # --- Polygon Attributes (Hardware CMD 0x29 Format) ---
            p_id = int(m.get("polygon_id", "0"))
            fog = m.get("fog_flag", "0") == "1"
            
            # Lights 0-3: Enable state
            l0 = m.get("light0", "1") == "1"
            l1 = m.get("light1", "1") == "1"
            l2 = m.get("light2", "1") == "1"
            l3 = m.get("light3", "1") == "1"
            
            # Face Culling
            face = m.get("face", "front")
            show_front = face in ("front", "both")
            show_back = face in ("back", "both")
            
            # Pack PolyAttr: [ID(6) | Alpha(5) | Fog(1) | Modes... | Light(4)]
            p_attr = (l0 << 0) | (l1 << 1) | (l2 << 2) | (l3 << 3)
            p_attr |= (0 << 4) # Poly Mode: 0=Modulation, 1=Decal, 2=Toon
            if show_back: p_attr |= (1 << 6)
            if show_front: p_attr |= (1 << 7)
            if fog: p_attr |= (1 << 11)
            p_attr |= (alpha & 0x1F) << 16
            p_attr |= (p_id & 0x3F) << 24
            
            # Material Structure [60 bytes]
            w.write("<H", 0) # [0x00] Header (Dummy)
            w.write("<H", 0) # [0x02] Block Size Placeholder (Filled below)
            
            # [0x04] Diffuse + Ambient
            w.write("<I", (diff_bgr) | (amb_bgr << 16))
            # [0x08] Specular + Emission
            w.write("<I", (spec_bgr) | (emi_bgr << 16))
            
            # [0x0C] Polygon Attributes (Register 0x29)
            w.write("<I", p_attr)
            w.write("<I", 0) # [0x10] Attribute Mask (0xFFFFFFFF usually)
            
            # --- Texture Parameters ---
            w.write("<H", 0) # [0x14] Tex VRAM Offset
            
            # Wrapping Mode: Repeat, Clamp, Mirror (Flip)
            tiling_str = m.get("tex_tiling", "repeat repeat").split()
            tiling_u = tiling_str[0] if len(tiling_str) > 0 else "repeat"
            tiling_v = tiling_str[1] if len(tiling_str) > 1 else "repeat"
            
            t_param = 0
            if tiling_u == "clamp": t_param |= 1
            if tiling_v == "clamp": t_param |= 2
            if tiling_u == "flip":  t_param |= 4
            if tiling_v == "flip":  t_param |= 8
            
            # Detect SRT (Scale-Rotate-Translate) transforms
            tex_scale = [float(x) for x in m.get("tex_scale", "1 1").split()]
            tex_rot = float(m.get("tex_rotate", "0"))
            tex_trans = [float(x) for x in m.get("tex_translate", "0 0").split()]
            
            is_srt = (tex_scale != [1.0, 1.0]) or (tex_rot != 0.0) or (tex_trans != [0.0, 0.0])
            if is_srt:
                t_param |= (1 << 14) # Mode 1: SRT enabled
                
            w.write("<H", t_param) # [0x16] Texture Parameters
            w.write("<I", 0)       # [0x18] Parameter Mask
            w.write("<I", 0)       # [0x1C] Constant Overwrite
            
            # Texture Dimensions (Needed for UV wrapping logic)
            t_idx = int(m.get("tex_image_idx", -1))
            tw, th = 0, 0
            if t_idx != -1:
                tex_info = self.get_texture_info(xml_root, t_idx)
                if tex_info:
                    tw, th = tex_info['w'], tex_info['h']
            
            w.write("<h", tw) # [0x20] Width
            w.write("<h", th) # [0x22] Height
            w.write("<I", 0)  # [0x24] Padding/Unknown
            w.write("<I", 0)  # [0x28] Padding/Unknown
            
            # SRT Transform Payload (Optional block)
            if is_srt:
                w.write("<I", int(tex_scale[0] * 4096))
                w.write("<I", int(tex_scale[1] * 4096))
                # Note: Rotation scaling is approximate. 
                # DS hardware registers for texture gen are usually 4.12 fixed point.
                w.write("<h", int(tex_rot * 4096 / 360.0)) 
                w.write("<h", int(tex_trans[0] * 4096))
                w.write("<h", int(tex_trans[1] * 4096))
            
            # Size Finalization
            current_len = w.tell() - mat_def_start
            struct.pack_into("<H", w.data, mat_def_start + 2, current_len)
            
            # Alignment to 4-byte boundaries
            pad = (4 - (w.tell() % 4)) % 4
            w.write_bytes(b'\x00' * pad)

        # Update offsets table in material block header
        current = w.tell()
        w.seek(offsets_start)
        for off in mat_def_offsets:
             w.write("<I", off)
        w.seek(current)
        
        # --- Build Texture and Palette Maps ---
        # Associate Material IDs with Texture/Palette names for runtime binding.
        tex_map = {}
        pal_map = {}
        
        for i, m in enumerate(materials):
            t_idx = int(m.get("tex_image_idx", -1))
            p_idx = int(m.get("tex_palette_idx", -1))
            
            if t_idx != -1:
                t_name = self.get_texture_name(xml_root, t_idx)
                if t_name:
                    if t_name not in tex_map: tex_map[t_name] = []
                    tex_map[t_name].append(i)
            
            if p_idx != -1:
                p_name = self.get_palette_name(xml_root, p_idx)
                if p_name:
                    if p_name not in pal_map: pal_map[p_name] = []
                    pal_map[p_name].append(i)
        
        # --- Linkage Blocks (Texture & Palette) ---
        tex_linkage_pos = w.tell()
        self.build_linkage_block(w, tex_map, base_offset)
        pal_linkage_pos = w.tell()
        self.build_linkage_block(w, pal_map, base_offset)
        
        # Finalize material block header
        current = w.tell()
        w.seek(tex_off_ptr)
        w.write("<H", tex_linkage_pos - base_offset)
        w.seek(pal_off_ptr)
        w.write("<H", pal_linkage_pos - base_offset)
        w.seek(current)


    def build_linkage_block(self, w, item_map, section_start):
        """
        Constructs a linkage block mapping names (textures/palettes) to lists 
        of materials that use them.
        """
        base = w.tell()
        items = sorted(item_map.keys())
        count = len(items)
        
        # Header: Count and padding
        self.build_nitro_dictionary(w, count)
        
        if count == 0: return
            
        # TWO Offset Tables (skipped then read at NSBMD.cs:683-684)
        w.write_bytes(b'\x00' * (4 * count)) # SKIPPED
        offsets_table_start = w.tell()
        w.write_bytes(b'\x00' * (4 * count)) # READ

        
        # Names (16 bytes each)
        for name in items:
            n_bytes = name.encode('ascii', 'ignore')[:15]
            w.write_bytes(n_bytes.ljust(16, b'\x00'))
            
        # Data Payloads (Lists of material byte-indices)
        data_offsets = []
        for name in items:
            mat_indices = item_map[name]
            pos = w.tell()
            data_offsets.append({'pos': pos, 'count': len(mat_indices)})
            
            for mid in mat_indices:
                w.write("<B", mid)
                
            # Alignment to 4 bytes for next payload
            pad = (4 - (w.tell() % 4)) % 4
            w.write_bytes(b'\x00' * pad)

        # Fill relative offsets in the table
        curr = w.tell()
        w.seek(offsets_table_start)
        for entry in data_offsets:
            rel_offset = entry['pos'] - section_start
            num_pairs = entry['count']
            # Pack: [Count (16 bits) | Offset (16 bits)]
            val = (num_pairs << 16) | (rel_offset & 0xFFFF)
            w.write("<I", val)
        w.seek(curr)

    def get_texture_name(self, xml_root, idx):
        arr = xml_root.find(".//tex_image_array")
        if arr is not None:
            imgs = arr.findall("tex_image")
            if 0 <= idx < len(imgs):
                return imgs[idx].get("name")
        return None

    def get_palette_name(self, xml_root, idx):
        arr = xml_root.find(".//tex_palette_array")
        if arr is not None:
            pals = arr.findall("tex_palette")
            if 0 <= idx < len(pals):
                return pals[idx].get("name")
        return None

    def get_texture_info(self, xml_root, idx):
        arr = xml_root.find(".//tex_image_array")
        if arr is not None:
            imgs = arr.findall("tex_image")
            if 0 <= idx < len(imgs):
                img = imgs[idx]
                return {
                    'w': int(img.get("width", 0)),
                    'h': int(img.get("height", 0))
                }
        return None


    def build_polygon_block(self, w, xml_root):
        """
        Constructs the Polygon Group.
        This section contains the raw Display Lists (hardware commands) that 
        draw the geometry on the screen.
        """
        base_offset = w.tell()
        
        polygons = []
        poly_arr = xml_root.find(".//polygon_array")
        if poly_arr is not None:
             polygons = poly_arr.findall("polygon")
             
        num_polys = len(polygons)
        
        # --- Dictionary for Polygons ---
        self.build_nitro_dictionary(w, num_polys)
        
        # TWO Offset Tables (skipped then read at NSBMD.cs:762-766)
        w.write_bytes(b'\x00' * (4 * num_polys)) # SKIPPED
        offsets_start = w.tell()
        w.write_bytes(b'\x00' * (4 * num_polys)) # READ

        
        # --- Names ---
        for i, p in enumerate(polygons):
            name = f"polygon{i}".encode('ascii')[:15]
            w.write_bytes(name.ljust(16, b'\x00'))
            
        # --- Polygon Headers & Data Payloads ---
        poly_headers_offsets = []
        for i, p in enumerate(polygons):
            h_start = w.tell()
            poly_headers_offsets.append(h_start - base_offset)
            
            # Polygon Header Structure
            w.write("<h", 0)  # Dummy/Zero
            w.write("<h", 28) # Header Size (Fixed 28 bytes for this version)
            w.write("<I", 0)  # Reserved/Unknown
            
            # Offsets to the actual command stream (Display List)
            rel_offset_pos = w.tell()
            w.write("<I", 0) # [0x08] Relative Offset to Data (from Header Start)
            w.write("<I", 0) # [0x0C] Data Size (in bytes)
            
            # Padding to header end
            w.write_bytes(b'\x00' * 8) 
            
            # --- Display List Generation ---
            dl = BinaryWriter()
            mtx_prim = p.find("mtx_prim")
            if mtx_prim is not None:
                # Recursively parse meshes into FIFO commands
                self.process_mtx_prim(mtx_prim, dl) 
            
            dl_data = dl.data
            dl_size = len(dl_data)
            
            w.write_bytes(dl_data)
            
            # Backfill Header offsets for this polygon
            curr = w.tell()
            w.seek(rel_offset_pos)
            w.write("<I", 28) # Data follows immediately after 28-byte header
            w.write("<I", dl_size)
            w.seek(curr)
            
            # 4-byte Alignment for subsequent polygons
            pad = (4 - (w.tell() % 4)) % 4
            w.write_bytes(b'\x00' * pad)
            
        # Backfill Main Offsets Table
        curr = w.tell()
        w.seek(offsets_start)
        for off in poly_headers_offsets:
            w.write("<I", off)
        w.seek(curr)

    def rgb_to_bgr15(self, r, g, b):
        return ((b >> 3) << 10) | ((g >> 3) << 5) | (r >> 3)

    
    def process_mtx_prim(self, mtx_prim_node, w):
        """
        Parses an <mtx_prim> node and writes the corresponding Display List commands.
        An mtx_prim usually contains matrix stack adjustments followed by 
        primitive arrays (Triangles, Quads, etc.).
        """
        prim_array = mtx_prim_node.find("primitive_array")
        if prim_array is None:
            return

        for primitive in prim_array.findall("primitive"):
             prim_type = primitive.get("type", "triangles")
             
             # Hardware Primitive Type (Packed into BEGIN_VTXS command)
             hw_type = 0 # triangles
             if prim_type == "quads": hw_type = 1
             elif prim_type == "triangle_strip": hw_type = 2
             elif prim_type == "quad_strip": hw_type = 3
             
             # Command: BEGIN_VTXS (0x40)
             # Switches the geometry engine into a specific primitive rendering state.
             w.write("<B", self.CMD_BEGIN_VTXS)
             w.write("<B", hw_type) 
             w.write("<H", 0) # Padding to 32-bit word alignment
             
             # Process vertices, colors, normals, and UVs for this primitive
             self.process_primitive_nodes(primitive, w)
             
             # Command: END_VTXS (0x41)
             w.write("<B", self.CMD_END_VTXS)
             w.write_bytes(b'\x00\x00\x00') # Padding to 32-bit word alignment

    def process_primitive_nodes(self, primitive_node, w):
        """
        Iterates over individual vertex attributes (positions, normals, colors, UVs)
        and converts them into raw hardware commands.
        
        Note: NDS uses 4.12 fixed-point (FX12) for most spatial coordinates. 
        1.0 in float maps to 4096 in integer.
        """
        for node in primitive_node:
            tag = node.tag
            
            if tag == "mtx": 
                 # Command: RESTORE_MTX (0x14)
                 # Replaces the current matrix with one from the stack.
                 idx = int(node.get("idx", 0))
                 self.write_command(w, self.CMD_MTX_RESTORE, [idx])
            
            elif tag == "tex":
                 # Command: TEXCOORD (0x22)
                 # Sets S,T texture coordinates. In XML, these are floats (0.0 to 1.0).
                 # Hardware expects FX12 (multiplied by 16 if mapping to texels? 
                 # Actually, it's usually pixel-based or 1.0=16 for some reason in certain tools).
                 # We scale by 16 to match common Map Studio expectations.
                 st = [float(x) for x in node.get("st").split()]
                 s = int(st[0] * 16) 
                 t = int(st[1] * 16) 
                 
                 # Pack: [ T (16 bits) | S (16 bits) ]
                 param = ((t & 0xFFFF) << 16) | (s & 0xFFFF)
                 self.write_command(w, self.CMD_TEXCOORD, [param])
            
            elif tag == "nrm":
                 # Command: NORMAL (0x21)
                 # Sets the normal vector. Each component is 10-bit signed (1.9 format).
                 # Range: -1.0 to 0.998. Float 1.0 maps to 511.
                 xyz = [float(x) for x in node.get("xyz").split()]
                 nx = int(xyz[0] * 511) & 0x3FF
                 ny = int(xyz[1] * 511) & 0x3FF
                 nz = int(xyz[2] * 511) & 0x3FF
                 
                 # Pack: [0 (2 bits) | Z (10) | Y (10) | X (10)]
                 param = (nz << 20) | (ny << 10) | nx
                 self.write_command(w, self.CMD_NORMAL, [param])

            elif tag == "clr":
                 # Command: COLOR (0x20)
                 # Sets vertex color in BGR555 format.
                 rgb = [int(x) for x in node.get("rgb").split()]
                 r, g, b = rgb[0] >> 3, rgb[1] >> 3, rgb[2] >> 3 # Convert 8-bit to 5-bit
                 
                 # Pack: [0 | B (5) | G (5) | R (5)]
                 param = (b << 10) | (g << 5) | r
                 self.write_command(w, self.CMD_COLOR, [param])

            elif tag == "pos_xyz":
                 # Command: VTX_16 (0x23)
                 # Full 16-bit X,Y,Z vertex position.
                 xyz = [float(x) for x in node.get("xyz").split()]
                 vx = int(xyz[0] * 4096) & 0xFFFF 
                 vy = int(xyz[1] * 4096) & 0xFFFF
                 vz = int(xyz[2] * 4096) & 0xFFFF
                 
                 # Pack into two 32-bit words: [Y|X] and [P|Z]
                 p1 = (vy << 16) | vx
                 p2 = vz & 0xFFFF
                 self.write_command(w, self.CMD_VTX_16, [p1, p2])

            elif tag == "pos_xy":
                 # Command: VTX_XY (0x25) - Optimized XY-only update
                 xy = [float(x) for x in node.get("xy").split()]
                 vx = int(xy[0] * 4096) & 0xFFFF 
                 vy = int(xy[1] * 4096) & 0xFFFF
                 p1 = (vy << 16) | vx
                 self.write_command(w, self.CMD_VTX_XY, [p1])

            elif tag == "pos_xz":
                 # Command: VTX_XZ (0x26)
                 xz = [float(x) for x in node.get("xz").split()]
                 vx = int(xz[0] * 4096) & 0xFFFF 
                 vz = int(xz[1] * 4096) & 0xFFFF
                 p1 = (vz << 16) | vx
                 self.write_command(w, self.CMD_VTX_XZ, [p1])

            elif tag == "pos_yz":
                 # Command: VTX_YZ (0x27)
                 yz = [float(x) for x in node.get("yz").split()]
                 vy = int(yz[0] * 4096) & 0xFFFF 
                 vz = int(yz[1] * 4096) & 0xFFFF
                 p1 = (vz << 16) | vy
                 self.write_command(w, self.CMD_VTX_YZ, [p1])

            elif tag == "pos_diff":
                 # Command: VTX_DIFF (0x28)
                 # Relative vertex update using compact 10-bit signed differences.
                 xyz = [float(x) for x in node.get("xyz").split()]
                 vx = int(xyz[0] * 4096) & 0x3FF
                 vy = int(xyz[1] * 4096) & 0x3FF
                 vz = int(xyz[2] * 4096) & 0x3FF
                 
                 # Pack: [0 | Z (10) | Y (10) | X (10)]
                 param = (vz << 20) | (vy << 10) | vx
                 self.write_command(w, self.CMD_VTX_DIFF, [param])

    def write_command(self, w, cmd, params):
         """
         Writes a single NDS Display List command.
         Note: This function uses a simplified 'one command per word' packing.
         Real hardware supports packing 4 commands into one control word, 
         but for conversion simplicity, we use the unpacked format (Command + Pads).
         """
         # Header: [Command Byte] [00 00 00]
         w.write("<B", cmd)
         w.write_bytes(b'\x00\x00\x00')
         
         # Write Parameters (32-bit each)
         for p in params:
             w.write("<I", p)


    def build_tex0_block(self, xml_root):
        """
        Constructs the TEX0 (Texture 0) section of the NSBMD file.
        This block stores the raw bitmap and palette data, as well as the 
        lookup dictionaries that map names to VRAM offsets.
        """
        w = BinaryWriter()
        w.write("<4s", b"TEX0")   # Magic Number
        size_offset = w.tell()
        w.write("<I", 0)          # Placeholder for Size
        
        # Parse XML for Texture and Palette Data
        textures = []
        tex_image_array = xml_root.find(".//tex_image_array")
        if tex_image_array is not None:
             for tex_node in tex_image_array.findall("tex_image"):
                 name = tex_node.get("name")
                 width = int(tex_node.get("width"))
                 height = int(tex_node.get("height"))
                 fmt_str = tex_node.get("format")
                 color0_mode = tex_node.get("color0_mode")
                 
                 # Standard NDS Texture Format Values
                 # 1: A3I5
                 # 2: 4-color (Palette4)
                 # 3: 16-color (Palette16)
                 # 4: 256-color (Palette256)
                 # 5: 4x4-Texel
                 # 6: A5I3
                 # 7: Direct Color
                 fmt_map = {
                     "a3i5": 1,
                     "palette4": 2,
                     "palette16": 3,
                     "palette256": 4,
                     "texture4x4": 5, # Guessing name for 4x4 if used
                     "a5i3": 6,
                     "direct": 7
                 }
                 
                 fmt = fmt_map.get(fmt_str, 0)
                 if fmt == 0:
                     print(f"Warning: Unknown texture format '{fmt_str}', defaulting to 0")

                 bitmap_node = tex_node.find("bitmap")
                 hex_data = ""
                 if bitmap_node is not None and bitmap_node.text:
                     hex_data = bitmap_node.text.strip().replace(" ", "").replace("\n", "")
                 
                 data = bytes.fromhex(hex_data)
                 textures.append({
                     "name": name,
                     "width": width,
                     "height": height,
                     "format": fmt,
                     "color0": 1 if color0_mode == "transparency" else 0,
                     "data": data
                 })
                 
        palettes = []
        tex_palette_array = xml_root.find(".//tex_palette_array")
        if tex_palette_array is not None:
            for pal_node in tex_palette_array.findall("tex_palette"):
                name = pal_node.get("name")
                hex_data = ""
                if pal_node.text:
                     hex_data = pal_node.text.strip().replace(" ", "").replace("\n", "")
                data = bytes.fromhex(hex_data)
                palettes.append({
                    "name": name,
                    "data": data
                })

        # --- Data Aggregation & Alignment ---
        tex_data_blob = bytearray()
        tex_offsets = []
        for t in textures:
            tex_offsets.append(len(tex_data_blob))
            tex_data_blob.extend(t["data"])
            pad = (8 - (len(tex_data_blob) % 8)) % 8 # 8-byte alignment for hardware VRAMDMA
            tex_data_blob.extend(b'\x00' * pad)
            
        pal_data_blob = bytearray()
        pal_offsets = []
        for p in palettes:
            pal_offsets.append(len(pal_data_blob))
            pal_data_blob.extend(p["data"])
            pad = (8 - (len(pal_data_blob) % 8)) % 8
            pal_data_blob.extend(b'\x00' * pad)

        # Count and Dictionary Sizes
        tex_info_size = 16 + (len(textures) * 28) # Header(16) + Dict(4*N) + Struct(8*N) + Name(16*N)
        pal_info_size = 16 + (len(palettes) * 24) # Header(16) + Dict(4*N) + Struct(4*N) + Name(16*N)

        # --- Write Header ---
        # Base Offset = Block Start (current w.tell())
        # The offsets in the header are RELATIVE TO BLOCK START.
        
        # Structure derived from NSBTXLoader read pattern
        # Size (0x4 - 0x7) Filled later
        # [0x8] Padding (4)
        w.write_bytes(b'\x00' * 4)
        
        # [0xC] TexDataSize >> 3 (2)
        w.write("<H", len(tex_data_blob) >> 3)
        
        # [0xE] InfoHeaderSize / Padding (6)
        w.write_bytes(b'\x00' * 6)
        
        # [0x14] Tex Data Offset (4) 
        # Layout: Header -> TexInfo -> TexData -> SpData -> PalInfo -> PalData
        # Need to calc block sizes first.
        header_size = 60 # Fixed basic header size
        
        # We need to compute where everything lands.
        # Tex Info starts immediately after header?
        # But `NSBTXLoader` logic:
        #  read header... then `stream.Skip(1); texnum = ...`
        # So Info is right after header.
        
        offset_tex_info = header_size
        offset_tex_data = offset_tex_info + tex_info_size
        
        # Alignment for Tex Data? 
        # Usually data blocks are aligned to 0x8 or 0x16?
        # Let's align offset_tex_data to 8
        pad_td = (8 - (offset_tex_data % 8)) % 8
        offset_tex_data += pad_td
        
        offset_sp_data = offset_tex_data + len(tex_data_blob)
        offset_pal_info = offset_sp_data # No SP data for now
        
        offset_pal_data = offset_pal_info + pal_info_size
        pad_pd = (8 - (offset_pal_data % 8)) % 8
        offset_pal_data += pad_pd
        
        # Write offsets
        w.write("<I", offset_tex_data)
        
        # [0x18] SP fields (Zero for now)
        w.write_bytes(b'\x00' * 4) # Skip 4
        w.write("<H", 0) # SpTex Size
        w.write_bytes(b'\x00' * 6) # Skip 6
        w.write("<I", 0) # SpTex Offset
        w.write("<I", 0) # SpData Offset
        
        # [0x2C] Skip 4
        w.write_bytes(b'\x00' * 4)
        
        # [0x30] Pal Data Size >> 3 (2)
        w.write("<H", len(pal_data_blob) >> 3)
        
        # [0x32] Skip 2
        w.write_bytes(b'\x00' * 2)
        
        # [0x34] Pal Info Offset (4)
        w.write("<I", offset_pal_info)
        
        # [0x38] Pal Data Offset (4)
        w.write("<I", offset_pal_data)
        
        # Ensure we are at 0x3C (60 bytes)
        current = w.tell() - size_offset + 4 # relative to block start
        if current < 60:
             w.write_bytes(b'\x00' * (60 - current))
             
        # --- Write Tex Info Block ---
        w.write("<B", 0) 
        w.write("<B", len(textures))
        
        # Skip Dictionary (14 + 4*N)
        dict_size = 14 + (len(textures) * 4)
        w.write_bytes(b'\x00' * dict_size)
        
        # Texture Structs (8 bytes each)
        for i, t in enumerate(textures):
            # Offset >> 3
            w.write("<H", tex_offsets[i] >> 3)
            
            # Param
            # Format: [0-2]?? | Color0[13] | Format[10-12] | Height[7-9] | Width[4-6]
            # Width/Height values are Shifts. 8 << val. 
            # val = log2(dim / 8).
            # e.g W=8 -> val=0. W=16 -> val=1.
            import math
            try:
                w_shift = int(math.log2(t["width"] // 8))
                h_shift = int(math.log2(t["height"] // 8))
            except ValueError:
                w_shift = 0
                h_shift = 0
                
            fmt = t["format"]
            col0 = t["color0"]
            
            param = 0
            param |= (w_shift & 7) << 4
            param |= (h_shift & 7) << 7
            param |= (fmt & 7) << 10
            param |= (col0 & 1) << 13
            
            w.write("<H", param)
            w.write_bytes(b'\x00' * 4) # Skip 4
            
        # Texture Names (16 bytes fixed)
        for t in textures:
            name_bytes = t["name"].encode('ascii', 'ignore')[:15]
            w.write_bytes(name_bytes)
            w.write_bytes(b'\x00' * (16 - len(name_bytes)))
            
        # Padding to align Tex Data
        curr_len = w.tell() - size_offset + 4
        if curr_len < offset_tex_data:
            w.write_bytes(b'\x00' * (offset_tex_data - curr_len))
            
        # --- Write Tex Data ---
        w.write_bytes(tex_data_blob)
        
        # --- Write Pal Info Block ---
        # Padding to align Pal Info? (offset_pal_info)
        curr_len = w.tell() - size_offset + 4
        if curr_len < offset_pal_info:
             w.write_bytes(b'\x00' * (offset_pal_info - curr_len))
             
        w.write("<B", 0)
        w.write("<B", len(palettes))
        
        dict_size = 14 + (len(palettes) * 4)
        w.write_bytes(b'\x00' * dict_size)
        
        # Palette Structs (4 bytes each based on NSBTXLoader)
        for i, p in enumerate(palettes):
            # Offset >> 3
            w.write("<H", pal_offsets[i] >> 3)
            w.write_bytes(b'\x00' * 2) # Skip 2
            
        # Palette Names
        for p in palettes:
            name_bytes = p["name"].encode('ascii', 'ignore')[:15]
            w.write_bytes(name_bytes)
            w.write_bytes(b'\x00' * (16 - len(name_bytes)))
            
        # --- Write Pal Data ---
        curr_len = w.tell() - size_offset + 4
        if curr_len < offset_pal_data:
            w.write_bytes(b'\x00' * (offset_pal_data - curr_len))
            
        w.write_bytes(pal_data_blob)
        
        # Final Alignment
        final_len = w.tell()
        pad = (4 - (final_len % 4)) % 4
        w.write_bytes(b'\x00' * pad)
        
        total_size = w.tell()
        struct.pack_into("<I", w.data, size_offset, total_size)
        return w.data

    def convert(self, input_path, output_path, export_mode):
        """
        Orchestrates the conversion from IMD XML to NSBMD Binary.
        This method parses the XML, builds the individual binary sections, 
        and assembles the final file with correct headers and offsets.
        """
        xml_root = self.parse_imd(input_path)
        
        # --- File Header (BMD0/BTX0) ---
        signature = b"BMD0"
        if export_mode == 'texture':
            signature = b"BTX0"
            
        self.writer.write("<4s", signature)       # Signature
        self.writer.write("<H", 0xFEFF)           # Byte Order Mark (Little Endian)
        self.writer.write("<H", 0x0002)           # Version (2.0) - Required by DSPRE
        
        file_size_offset = self.writer.tell()
        self.writer.write("<I", 0)                # Total File Size Placeholder
        
        # Determine number of chunks based on mode
        chunks_to_write = []
        if export_mode == 'model':
            chunks_to_write = ['MDL0']
        elif export_mode == 'texture':
            chunks_to_write = ['TEX0']
        elif export_mode == 'both':
            chunks_to_write = ['MDL0', 'TEX0']
            
        # Number of blocks - CRITICAL FIX:
        # Pokemon-DS-Rom-Editor reads this as 32-bit, then shifts right by 16 bits
        # to get num_blocks. So we encode: (num_blocks << 16) | header_size
        # See NSBMD.cs lines 1097-1098:
        #   int numblock = reader.ReadInt32();
        #   numblock >>= 16;
        num_blocks = len(chunks_to_write)
        header_size = 16
        self.writer.write("<I", (header_size) | (num_blocks << 16)) 
        
        # --- Block Offset Table ---
        # Nitro files require a table of absolute offsets to each block immediately following the header
        block_offsets_pos = self.writer.tell()
        for _ in range(num_blocks):
            self.writer.write("<I", 0) # Placeholder for block offset
            
        # --- Write Chunks ---
        actual_offsets = []
        for chunk_tag in chunks_to_write:
            # Align each block to 4 bytes
            align_pad = (4 - (self.writer.tell() % 4)) % 4
            self.writer.write_bytes(b'\x00' * align_pad)
            
            actual_offsets.append(self.writer.tell())
            if chunk_tag == 'MDL0':
                print("Building MDL0 (Model Data)...")
                mdl0_data = self.build_mdl0_block(xml_root)
                self.writer.write_bytes(mdl0_data)
            elif chunk_tag == 'TEX0':
                print("Building TEX0 (Texture Data)...")
                tex0_data = self.build_tex0_block(xml_root)
                self.writer.write_bytes(tex0_data)
        
        # --- Backfill Block Offsets ---
        curr_pos = self.writer.tell()
        self.writer.seek(block_offsets_pos)
        for offset in actual_offsets:
            self.writer.write("<I", offset)
        self.writer.seek(curr_pos)
        
        # --- Finalize ---
        # Update Total File Size in Header
        total_size = self.writer.tell()
        struct.pack_into("<I", self.writer.data, file_size_offset, total_size)
        
        print(f"Writing {total_size} bytes to {output_path}...")
        with open(output_path, 'wb') as f:
            f.write(self.writer.data)
        print("Done.")

def main():
    """
    Entry point for the converter.
    Emulates the command-line interface of g3dcvtr.exe for compatibility 
    with existing Map Studio export pipelines.
    """
    
    parser = argparse.ArgumentParser(description="IMD to NSBMD Converter (g3dcvtr replacement)")
    parser.add_argument("input", help="Input IMD file path")
    
    # Flags matching original tool
    parser.add_argument("-eboth", action="store_true", help="Export both Model (MDL0) and Texture (TEX0) data")
    parser.add_argument("-emdl", action="store_true", help="Export Model (MDL0) data only")
    parser.add_argument("-etex", action="store_true", help="Export Texture (TEX0) data only") # Rare use-case
    parser.add_argument("-o", "--output", help="Output NSBMD file path", required=True)
    
    # Handle unknown arguments gracefully (some tools pass extra flags we might ignore)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Ignoring unknown arguments: {unknown}")

    # Determine Export Mode
    export_mode = 'model'
    if args.eboth:
        export_mode = 'both'
    elif args.etex:
        export_mode = 'texture' 
    
    converter = NSBMDConverter()
    converter.convert(args.input, args.output, export_mode)

if __name__ == "__main__":
    main()

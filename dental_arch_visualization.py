import numpy as np
from vedo import Plotter, Text2D, Line, Rectangle, Text3D, Points, Grid, Disc, colors, Sphere, Video # Added Video
import logging
# import os # Not strictly needed in this version

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DentalArchVisualizer: # This is your T-Scan Grid visualizer
    def __init__(self, processor, video_filename="tscan_grid_animation.mp4"): # video_filename for later
        self.processor = processor
        if self.processor.cleaned_data is None: self.processor.create_force_matrix()
            
        self.num_data_teeth = len(self.processor.tooth_ids) if self.processor.tooth_ids else 0
        if self.num_data_teeth == 0:
            logging.error("No tooth data found. Cannot initialize T-Scan style grid."); self._initialize_empty_state(); return

        self.arch_layout_width = 16.0
        self.arch_layout_depth = 10.0
        self.tooth_cell_definitions = self._define_explicit_tscan_layout(self.num_data_teeth)
        
        self.max_force_for_scaling = self.processor.max_force_overall if hasattr(self.processor, 'max_force_overall') else 100.0

        self.plotter = Plotter(bg=(0.92, 0.92, 0.92), axes=0, title="T-Scan Style Force Grid")
        self.plotter.camera.ParallelProjectionOn()
        self._fit_camera_to_grid()

        self.timestamps = self.processor.timestamps
        self.current_timestamp_idx = 0
        self.last_animated_timestamp = None
        self.selected_tooth_id_grid = None # For highlighting

        self.grid_outline_actors = {} 
        self.tooth_label_actors = {}  
        self.intra_tooth_heatmap_actors = [] # Reverted to list for recreate-each-frame
        self.force_percentage_actors = []    
        self.force_percentage_bg_actors = [] 

        self.left_right_bar_actor_left = None; self.left_right_bar_actor_right = None
        self.left_bar_label_actor = None; self.right_bar_label_actor = None
        self.left_bar_percentage_actor = None; self.right_bar_percentage_actor = None
        
        self.cof_trajectory_line_actor = None 
        self.cof_current_marker_actor = None  
        self.time_text_actor = None   
        self.selected_tooth_info_text_actor = None # For detailed info on click        

        self._initialize_static_grid_elements() 
        if self.tooth_cell_definitions: 
            self.processor.calculate_cof_trajectory(self.tooth_cell_definitions)
        
        self.plotter.add_callback('mouse click', self._on_mouse_click)

        # For saving animation
        self.video_filename = video_filename
        self.video_writer = None
        self.is_recording = False
        self.animation_controller_ref = None # For graph link

    def _initialize_empty_state(self):
        # ... (ensure all actor attributes are reset) ...
        self.tooth_cell_definitions = {}; self.plotter = Plotter(bg='lightgrey'); self.timestamps = []
        self.current_timestamp_idx=0; self.last_animated_timestamp=None; self.intra_tooth_heatmap_actors=[] # List
        self.tooth_label_actors={}; self.force_percentage_actors=[]; self.time_text_actor=None; 
        self.grid_outline_actors={}; self.force_percentage_bg_actors=[] # List
        self.left_right_bar_actor_left=None; self.left_right_bar_actor_right=None
        self.left_bar_label_actor=None; self.right_bar_label_actor=None
        self.left_bar_percentage_actor=None; self.right_bar_percentage_actor=None
        self.cof_trajectory_line_actor = None; self.cof_current_marker_actor = None
        self.selected_tooth_info_text_actor = None; self.selected_tooth_id_grid = None
        self.video_writer = None; self.is_recording = False
        self.animation_controller_ref = None


    # --- ADDED VIDEO RECORDING METHODS ---
# In dental_arch_visualization.py -> start_recording method

    def start_recording(self, filename=None, fps=10):
        if filename: self.video_filename = filename
        if self.video_writer is not None: 
            logging.warning("Video writer already exists. Closing previous one.")
            self.video_writer.close()
        
        video_fps = fps # Use provided fps or determine from timer later if fps is None

        if fps is None: # If fps is not explicitly provided
            if self.plotter and self.plotter.timer_callbacks:
                timer_cb_info = next((cb for cb in self.plotter.callbacks if cb.get('event') == 'timer'), None)
                if timer_cb_info and timer_cb_info.get('dt', 0) > 0:
                    video_fps = 1000 / timer_cb_info['dt']
                else: 
                    video_fps = 10 
            else: 
                video_fps = 10
        
        try:
            # --- SPECIFY OPENCV BACKEND ---
            self.video_writer = Video(name=self.video_filename, fps=video_fps, backend='opencv') 
            # --- End Specification ---
            self.is_recording = True
            logging.info(f"REC: Started video recording to {self.video_filename} at {video_fps:.1f} FPS using OpenCV.")
        except Exception as e:
            logging.error(f"REC: Failed to start video recording with OpenCV: {e}")
            # Try ffmpeg as a fallback if opencv fails and ffmpeg was the original default
            try:
                logging.info("REC: OpenCV backend failed, trying ffmpeg backend...")
                self.video_writer = Video(name=self.video_filename, fps=video_fps, backend='ffmpeg')
                self.is_recording = True
                logging.info(f"REC: Started video recording to {self.video_filename} at {video_fps:.1f} FPS using ffmpeg.")
            except Exception as e2:
                logging.error(f"REC: Failed to start video recording with ffmpeg backend as well: {e2}")
                self.video_writer = None; self.is_recording = False

    def stop_recording(self):
        if self.is_recording and self.video_writer is not None:
            try:
                self.video_writer.close()
                logging.info(f"REC: Stopped video recording. Video saved to {self.video_filename}")
            except Exception as e:
                logging.error(f"REC: Error closing video writer: {e}")
        self.video_writer = None; self.is_recording = False

    def add_frame_to_video(self):
        if self.is_recording and self.video_writer is not None:
            try: 
                self.video_writer.add_frame()
                # logging.debug("REC: Added frame to video.") # Can be verbose
            except Exception as e: 
                logging.error(f"REC: Error adding frame: {e}")
                # self.stop_recording() # Optionally stop if adding frame fails repeatedly
    # --- END VIDEO RECORDING METHODS ---

    def set_animation_controller_for_graph_link(self, controller): # For graph link
        self.animation_controller_ref = controller

    def _fit_camera_to_grid(self): # ... (same) ...
        if not self.tooth_cell_definitions: self.plotter.camera.SetParallelScale(10); return
        all_props = list(self.tooth_cell_definitions.values()) if isinstance(self.tooth_cell_definitions, dict) else self.tooth_cell_definitions
        if not all_props: self.plotter.camera.SetParallelScale(10); return
        all_x = [p['center'][0] + p['width']/2 for p in all_props] + [p['center'][0] - p['width']/2 for p in all_props]
        all_y = [p['center'][1] + p['height']/2 for p in all_props] + [p['center'][1] - p['height']/2 for p in all_props]
        min_cell_y_bottom = min(p['center'][1] - p['height']/2 for p in all_props) if all_props else 0
        text_space_below = 3.0; all_y.append(min_cell_y_bottom - text_space_below) 
        if not all_x or not all_y: self.plotter.camera.SetParallelScale(10); return
        min_x, max_x = min(all_x), max(all_x); min_y, max_y = min(all_y), max(all_y)
        padding_factor = 1.15; view_h = (max_y - min_y) * padding_factor; view_w = (max_x - min_x) * padding_factor
        scale = max(view_h, view_w, 1.0) / 2.0;  scale = max(0.1, scale)
        self.plotter.camera.SetParallelScale(scale)
        fx, fy = (min_x+max_x)/2, (min_y+max_y)/2
        self.plotter.camera.SetFocalPoint(fx, fy, 0); self.plotter.camera.SetPosition(fx, fy, 10)

    def _define_explicit_tscan_layout(self, num_teeth_from_data): # ... (same) ...
        layout = {} 
        if num_teeth_from_data == 0 or not self.processor.tooth_ids: return layout
        base_arch_width_for_centers = self.arch_layout_width*0.80; base_arch_depth_for_centers = self.arch_layout_depth*0.70
        arch_centers_3d = self._get_arch_positions_for_layout(num_teeth_from_data,base_arch_width_for_centers,base_arch_depth_for_centers)
        arch_centers_xy = [ac[:2] for ac in arch_centers_3d]
        if num_teeth_from_data > 1:
            sorted_x_centers = sorted([c[0] for c in arch_centers_xy])
            dx = np.abs(np.diff(sorted_x_centers)); avg_spacing_x = np.mean(dx) if len(dx) > 0 else base_arch_width_for_centers / num_teeth_from_data
            base_cell_w = avg_spacing_x*0.90; base_cell_h = base_cell_w*1.1 
        else: base_cell_w = self.arch_layout_width*0.15; base_cell_h = self.arch_layout_depth*0.15
        base_cell_w = max(0.7,base_cell_w); base_cell_h = max(0.9,base_cell_h)
        for i in range(num_teeth_from_data):
            actual_id = self.processor.tooth_ids[i]; center_xy = arch_centers_xy[i]
            current_w = base_cell_w; current_h = base_cell_h
            norm_x_center = abs(center_xy[0])/(base_arch_width_for_centers/2.0) if base_arch_width_for_centers > 0 else 0
            width_scale_factor=1.0; height_scale_factor=1.0
            if norm_x_center > 0.75: width_scale_factor=1.35; height_scale_factor=0.85;
            elif norm_x_center > 0.50: width_scale_factor=1.1; height_scale_factor=1.0;
            elif norm_x_center < 0.10: width_scale_factor=0.70; height_scale_factor=1.20;
            elif norm_x_center < 0.35: width_scale_factor=0.85; height_scale_factor=1.10;
            final_w = current_w*width_scale_factor; final_h = current_h*height_scale_factor
            final_w = max(0.6,final_w); final_h = max(0.8,final_h)
            layout[i] = {'center':center_xy, 'width':final_w, 'height':final_h, 'actual_id':actual_id}
        return layout

    def _get_arch_positions_for_layout(self, num_teeth, arch_width, arch_depth): # ... (same) ...
        if num_teeth == 0: return np.array([])
        if num_teeth == 1: x_coords = np.array([0.0])
        else: x_coords = np.linspace(-arch_width/2, arch_width/2, num_teeth)
        k = arch_depth / ((arch_width/2)**2) if arch_width != 0 else 0
        positions = []; [positions.append(np.array([x, arch_depth - k*(x**2),0])) for x in x_coords]
        return np.array(positions)

    def _initialize_static_grid_elements(self):
        """Creates ONLY static actors once. Outlines and labels now stored in dicts."""
        if not self.tooth_cell_definitions: return
        
        # Clear previous if any (e.g. re-initialization scenario)
        if self.grid_outline_actors: self.plotter.remove(list(self.grid_outline_actors.values()))
        if self.tooth_label_actors: self.plotter.remove(list(self.tooth_label_actors.values()))
        self.grid_outline_actors.clear()
        self.tooth_label_actors.clear()

        actors_to_add_at_once = []
        for _layout_idx, cell_prop in self.tooth_cell_definitions.items():
            tooth_id = cell_prop['actual_id']
            cx, cy = cell_prop['center']; w, h = cell_prop['width'], cell_prop['height']
            p1, p2 = (cx - w/2, cy - h/2), (cx + w/2, cy + h/2)
            
            outline = Rectangle(p1, p2, c=(0.3,0.3,0.3), alpha=0.8); outline.lw(1.0)
            outline.name = f"Outline_Tooth_{tooth_id}"; outline.pickable = True
            self.grid_outline_actors[tooth_id] = outline # Store in dict by tooth_id
            actors_to_add_at_once.append(outline)

            label_pos_xy = (cx, cy + h * 0.5 + 0.2)
            text_s = h * 0.30; text_s = max(0.25, min(text_s, 0.5)) 
            label = Text3D(str(tooth_id), pos=(label_pos_xy[0], label_pos_xy[1], 0.12), 
                           s=text_s, c=(0.05,0.05,0.3), justify='center-center', depth=0.01) 
            self.tooth_label_actors[tooth_id] = label # Store in dict
            actors_to_add_at_once.append(label)
        
        if actors_to_add_at_once:
            self.plotter.add(actors_to_add_at_once)

    def _create_intra_tooth_heatmap(self, cell_prop, forces_on_this_tooth_sensors):
        # ... (same as before) ...
        if not forces_on_this_tooth_sensors: return None
        num_sp = len(forces_on_this_tooth_sensors); cx, cy = cell_prop['center']; cw, ch = cell_prop['width'], cell_prop['height']
        if num_sp == 0: return None
        heatmap_grid_resolution_param = (1,1); heatmap_grid = Grid(s=[cw*0.96,ch*0.96],res=heatmap_grid_resolution_param)
        heatmap_grid.pos(cx,cy,0.05).lw(0).alpha(0.95)
        heatmap_grid.name = f"Heatmap_Tooth_{cell_prop['actual_id']}"; heatmap_grid.pickable = True
        grid_points_forces = np.zeros(heatmap_grid.npoints) 
        if num_sp == 4:
            grid_points_forces[2]=forces_on_this_tooth_sensors.get(1,0); grid_points_forces[3]=forces_on_this_tooth_sensors.get(2,0) 
            grid_points_forces[0]=forces_on_this_tooth_sensors.get(3,0); grid_points_forces[1]=forces_on_this_tooth_sensors.get(4,0) 
        elif num_sp > 0: avg_force=np.mean(list(forces_on_this_tooth_sensors.values())); grid_points_forces.fill(avg_force)
        heatmap_grid.pointdata["forces"] = np.nan_to_num(grid_points_forces)
        custom_cmap_rgb = ['darkblue',(0,0,1),(0,1,0),(1,1,0),(1,0,0)]; heatmap_grid.cmap(custom_cmap_rgb,"forces",vmin=0,vmax=self.max_force_for_scaling)
        return heatmap_grid


    def render_arch(self, timestamp):
        if not self.tooth_cell_definitions: self.plotter.render(); return

        # Clear dynamic actors from previous frame
        actors_to_remove = []
        if self.time_text_actor: actors_to_remove.append(self.time_text_actor); self.time_text_actor = None
        
        # Clear lists of actors (these are lists because they are recreated each frame)
        # For intra_tooth_heatmap_actors, it became a dict in the previous step to hold actor references
        # for the highlighting logic if we were updating. If we are fully recreating, it should be a list.
        # Let's assume for this "recreate-all-dynamic" version, they are simple lists.
        if self.intra_tooth_heatmap_actors: 
            # If it was a dict {id: actor}, use list(self.intra_tooth_heatmap_actors.values())
            # If it became a list of actors current_frame_heatmaps_list, use that.
            if isinstance(self.intra_tooth_heatmap_actors, dict):
                actors_to_remove.extend(list(self.intra_tooth_heatmap_actors.values()))
            else: # Assuming it's a list
                actors_to_remove.extend(self.intra_tooth_heatmap_actors)
            self.intra_tooth_heatmap_actors.clear() # Clears list or dict

        if self.force_percentage_actors: actors_to_remove.extend(self.force_percentage_actors); self.force_percentage_actors.clear()
        if self.force_percentage_bg_actors: actors_to_remove.extend(self.force_percentage_bg_actors); self.force_percentage_bg_actors.clear()
        
        # ... (Clearing for L/R bars and COF as before) ...
        if self.left_right_bar_actor_left: actors_to_remove.append(self.left_right_bar_actor_left); self.left_right_bar_actor_left = None
        if self.left_right_bar_actor_right: actors_to_remove.append(self.left_right_bar_actor_right); self.left_right_bar_actor_right = None
        if self.left_bar_label_actor: actors_to_remove.append(self.left_bar_label_actor); self.left_bar_label_actor = None
        if self.right_bar_label_actor: actors_to_remove.append(self.right_bar_label_actor); self.right_bar_label_actor = None
        if self.left_bar_percentage_actor: actors_to_remove.append(self.left_bar_percentage_actor); self.left_bar_percentage_actor = None
        if self.right_bar_percentage_actor: actors_to_remove.append(self.right_bar_percentage_actor); self.right_bar_percentage_actor = None
        if self.cof_trajectory_line_actor: actors_to_remove.append(self.cof_trajectory_line_actor); self.cof_trajectory_line_actor = None
        if self.cof_current_marker_actor: actors_to_remove.append(self.cof_current_marker_actor); self.cof_current_marker_actor = None
        if actors_to_remove: self.plotter.remove(actors_to_remove)


        # Add time text (recreated)
        self.time_text_actor = Text2D(f"Time: {timestamp:.1f}s",pos="bottom-left",c='k',bg=(1,1,1),alpha=0.7,s=0.7)
        
        ordered_pairs, forces_all_sensor_points = self.processor.get_all_forces_at_time(timestamp)
        if not ordered_pairs: self.plotter.add(self.time_text_actor).render(); return
        
        force_map_all_sensors = {p:f for p,f in zip(ordered_pairs,forces_all_sensor_points)}
        total_force_on_arch_this_step = sum(f for f in forces_all_sensor_points if np.isfinite(f) and f > 0); total_force_on_arch_this_step = max(total_force_on_arch_this_step,1e-6)
        force_left_side=0; force_right_side=0

        # These lists will hold actors created in this frame
        current_frame_heatmaps_list = [] 
        current_frame_perc_bgs_list = [] 
        current_frame_perc_texts_list = []

        for _layout_idx, cell_prop in self.tooth_cell_definitions.items():
            tooth_id = cell_prop['actual_id']
            
            # Highlighting Logic for Outline (outlines are persistent, modified in place)
            outline_actor = self.grid_outline_actors.get(tooth_id) 
            if outline_actor:
                if self.selected_tooth_id_grid == tooth_id:
                    outline_actor.color('hotpink').lw(3.5).alpha(1.0).linecolor('magenta')
                else:
                    outline_actor.color((0.3,0.3,0.3)).lw(0.5).alpha(0.75).linecolor('black')
            
            current_tooth_total_force = 0
            sensor_ids_for_this_tooth = [spid for tid,spid in self.processor.ordered_tooth_sensor_pairs if tid==tooth_id]
            forces_on_this_tooth_sensors = {}
            for sp_id in sensor_ids_for_this_tooth:
                force = force_map_all_sensors.get((tooth_id,sp_id),0.0); force=np.nan_to_num(force)
                forces_on_this_tooth_sensors[sp_id]=force; current_tooth_total_force+=force
            
            if cell_prop['center'][0] < -0.01: force_right_side += current_tooth_total_force
            elif cell_prop['center'][0] > 0.01: force_left_side += current_tooth_total_force
            else: force_left_side+=current_tooth_total_force/2.0; force_right_side+=current_tooth_total_force/2.0
            
            heatmap_actor = self._create_intra_tooth_heatmap(cell_prop, forces_on_this_tooth_sensors)
            if heatmap_actor:
                if self.selected_tooth_id_grid == tooth_id: heatmap_actor.alpha(1.0)
                else: heatmap_actor.alpha(0.75)
                current_frame_heatmaps_list.append(heatmap_actor)
            
            perc = (current_tooth_total_force/total_force_on_arch_this_step)*100
            text_s = cell_prop['height']*0.20; text_s = max(0.20, min(text_s,0.45)) 
            perc_pos_xy = (cell_prop['center'][0], cell_prop['center'][1]-cell_prop['height']*0.70); perc_z = 0.16 
            num_chars=len(f"{perc:.1f}%"); bg_w_est=text_s*num_chars*0.50; bg_h_est=text_s*1.0
            bg_w_est=max(cell_prop['width']*0.25,bg_w_est); bg_h_est=max(cell_prop['height']*0.15,bg_h_est)
            p1_bg=(perc_pos_xy[0]-bg_w_est/2,perc_pos_xy[1]-bg_h_est/2); p2_bg=(perc_pos_xy[0]+bg_w_est/2,perc_pos_xy[1]+bg_h_est/2)
            perc_text_bg_color_rgb=(0.95,0.95,0.85); perc_text_bg_alpha=0.75
            perc_text_bg = Rectangle(p1_bg,p2_bg,c=perc_text_bg_color_rgb,alpha=perc_text_bg_alpha); perc_text_bg.z(perc_z-0.02) 
            
            # --- CORRECTED APPEND ---
            current_frame_perc_bgs_list.append(perc_text_bg)
            # --- END CORRECTION ---

            perc_label = Text3D(f"{perc:.1f}%",pos=(perc_pos_xy[0],perc_pos_xy[1],perc_z),s=text_s,c='k',justify='cc',depth=0.01) 
            current_frame_perc_texts_list.append(perc_label)
        
        # Store the newly created actors (which are lists) for the next frame's clearing
        self.intra_tooth_heatmap_actors = current_frame_heatmaps_list 
        self.force_percentage_bg_actors = current_frame_perc_bgs_list
        self.force_percentage_actors = current_frame_perc_texts_list

        # L/R Distribution (recreated) & COF (recreated)
        # ... (This part remains the same as the last working version) ...
        perc_left=(force_left_side/total_force_on_arch_this_step)*100; perc_right=(force_right_side/total_force_on_arch_this_step)*100
        actors_to_add_this_frame = [self.time_text_actor] + self.intra_tooth_heatmap_actors + \
                                   self.force_percentage_bg_actors + self.force_percentage_actors # Use the lists

        if self.tooth_cell_definitions:
            # ... (L/R bar creation as before, then add to actors_to_add_this_frame) ...
            min_y_overall=min(p['center'][1]-p['height']/2 for p in self.tooth_cell_definitions.values())
            bar_base_y=min_y_overall-1.8; bar_overall_width=self.arch_layout_width*0.30; bar_max_h=0.8 
            left_bar_h=max(0.02,(perc_left/100.0)*bar_max_h); right_bar_h=max(0.02,(perc_right/100.0)*bar_max_h)
            bar_label_s=0.30; bar_perc_s=0.25
            l_bar_cx=-bar_overall_width*0.75; l_p1=(l_bar_cx-bar_overall_width/2,bar_base_y); l_p2=(l_bar_cx+bar_overall_width/2,bar_base_y+left_bar_h)
            self.left_right_bar_actor_left=Rectangle(l_p1,l_p2,c='g',alpha=0.85)
            left_label_pos=(l_bar_cx,bar_base_y+left_bar_h+0.2,0.1); self.left_bar_label_actor=Text3D("Left",pos=left_label_pos,s=bar_label_s,c='k',justify='cb',depth=0.01)
            left_perc_pos=(l_bar_cx,bar_base_y+left_bar_h/2,0.12); self.left_bar_percentage_actor=Text3D(f"{perc_left:.0f}%",pos=left_perc_pos,s=bar_perc_s,c='w',justify='cc',depth=0.01)
            r_bar_cx=bar_overall_width*0.75; r_p1=(r_bar_cx-bar_overall_width/2,bar_base_y); r_p2=(r_bar_cx+bar_overall_width/2,bar_base_y+right_bar_h)
            self.left_right_bar_actor_right=Rectangle(r_p1,r_p2,c='r',alpha=0.85)
            right_label_pos=(r_bar_cx,bar_base_y+right_bar_h+0.2,0.1); self.right_bar_label_actor=Text3D("Right",pos=right_label_pos,s=bar_label_s,c='k',justify='cb',depth=0.01)
            right_perc_pos=(r_bar_cx,bar_base_y+right_bar_h/2,0.12); self.right_bar_percentage_actor=Text3D(f"{perc_right:.0f}%",pos=right_perc_pos,s=bar_perc_s,c='w',justify='cc',depth=0.01)
            actors_to_add_this_frame.extend([self.left_right_bar_actor_left,self.left_bar_label_actor,self.left_bar_percentage_actor,
                                             self.left_right_bar_actor_right,self.right_bar_label_actor,self.right_bar_percentage_actor])
        
        # COF Rendering
        cof_points_to_draw = self.processor.get_cof_up_to_timestamp(timestamp)
        if len(cof_points_to_draw) > 1:
            cof_line_pts_3d=[(p[0],p[1],0.25) for p in cof_points_to_draw]; self.cof_trajectory_line_actor=Line(cof_line_pts_3d,c=(0.8,0.1,0.8),lw=2,alpha=0.6)
            actors_to_add_this_frame.append(self.cof_trajectory_line_actor)
        if cof_points_to_draw:
            curr_cof_x,curr_cof_y=cof_points_to_draw[-1]; self.cof_current_marker_actor=Sphere(pos=(curr_cof_x,curr_cof_y,0.27),r=0.10,c='darkred',alpha=0.9)
            actors_to_add_this_frame.append(self.cof_current_marker_actor)

        if actors_to_add_this_frame: # Ensure list is not empty before calling add
            self.plotter.add(actors_to_add_this_frame)
        self.plotter.render()
        self.add_frame_to_video() # Call to add frame to video


    # In dental_arch_visualization.py -> _on_mouse_click method

    def _on_mouse_click(self, event):
        clicked_tooth_id_parsed = None; is_background_click = True
        original_selected_tooth_id = self.selected_tooth_id_grid 

        if event.actor:
            is_background_click = False; actor_name = event.actor.name
            logging.info(f"Grid Plotter Clicked: Actor '{actor_name}' at {event.picked3d}")
            if actor_name and (actor_name.startswith("Heatmap_Tooth_") or actor_name.startswith("Outline_Tooth_")):
                try: clicked_tooth_id_parsed = int(actor_name.split("_")[-1])
                except ValueError: logging.warning(f"Could not parse: {actor_name}")
        else: logging.info("Grid Plotter Clicked: Background")

        if clicked_tooth_id_parsed is not None:
            if self.selected_tooth_id_grid == clicked_tooth_id_parsed: self.selected_tooth_id_grid = None 
            else: self.selected_tooth_id_grid = clicked_tooth_id_parsed
            print(f"--- Tooth {self.selected_tooth_id_grid if self.selected_tooth_id_grid else 'deselected'} ---")
        elif is_background_click: self.selected_tooth_id_grid = None # Clicked non-tooth actor
        else: self.selected_tooth_id_grid = None # Clicked background
            
        # Update Graph Link: Pass selected_tooth_id_grid (can be None)
        if self.animation_controller_ref:
            self.animation_controller_ref.update_graph_for_selected_tooth(self.selected_tooth_id_grid) # Pass current selection
        
        # Update Detailed Info Text (as before)
        if self.selected_tooth_info_text_actor: self.plotter.remove(self.selected_tooth_info_text_actor); self.selected_tooth_info_text_actor = None
        if self.selected_tooth_id_grid is not None:
            timestamp_for_info = self.last_animated_timestamp if self.last_animated_timestamp is not None else (self.timestamps[0] if self.timestamps else 0)
            info_text_lines = [f"Tooth ID: {self.selected_tooth_id_grid}", f"Forces @ {timestamp_for_info:.1f}s:"]
            total_force_on_selected_tooth = 0.0
            sensor_ids = [spid for tid,spid in self.processor.ordered_tooth_sensor_pairs if tid==self.selected_tooth_id_grid]
            _p, forces_ts = self.processor.get_all_forces_at_time(timestamp_for_info); map_ts = {p:f for p,f in zip(_p,forces_ts)}
            for sp_id in sensor_ids: force = map_ts.get((self.selected_tooth_id_grid,sp_id),0.0); info_text_lines.append(f" S{sp_id}:{force:.1f}N"); total_force_on_selected_tooth+=force
            info_text_lines.append(f"Total: {total_force_on_selected_tooth:.1f}N"); info_text_full = "\n".join(info_text_lines)
            info_text_bg_rgb = (0.95, 0.93, 0.8); info_text_bg_alpha = 0.85
            self.selected_tooth_info_text_actor = Text2D(
                info_text_full, 
                pos="top-right", # Use full string "top-right"
                c='black', 
                bg=info_text_bg_rgb,
                alpha=info_text_bg_alpha, 
                s=0.70, 
                font="Calco" 
            )
            self.plotter.add(self.selected_tooth_info_text_actor)
        
        is_timer_active = any(cb.get('event')=='timer' for cb in (self.plotter.callbacks if hasattr(self.plotter,'callbacks') else []))
        if not is_timer_active and self.timestamps:
            current_t = self.last_animated_timestamp if self.last_animated_timestamp is not None else (self.timestamps[0] if self.timestamps else 0.0)
            if current_t is not None: self.render_arch(current_t)
            #render_arch will apply highlight to re-render for highlight.")

    def animate(self, event=None): 
        # self.is_animation_running = True # Set flag when animate is called by timer
        # ... (rest of animate method as before) ...
        if not self.timestamps: 
            try: self.plotter.timer_callback('destroy')
            except Exception: pass
            # self.is_animation_running = False
            return
        if self.current_timestamp_idx < len(self.timestamps):
            t = self.timestamps[self.current_timestamp_idx]
            self.last_animated_timestamp = t; self.render_arch(t); self.current_timestamp_idx += 1
        else:
            self.current_timestamp_idx = 0 
            if self.timestamps: 
                t = self.timestamps[self.current_timestamp_idx]
                self.last_animated_timestamp = t; self.render_arch(t); self.current_timestamp_idx += 1
            # else: # Should not happen if timestamps list is checked at start
                # self.is_animation_running = False


    def start_animation(self, dt=100):
        # ... (start_animation method as before) ...
        if not self.timestamps: logging.error("No timestamps."); self.plotter.add(Text2D("Error: No data.", c='r')).render(); return
        self.animate_callback_ref = self.animate 
        self.plotter.timer_callback('destroy'); self.plotter.remove_callback('timer') 
        self.plotter.add_callback('timer', self.animate_callback_ref); self.plotter.timer_callback('create', dt=dt)


if __name__ == '__main__':
    # ... (same as before) ...
    from data_acquisition import SensorDataReader
    from data_processing import DataProcessor
    reader = SensorDataReader()
    data = reader.simulate_data(duration=3, num_teeth=16, num_sensor_points_per_tooth=4)
    processor = DataProcessor(data.copy()) 
    print("\n--- TESTING 2D T-SCAN STYLE GRID WITH HEATMAPS & ENHANCED TEXT & L/R DISTRIBUTION ---")
    visualizer_tscan = DentalArchVisualizer(processor) 
    if processor.timestamps: 
        visualizer_tscan.start_animation(dt=200)
        visualizer_tscan.plotter.show(title="T-Scan Style Grid - L/R Distribution", interactive=True)
    else: print("Skipping T-Scan style test: Timestamps not available from processor.")
# --- END OF FILE dental_arch_visualization.py ---
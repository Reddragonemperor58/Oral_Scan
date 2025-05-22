# --- START OF FILE dental_arch_visualization.py ---
import numpy as np
from vedo import Plotter, Text2D, Line, Rectangle, Text3D, Points, Grid, Disc 
import logging
# import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DentalArchVisualizer:
    # ... (init and other methods up to _create_intra_tooth_heatmap remain the same) ...
    def __init__(self, processor):
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

        self.grid_outline_actors = []
        self.intra_tooth_heatmap_actors = []
        self.tooth_label_actors = []
        self.force_percentage_actors = []
        self.force_percentage_bg_actors = [] 
        self.time_text_actor = None
        
        self._initialize_static_grid_elements()

    def _initialize_empty_state(self):
        self.tooth_cell_definitions = {}; self.plotter = Plotter(bg='lightgrey'); self.timestamps = []
        self.current_timestamp_idx = 0; self.last_animated_timestamp = None; self.intra_tooth_heatmap_actors = []
        self.tooth_label_actors = []; self.force_percentage_actors = []; self.time_text_actor = None; 
        self.grid_outline_actors = []; self.force_percentage_bg_actors = []


    def _fit_camera_to_grid(self):
        if not self.tooth_cell_definitions: self.plotter.camera.SetParallelScale(10); return
        all_x = [d['center'][0] + d['width']/2 for d in self.tooth_cell_definitions.values()] + \
                [d['center'][0] - d['width']/2 for d in self.tooth_cell_definitions.values()]
        all_y = [d['center'][1] + d['height']/2 for d in self.tooth_cell_definitions.values()] + \
                [d['center'][1] - d['height']/2 for d in self.tooth_cell_definitions.values()]
        if not all_x or not all_y: self.plotter.camera.SetParallelScale(10); return

        min_x, max_x = min(all_x), max(all_x); min_y, max_y = min(all_y), max(all_y)
        padding_factor = 1.20 
        view_h = (max_y - min_y) * padding_factor; view_w = (max_x - min_x) * padding_factor
        scale = max(view_h, view_w, 1.0) / 2.0 
        self.plotter.camera.SetParallelScale(scale)
        fx, fy = (min_x+max_x)/2, (min_y+max_y)/2
        self.plotter.camera.SetFocalPoint(fx, fy, 0); self.plotter.camera.SetPosition(fx, fy, 10)

    def _get_arch_positions_for_layout(self, num_teeth, arch_width, arch_depth):
        if num_teeth == 0: return np.array([])
        if num_teeth == 1: x_coords = np.array([0.0])
        else: x_coords = np.linspace(-arch_width / 2, arch_width / 2, num_teeth)
        k = arch_depth / ((arch_width / 2)**2) if arch_width != 0 else 0
        positions = []
        for x_coord in x_coords:
            y_val = arch_depth - k * (x_coord**2)
            positions.append(np.array([x_coord, y_val, 0]))
        return np.array(positions)


    def _define_explicit_tscan_layout(self, num_teeth_from_data):
        layout = {} 
        if num_teeth_from_data == 0: return layout
        arch_centers_3d = self._get_arch_positions_for_layout(num_teeth_from_data, 
                                                              self.arch_layout_width * 0.9,
                                                              self.arch_layout_depth * 0.8)
        arch_centers_xy = [ac[:2] for ac in arch_centers_3d]
        if num_teeth_from_data > 1:
            point_distances = np.linalg.norm(np.diff(arch_centers_xy, axis=0), axis=1)
            avg_spacing = np.mean(point_distances) if len(point_distances) > 0 else self.arch_layout_width / num_teeth_from_data
            avg_cell_w = avg_spacing * 0.85 
            avg_cell_h = avg_spacing * 1.0
        else: 
            avg_cell_w = self.arch_layout_width * 0.3
            avg_cell_h = self.arch_layout_depth * 0.3
        avg_cell_w = max(0.5, avg_cell_w); avg_cell_h = max(0.5, avg_cell_h)

        for i in range(num_teeth_from_data):
            actual_id = self.processor.tooth_ids[i]
            center_xy = arch_centers_xy[i]
            width = avg_cell_w; height = avg_cell_h
            norm_x = abs(center_xy[0]) / (self.arch_layout_width * 0.9 / 2) if (self.arch_layout_width * 0.9 / 2) != 0 else 0
            if norm_x > 0.7: width *= 1.35; height *= 0.85;
            elif norm_x > 0.45: width *= 1.1; height *= 0.95;
            elif norm_x < 0.15: width *= 0.65; height *= 1.25;
            elif norm_x < 0.35: width *= 0.80; height *= 1.15;
            width = max(0.5, min(width, avg_cell_w * 1.8)) 
            height = max(0.5, min(height, avg_cell_h * 1.8))
            layout[i] = {'center': center_xy, 'width': width, 'height': height, 'actual_id': actual_id}
        return layout


    def _initialize_static_grid_elements(self):
        if not self.tooth_cell_definitions: return
        for _layout_idx, cell_prop in self.tooth_cell_definitions.items():
            cx, cy = cell_prop['center']; w, h = cell_prop['width'], cell_prop['height']
            p1, p2 = (cx - w/2, cy - h/2), (cx + w/2, cy + h/2)
            outline = Rectangle(p1, p2, c=(0.4,0.4,0.4), alpha=0.7); outline.lw(1.5) 
            self.grid_outline_actors.append(outline)
            label_pos_xy = (cx, cy + h * 0.60) 
            text_s = h * 0.30; text_s = max(0.25, min(text_s, 0.55)) 
            label = Text3D(str(cell_prop['actual_id']), pos=(label_pos_xy[0], label_pos_xy[1], 0.12), 
                           s=text_s, c=(0.1,0.1,0.35), justify='center-center', depth=0.01) 
            self.tooth_label_actors.append(label)
        self.plotter.add(self.grid_outline_actors, self.tooth_label_actors)

    def _create_intra_tooth_heatmap(self, cell_prop, forces_on_this_tooth_sensors):
        if not forces_on_this_tooth_sensors: return None
        num_sp = len(forces_on_this_tooth_sensors)
        if num_sp == 0: return None
        
        cx, cy = cell_prop['center']; cw, ch = cell_prop['width'], cell_prop['height']
        heatmap_grid_resolution_param = (1, 1) 
        heatmap_grid = Grid(s=[cw * 0.98, ch * 0.98], res=heatmap_grid_resolution_param)
        heatmap_grid.pos(cx, cy, 0.05) 
        grid_points_forces = np.zeros(heatmap_grid.npoints) 

        if num_sp == 4:
            grid_points_forces[2] = forces_on_this_tooth_sensors.get(1, 0) 
            grid_points_forces[3] = forces_on_this_tooth_sensors.get(2, 0) 
            grid_points_forces[0] = forces_on_this_tooth_sensors.get(3, 0) 
            grid_points_forces[1] = forces_on_this_tooth_sensors.get(4, 0) 
        elif num_sp > 0:
            avg_force = np.mean(list(forces_on_this_tooth_sensors.values()))
            grid_points_forces.fill(avg_force)
        
        heatmap_grid.pointdata["forces"] = np.nan_to_num(grid_points_forces)
        
        # --- CORRECTED custom_cmap: Use 3-component RGB ---
        custom_cmap_rgb = ['darkblue', (0,0,1) , (0,1,0), (1,1,0), (1,0,0)] 
        heatmap_grid.cmap(custom_cmap_rgb, "forces", vmin=0, vmax=self.max_force_for_scaling)
        # --- End CORRECTION ---

        heatmap_grid.lw(0)
        heatmap_grid.alpha(0.9) # Set alpha for the whole heatmap grid actor if desired
        return heatmap_grid


    def render_arch(self, timestamp):
        # ... (Text2D for time is fine) ...
        if not self.tooth_cell_definitions: self.plotter.render(); return

        if self.time_text_actor: self.plotter.remove(self.time_text_actor)
        self.time_text_actor = Text2D(
            f"Time: {timestamp:.1f}s", pos="bottom-left", c='black', 
            bg=(1,1,1), alpha=0.7, s=0.7 
        )
        self.plotter.add(self.time_text_actor)
        
        # ... (Clearing actors as before) ...
        if self.intra_tooth_heatmap_actors: self.plotter.remove(self.intra_tooth_heatmap_actors); self.intra_tooth_heatmap_actors.clear()
        if self.force_percentage_actors: self.plotter.remove(self.force_percentage_actors); self.force_percentage_actors.clear()
        if self.force_percentage_bg_actors: self.plotter.remove(self.force_percentage_bg_actors); self.force_percentage_bg_actors.clear()

        ordered_pairs, forces_all_sensor_points = self.processor.get_all_forces_at_time(timestamp)
        if not ordered_pairs: self.plotter.render(); return
        
        force_map_all_sensors = {pair: force for pair, force in zip(ordered_pairs, forces_all_sensor_points)}
        arch_total_force = sum(f for f in forces_all_sensor_points if np.isfinite(f) and f > 0)
        if arch_total_force < 1e-6 : arch_total_force = 1.0 

        for _layout_idx, cell_prop in self.tooth_cell_definitions.items():
            tooth_id = cell_prop['actual_id']
            forces_on_this_tooth_sensors = {}
            current_tooth_total_force = 0
            sensor_ids_for_this_tooth = [spid for tid, spid in self.processor.ordered_tooth_sensor_pairs if tid == tooth_id]

            for sp_id in sensor_ids_for_this_tooth:
                force = force_map_all_sensors.get((tooth_id, sp_id), 0.0)
                force = np.nan_to_num(force)
                forces_on_this_tooth_sensors[sp_id] = force
                current_tooth_total_force += force
            
            heatmap_actor = self._create_intra_tooth_heatmap(cell_prop, forces_on_this_tooth_sensors)
            if heatmap_actor: self.intra_tooth_heatmap_actors.append(heatmap_actor)
            
            perc = (current_tooth_total_force / arch_total_force) * 100
            text_s = cell_prop['height'] * 0.18 
            text_s = max(0.18, min(text_s, 0.45)) 
            perc_pos_xy = (cell_prop['center'][0], cell_prop['center'][1] - cell_prop['height'] * 0.70)
            perc_z = 0.16 

            bg_radius = text_s * 1.3 
            bg_radius = max(cell_prop['width']*0.1, bg_radius) 
            perc_text_bg = Disc(pos=(perc_pos_xy[0], perc_pos_xy[1], perc_z - 0.01), 
                                r1=0, r2=bg_radius, c='white', alpha=0.65)
            self.force_percentage_bg_actors.append(perc_text_bg)

            perc_label = Text3D(f"{perc:.1f}%", pos=(perc_pos_xy[0], perc_pos_xy[1], perc_z), 
                                s=text_s, c='black', justify='center-center', depth=0.01) 
            self.force_percentage_actors.append(perc_label)

        if self.intra_tooth_heatmap_actors: self.plotter.add(self.intra_tooth_heatmap_actors)
        if self.force_percentage_bg_actors: self.plotter.add(self.force_percentage_bg_actors)
        if self.force_percentage_actors: self.plotter.add(self.force_percentage_actors)
        self.plotter.render()
    
    # ... (animate and start_animation methods remain the same) ...
    def animate(self, event=None): 
        if not self.timestamps: 
            try: self.plotter.timer_callback('destroy')
            except Exception: pass; return
        if self.current_timestamp_idx < len(self.timestamps):
            t = self.timestamps[self.current_timestamp_idx]
            self.last_animated_timestamp = t; self.render_arch(t); self.current_timestamp_idx += 1
        else:
            self.current_timestamp_idx = 0 
            if self.timestamps: t = self.timestamps[self.current_timestamp_idx]; self.last_animated_timestamp = t; self.render_arch(t); self.current_timestamp_idx += 1

    def start_animation(self, dt=100):
        if not self.timestamps: logging.error("No timestamps."); self.plotter.add(Text2D("Error: No data.", c='r')).render(); return
        self.animate_callback_ref = self.animate 
        self.plotter.timer_callback('destroy'); self.plotter.remove_callback('timer') 
        self.plotter.add_callback('timer', self.animate_callback_ref); self.plotter.timer_callback('create', dt=dt)


if __name__ == '__main__':
    from data_acquisition import SensorDataReader
    from data_processing import DataProcessor
    
    reader = SensorDataReader()
    data = reader.simulate_data(duration=3, num_teeth=16, num_sensor_points_per_tooth=4)
    processor = DataProcessor(data.copy()) 
    
    print("\n--- TESTING 2D T-SCAN STYLE GRID WITH HEATMAPS & ENHANCED TEXT ---")
    visualizer_tscan = DentalArchVisualizer(processor) 
    if processor.timestamps: 
        visualizer_tscan.start_animation(dt=200)
        visualizer_tscan.plotter.show(title="T-Scan Style Grid - Enhanced Text", interactive=True)
    else:
        print("Skipping T-Scan style test: Timestamps not available from processor.")
# --- END OF FILE dental_arch_visualization.py ---
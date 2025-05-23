# --- START OF FILE dental_arch_3d_bar_visualization.py ---
import numpy as np
# --- CORRECTED IMPORT: Ensure Text3D is imported ---
from vedo import Plotter, Text2D, Cylinder, Box, Line, Axes, Grid, Plane, Text3D 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DentalArch3DBarVisualizer:
    def __init__(self, processor):
        self.processor = processor
        if self.processor.cleaned_data is None:
            self.processor.create_force_matrix()

        self.num_data_teeth = len(self.processor.tooth_ids) if self.processor.tooth_ids else 0
        if self.num_data_teeth == 0:
            logging.error("No tooth data for 3D Bar Visualizer."); self._initialize_empty_state(); return

        self.arch_layout_width = 14.0
        self.arch_layout_depth = 8.0
        self.bar_base_radius = 0.5     
        
        self.tooth_bar_base_positions = self._create_bar_base_positions(
            self.num_data_teeth, self.arch_layout_width, self.arch_layout_depth
        )

        self.max_force_for_scaling = self.processor.max_force_overall if hasattr(self.processor, 'max_force_overall') else 100.0
        self.max_bar_height = 5.0 
        self.min_bar_height = 0.1 

        self.plotter = Plotter(bg=(0.8, 0.8, 0.85), axes=0, title="3D Force Bar Chart") # Initially no axes
        
        floor_size_x = self.arch_layout_width * 1.5
        floor_size_y = self.arch_layout_depth * 1.8
        floor_grid_resolution = (10, 10) 
        floor_grid = Grid(s=(floor_size_x, floor_size_y), res=floor_grid_resolution, c='gainsboro', alpha=0.4)
        self.grid_center_y = -self.arch_layout_depth * 0.3 # Store for camera
        floor_grid.pos(0, self.grid_center_y, -0.05) 
        self.plotter.add(floor_grid) # Add floor first to establish some bounds

          # --- Add global axes AFTER some content is present ---
        if self.plotter.actors: # Check if there's anything to base axes on
            self.axes_actor = Axes(self.plotter.actors, c='dimgrey', yzgrid=False) # Example: Axes based on current actors
            self.plotter.add(self.axes_actor)
        else: # Fallback if still no actors (should not happen if floor_grid is added)
             self.plotter.add_global_axes(axtype=1, c='dimgrey')
        # --- End Axes modification ---

        # --- Defer adding global axes or use specific ranges if floor isn't enough ---
        # self.axes_actor = self.plotter.add_global_axes(axtype=1, c='dimgrey') 
        # If the above still gives issues, build Axes manually after adding content:
        # Later in init, after _initialize_static_elements perhaps, or provide explicit ranges.

        self.plotter.camera.SetPosition(0, -self.arch_layout_depth * 2.5, self.max_bar_height * 2.2) 
        self.plotter.camera.SetFocalPoint(0, self.grid_center_y, self.max_bar_height / 3) 
        self.plotter.camera.SetViewUp(0, 0.3, 0.7) 

        self.timestamps = self.processor.timestamps
        self.current_timestamp_idx = 0
        self.last_animated_timestamp = None 

        self.force_bar_actors = []
        self.time_text_actor = None
        self.arch_base_line_actor = None 
        self.tooth_label_actors = [] # Initialize this list

        self._initialize_static_elements() # This adds arch line and labels
        self.plotter.add_callback('mouse click', self._on_mouse_click) # Add click callback

      

    def _initialize_empty_state(self):
        self.tooth_bar_base_positions = []
        self.plotter = Plotter(bg='lightgrey', axes=0, title="3D Force Bar Chart (No Data)")
        self.timestamps = []
        self.current_timestamp_idx = 0; self.last_animated_timestamp = None
        self.force_bar_actors = []; self.time_text_actor = None; self.arch_base_line_actor = None
        self.tooth_label_actors = []


    def _create_bar_base_positions(self, num_teeth, total_width, total_depth):
        if num_teeth == 0: return []
        positions_3d = []
        if num_teeth == 1: x_coords = np.array([0.0])
        else: x_coords = np.linspace(-total_width / 2, total_width / 2, num_teeth)
        
        k = total_depth / ((total_width / 2)**2) if total_width != 0 else 0
        for x_coord in x_coords:
            y_val_parabola = total_depth - k * (x_coord**2) 
            adjusted_y = y_val_parabola - total_depth * 0.8 # Shift Y to better match visual
            positions_3d.append(np.array([x_coord, adjusted_y, 0.0])) 
        return positions_3d


    def _initialize_static_elements(self):
        self.tooth_label_actors = [] # Ensure it's initialized here too
        if len(self.tooth_bar_base_positions) > 1:
            line_z_offset = 0.01 
            line_points = [(p[0], p[1], line_z_offset) for p in self.tooth_bar_base_positions]
            self.arch_base_line_actor = Line(line_points, c='dimgray', lw=3, alpha=0.8)
            self.plotter.add(self.arch_base_line_actor)
            
            for i, pos in enumerate(self.tooth_bar_base_positions):
                if i < len(self.processor.tooth_ids):
                    tooth_id = self.processor.tooth_ids[i]
                    label_pos = (pos[0], pos[1] + self.bar_base_radius * 0.5, -0.2) # Adjusted label pos
                    label = Text3D(str(tooth_id), pos=label_pos, s=0.22, c=(0.1,0.1,0.1), depth=0.01, justify='center-top')
                    self.tooth_label_actors.append(label) # Add to list
            if self.tooth_label_actors: # Add all labels at once
                self.plotter.add(self.tooth_label_actors)


    def render_display(self, timestamp):
        # ... (Time text, clearing actors as before) ...
        if not self.tooth_bar_base_positions: self.plotter.render(); return
        if self.time_text_actor: self.plotter.remove(self.time_text_actor)
        time_text_bg_rgb = (1,1,1); time_text_bg_alpha = 0.7
        self.time_text_actor = Text2D(f"Time: {timestamp:.1f}s", pos="bottom-right", c='black', bg=time_text_bg_rgb, alpha=time_text_bg_alpha, s=0.8)
        self.plotter.add(self.time_text_actor)
        if self.force_bar_actors: self.plotter.remove(self.force_bar_actors); self.force_bar_actors.clear()
        
        for i, base_pos in enumerate(self.tooth_bar_base_positions):
            if i >= len(self.processor.tooth_ids): continue
            tooth_id = self.processor.tooth_ids[i]
            _ , force_value_avg_series = self.processor.get_average_force_for_tooth(tooth_id)
            current_force = 0.0
            if self.timestamps and len(force_value_avg_series) == len(self.timestamps):
                try: time_idx = np.argmin(np.abs(np.array(self.timestamps) - timestamp)); current_force = force_value_avg_series[time_idx]
                except IndexError: logging.debug(f"TS index issue tooth {tooth_id} time {timestamp}")
                except Exception: logging.debug(f"Force fetch issue tooth {tooth_id} time {timestamp}")
            if not np.isfinite(current_force): current_force = 0.0
            norm_force = min(1.0, max(0.0, current_force / self.max_force_for_scaling))
            bar_height = self.min_bar_height + norm_force * (self.max_bar_height - self.min_bar_height)
            if bar_height < self.min_bar_height and current_force > 1e-3 : bar_height = self.min_bar_height 
            elif current_force < 1e-3: bar_height = 0.0 

            if bar_height > 1e-4: 
                if norm_force < 0.01: color_rgb = (0.1, 0.1, 0.6)    
                elif norm_force < 0.25: color_rgb = (0.2, 0.4, 1)  
                elif norm_force < 0.5:  color_rgb = (0.1, 0.8, 0.4) 
                elif norm_force < 0.75: color_rgb = (1, 0.9, 0.1)    
                elif norm_force < 0.9:  color_rgb = (1, 0.4, 0)  
                else: color_rgb = (0.9, 0.0, 0.2)                    
                bar_center_z = base_pos[2] + bar_height / 2.0
                bar_actor = Box(pos=(base_pos[0], base_pos[1], bar_center_z),
                                length=self.bar_base_radius * 1.7, width=self.bar_base_radius * 1.7,  
                                height=bar_height, c=color_rgb, alpha=0.92)
                bar_actor.name = f"Bar_Tooth_{tooth_id}" # Name for picking
                bar_actor.pickable = True
                self.force_bar_actors.append(bar_actor)
        if self.force_bar_actors: self.plotter.add(self.force_bar_actors)
        self.plotter.render()

    def _on_mouse_click(self, event): # Add this method to DentalArch3DBarVisualizer
        if not event.actor: return
        actor_name = event.actor.name
        logging.info(f"3D Bar Plotter Clicked: {actor_name} at {event.picked3d}")
        if actor_name and actor_name.startswith("Bar_Tooth_"):
            try:
                tooth_id_str = actor_name.split("_")[-1]
                print(f"--- 3D Bar View: Clicked on Bar for Tooth ID: {tooth_id_str} ---")
                # Add highlight or other interaction logic here
                # event.actor.color('yellow').render() # Example: highlight
            except Exception as e:
                logging.warning(f"Could not parse tooth_id from {actor_name}: {e}")
        # else:
            # print(f"--- 3D Bar View: Clicked on {actor_name if actor_name else 'background/other'} ---")

    def animate(self, event=None): 
        if not self.timestamps: 
            try: self.plotter.timer_callback('destroy')
            except Exception: pass; return
        if self.current_timestamp_idx < len(self.timestamps):
            t = self.timestamps[self.current_timestamp_idx]
            self.last_animated_timestamp = t; self.render_display(t); self.current_timestamp_idx += 1
        else:
            self.current_timestamp_idx = 0 
            if self.timestamps: t = self.timestamps[self.current_timestamp_idx]; self.last_animated_timestamp = t; self.render_display(t); self.current_timestamp_idx += 1

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
    
    print("\n--- TESTING 3D BAR CHART VISUALIZATION ---")
    visualizer_3d_bar = DentalArch3DBarVisualizer(processor) 
    if processor.timestamps: 
        visualizer_3d_bar.start_animation(dt=150)
        visualizer_3d_bar.plotter.show(title="3D Force Bar Chart Animation", interactive=True)
    else:
        print("Skipping 3D Bar Chart test: Timestamps not available from processor.")
# --- END OF FILE dental_arch_3d_bar_visualization.py ---
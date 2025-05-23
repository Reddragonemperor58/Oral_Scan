# --- START OF FILE animation.py ---
import matplotlib
try:
    matplotlib.use('Qt5Agg') 
    print("INFO: Using Matplotlib backend: Qt5Agg")
except ImportError:
    print("WARNING: PyQt5 not found, Matplotlib will use its default backend (likely TkAgg).")
    pass 

import logging
import matplotlib.pyplot as plt 
import numpy as np # Ensure numpy is imported if used for np.argmin

from data_acquisition import SensorDataReader
from data_processing import DataProcessor
from graph_visualization import GraphVisualizer
# Assuming DentalArchVisualizer is the grid one, DentalArch3DBarVisualizer is the bar one
from dental_arch_visualization import DentalArchVisualizer as DentalArchGridVisualizer 
from dental_arch_3d_bar_visualization import DentalArch3DBarVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnimationController:
    def __init__(self, graph_visualizer, arch_grid_visualizer, arch_3d_bar_visualizer, fps=10):
        self.graph_visualizer = graph_visualizer
        self.arch_grid_visualizer = arch_grid_visualizer
        self.arch_3d_bar_visualizer = arch_3d_bar_visualizer
        self.fps = fps
        self.graph_time_indicator = None
        self.currently_graphed_tooth_ids = [] 
        self.initial_graph_teeth = [] 

    def update_graph_for_selected_tooth(self, selected_tooth_id=None):
        if not self.graph_visualizer or self.graph_visualizer.ax is None:
            logging.warning("Graph visualizer or its axes not ready for update.")
            return
        if selected_tooth_id is not None:
            self.currently_graphed_tooth_ids = [selected_tooth_id]
        else: 
            self.currently_graphed_tooth_ids = self.initial_graph_teeth
        logging.info(f"Graph Link: Plotting teeth: {self.currently_graphed_tooth_ids}")
        self.graph_visualizer.plot_tooth_lines(self.currently_graphed_tooth_ids)

    def start_animation_integrated(self, initial_graph_tooth_ids_param): 
        self.initial_graph_teeth = list(initial_graph_tooth_ids_param) 
        self.currently_graphed_tooth_ids = list(initial_graph_tooth_ids_param)
        logging.info("Starting integrated animation...")

        if self.graph_visualizer.figure is None: 
            if self.currently_graphed_tooth_ids:
                self.graph_visualizer.create_graph_figure() 
                self.graph_visualizer.plot_tooth_lines(self.currently_graphed_tooth_ids)
        if self.graph_visualizer.figure:
            plt.ion(); self.graph_visualizer.figure.show()
            self.graph_visualizer.figure.canvas.flush_events(); self.graph_visualizer.figure.canvas.draw_idle()

        if hasattr(self.arch_grid_visualizer, 'set_animation_controller_for_graph_link'):
            self.arch_grid_visualizer.set_animation_controller_for_graph_link(self)

        # --- Start Video Recording for the Grid Visualizer ---
        if hasattr(self.arch_grid_visualizer, 'start_recording'):
            # You can customize the filename in DentalArchGridVisualizer's init or pass here
            self.arch_grid_visualizer.start_recording(fps=self.fps) 
        else:
            logging.warning("arch_grid_visualizer does not support start_recording(). Video will not be saved.")
        # --- End Video Recording Start ---

        def integrated_animation_step(event=None):
            self.arch_3d_bar_visualizer.animate(event) 
            current_timestamp = self.arch_3d_bar_visualizer.last_animated_timestamp
            
            if current_timestamp is not None:
                if self.arch_grid_visualizer.timestamps:
                    try: 
                        if len(self.arch_grid_visualizer.timestamps) > 0:
                            grid_time_idx = np.argmin(np.abs(np.array(self.arch_grid_visualizer.timestamps) - current_timestamp))
                            self.arch_grid_visualizer.current_timestamp_idx = grid_time_idx
                    except Exception as e: logging.error(f"Error syncing grid viz ts: {e}")
                self.arch_grid_visualizer.animate() # Calls render_arch -> add_frame_to_video

                if self.graph_visualizer.figure and self.graph_visualizer.ax: 
                    self.graph_visualizer.update_graph_to_timestamp(current_timestamp, self.currently_graphed_tooth_ids)
                    if self.graph_time_indicator:
                        try: self.graph_visualizer.ax.lines.remove(self.graph_time_indicator)
                        except (ValueError, AttributeError): pass 
                        self.graph_time_indicator = None 
                    self.graph_time_indicator = self.graph_visualizer.ax.axvline(current_timestamp, color='red', linestyle='--', lw=0.8, gid="time_indicator")
                    self.graph_visualizer.figure.canvas.draw_idle() 
            logging.debug(f"Integrated frame: Time {current_timestamp:.1f}s" if current_timestamp is not None else "Integrated frame: current_timestamp is None")
        
        driver_plotter = self.arch_3d_bar_visualizer.plotter
        self.arch_3d_bar_visualizer.animate_callback_ref = integrated_animation_step 
        driver_plotter.timer_callback('destroy'); driver_plotter.remove_callback('timer') 
        driver_plotter.add_callback('timer', self.arch_3d_bar_visualizer.animate_callback_ref)
        dt = int(1000 / self.fps); driver_plotter.timer_callback('create', dt=dt)
        logging.info(f"Main animation timer started with dt={dt}ms.")

        if self.arch_grid_visualizer.timestamps: # Initial render of grid view
            self.arch_grid_visualizer.render_arch(self.arch_grid_visualizer.timestamps[0])
        
        logging.info("Starting main Vedo event loop (3D Bar Chart driving)...")
        self.arch_3d_bar_visualizer.plotter.show(title="3D Force Bars (+ 2D Grid & Graph)")
        
        # --- Stop Video Recording ---
        if hasattr(self.arch_grid_visualizer, 'stop_recording'):
            self.arch_grid_visualizer.stop_recording()
        # --- End Video Recording Stop ---

        if self.graph_visualizer.figure: plt.ioff()
        logging.info("Animation stopped and Vedo window closed.")

    def stop_animation(self):
        if self.arch_3d_bar_visualizer and self.arch_3d_bar_visualizer.plotter:
            self.arch_3d_bar_visualizer.plotter.timer_callback('destroy')
        # Also stop recording if explicitly stopped
        if hasattr(self.arch_grid_visualizer, 'is_recording') and self.arch_grid_visualizer.is_recording:
            if hasattr(self.arch_grid_visualizer, 'stop_recording'):
                self.arch_grid_visualizer.stop_recording()
        logging.info("Animation stopping sequence initiated.")


if __name__ == '__main__':
    reader = SensorDataReader()
    data = reader.simulate_data(duration=5, num_teeth=16, num_sensor_points_per_tooth=4) 
    processor = DataProcessor(data.copy())
    if processor.cleaned_data is None: processor.create_force_matrix()

    if not processor.timestamps:
        logging.error("Data processing failed. Cannot animate.")
    else:
        graph_viz = GraphVisualizer(processor)
        initial_teeth_for_graph = [processor.tooth_ids[0], processor.tooth_ids[1]] if len(processor.tooth_ids) >= 2 else processor.tooth_ids[:1]
        if not initial_teeth_for_graph: logging.warning("No teeth for initial graph.")
        
        # Pass video_filename to the grid visualizer instance if you want to customize it
        arch_grid_viz = DentalArchGridVisualizer(processor, video_filename="tscan_grid_output.mp4") 
        arch_3d_bar_viz = DentalArch3DBarVisualizer(processor) 
        
        animator = AnimationController(graph_viz, arch_grid_viz, arch_3d_bar_viz, fps=10)
        animator.start_animation_integrated(initial_teeth_for_graph)
# --- END OF FILE animation.py ---
# --- START OF FILE graph_visualization.py ---
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GraphVisualizer:
    def __init__(self, processor):
        self.processor = processor
        self.figure = None
        self.ax = None
        self.lines = {} 
        self.full_data_cache = {} 

    def create_graph(self, tooth_ids_to_display, figsize=(10, 4)):
        if not tooth_ids_to_display:
            logging.warning("No tooth IDs provided for graph creation.")
            return None
            
        self.figure, self.ax = plt.subplots(figsize=figsize)
        num_lines = len(tooth_ids_to_display)
        colors = plt.cm.viridis(np.linspace(0, 1, num_lines))

        if self.processor.timestamps and len(self.processor.timestamps) > 0:
            self.ax.set_xlim(self.processor.timestamps[0], self.processor.timestamps[-1])
        else: 
            self.ax.set_xlim(0, 1) 
            
        max_y_force = 100.0 # Default
        # Ensure force_matrix is float before calling np.isnan or np.max
        fm_for_ylim = self.processor.force_matrix
        if fm_for_ylim is not None and fm_for_ylim.size > 0:
            if fm_for_ylim.dtype != float and fm_for_ylim.dtype != np.float64 and fm_for_ylim.dtype != np.float32:
                try:
                    fm_for_ylim = fm_for_ylim.astype(float)
                    logging.debug(f"GraphVisualizer: Cast force_matrix to float for Y lim calculation.")
                except ValueError:
                    logging.warning("GraphVisualizer: Could not cast force_matrix to float for Y lim. Using default.")
                    fm_for_ylim = np.array([[max_y_force]], dtype=float) # Fallback

            valid_forces = fm_for_ylim[~np.isnan(fm_for_ylim)] # Now this should work
            if valid_forces.size > 0:
                max_y_force = np.max(valid_forces)
            elif hasattr(self.processor, 'max_force_overall'): # Use processor's pre-calculated max
                 max_y_force = self.processor.max_force_overall


        self.ax.set_ylim(bottom=-5, top=max_y_force * 1.1 if max_y_force > 0 else 10) # Allow seeing 0 line better


        for i, tooth_id in enumerate(tooth_ids_to_display):
            full_times, full_forces = self.processor.get_average_force_for_tooth(tooth_id) # This should return float array
            self.full_data_cache[tooth_id] = (full_times, full_forces)
            
            line, = self.ax.plot([], [], label=f"Tooth {tooth_id} (Avg)", color=colors[i % len(colors)])
            self.lines[tooth_id] = line
        
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Average Force (N)")
        self.ax.set_title("Average Bite Force Over Time (Selected Teeth)")
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        
        logging.info("Graph created for average forces of teeth: %s (initially empty)", tooth_ids_to_display)
        self.figure.tight_layout()
        return self.figure

    def update_graph_to_timestamp(self, current_timestamp, tooth_ids_to_display):
        # ... (rest of this method is likely fine if full_data_cache contains float arrays) ...
        if self.figure is None or self.ax is None: return

        for tooth_id in tooth_ids_to_display:
            if tooth_id in self.lines and tooth_id in self.full_data_cache:
                full_times, full_forces = self.full_data_cache[tooth_id]
                if full_times is not None and len(full_times) > 0:
                    idx_up_to_time = np.searchsorted(full_times, current_timestamp, side='right')
                    times_to_plot = full_times[:idx_up_to_time]
                    forces_to_plot = full_forces[:idx_up_to_time]
                    self.lines[tooth_id].set_data(times_to_plot, forces_to_plot)
                else:
                    self.lines[tooth_id].set_data([], []) 
        self.figure.canvas.draw_idle()
        logging.debug("Graph data updated up to timestamp: %.2f for teeth: %s", current_timestamp, tooth_ids_to_display)
        return self.figure


if __name__ == '__main__':
    # ... (main block for testing GraphVisualizer) ...
    from data_acquisition import SensorDataReader # Assuming this is in the parent directory or PYTHONPATH
    from data_processing import DataProcessor
    
    reader = SensorDataReader()
    data = reader.simulate_data(duration=3, num_teeth=3, num_sensor_points_per_tooth=2)
    processor = DataProcessor(data.copy()) # Ensure data is copied
    processor.create_force_matrix() 
    
    print(f"Force matrix dtype in test: {processor.force_matrix.dtype}")

    if processor.tooth_ids and processor.timestamps:
        visualizer = GraphVisualizer(processor)
        teeth_to_show = processor.tooth_ids[:min(2, len(processor.tooth_ids))]
        fig = visualizer.create_graph(teeth_to_show)
        
        if fig:
            plt.ion()
            fig.show()
            for t_step in np.linspace(processor.timestamps[0], processor.timestamps[-1], num=50):
                visualizer.update_graph_to_timestamp(t_step, teeth_to_show)
                plt.pause(0.05) # Shorter pause for faster test
            plt.ioff()
            plt.show() 
    else:
        print("No tooth IDs or timestamps available for graph test.")
# --- END OF FILE graph_visualization.py ---
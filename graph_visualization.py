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
        self.active_legend = None # To manage legend recreation

    def create_graph_figure(self, figsize=(10, 4)):
        """Creates the Matplotlib figure and axes if they don't exist."""
        if self.figure is None or self.ax is None:
            # Close any pre-existing figure that might be lingering from a previous run in an interactive session
            # This is more for development convenience.
            # open_figs = plt.get_fignums()
            # if open_figs:
            #     logging.debug(f"Closing pre-existing Matplotlib figures: {open_figs}")
            #     for fig_num in open_figs:
            #         plt.close(plt.figure(fig_num))

            self.figure, self.ax = plt.subplots(figsize=figsize)
            logging.info("Matplotlib figure and axes created.")
        else:
            self.ax.clear() # Clear existing axes content if figure already exists
            self.lines.clear()
            self.full_data_cache.clear()
            if self.active_legend:
                self.active_legend.remove()
                self.active_legend = None
            logging.info("Matplotlib axes cleared for new plot.")
        
        # Set fixed X and Y axis limits based on the entire dataset
        if self.processor.timestamps and len(self.processor.timestamps) > 0:
            self.ax.set_xlim(self.processor.timestamps[0], self.processor.timestamps[-1])
        else: 
            self.ax.set_xlim(0, 1) 
            
        max_y_force = 100.0 
        fm_for_ylim = self.processor.force_matrix
        if fm_for_ylim is not None and fm_for_ylim.size > 0:
            if fm_for_ylim.dtype != float and fm_for_ylim.dtype != np.float64 and fm_for_ylim.dtype != np.float32:
                try: fm_for_ylim = fm_for_ylim.astype(float)
                except ValueError: fm_for_ylim = np.array([[max_y_force]], dtype=float)
            valid_forces = fm_for_ylim[~np.isnan(fm_for_ylim)]
            if valid_forces.size > 0: max_y_force = np.max(valid_forces)
            elif hasattr(self.processor, 'max_force_overall'): max_y_force = self.processor.max_force_overall
        self.ax.set_ylim(bottom=-5, top=max_y_force * 1.1 if max_y_force > 0 else 10)
        
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Average Force (N)")
        self.ax.set_title("Average Bite Force Over Time") # Title can be updated later if needed
        self.ax.grid(True)
        return self.figure, self.ax

    def plot_tooth_lines(self, tooth_ids_to_display):
        """Plots lines for the specified teeth on the existing axes."""
        if self.ax is None:
            logging.error("Axes not initialized. Call create_graph_figure first.")
            return

        # Clear only lines from previous plot_tooth_lines call, not the whole axes
        for line in list(self.ax.lines): # Iterate over a copy for safe removal
            line.remove()
        self.lines.clear()
        self.full_data_cache.clear() # Re-cache data for new teeth
        if self.active_legend:
            self.active_legend.remove()
            self.active_legend = None

        if not tooth_ids_to_display:
            logging.warning("No tooth IDs provided to plot_tooth_lines.")
            self.ax.set_title("Average Bite Force Over Time (No tooth selected)")
            if self.figure: self.figure.canvas.draw_idle()
            return

        num_lines = len(tooth_ids_to_display)
        colors = plt.cm.viridis(np.linspace(0, 1, max(1, num_lines))) # max(1,..) to avoid error if num_lines is 0

        for i, tooth_id in enumerate(tooth_ids_to_display):
            full_times, full_forces = self.processor.get_average_force_for_tooth(tooth_id)
            self.full_data_cache[tooth_id] = (full_times, full_forces)
            line, = self.ax.plot([], [], label=f"Tooth {tooth_id} (Avg)", color=colors[i % len(colors)])
            self.lines[tooth_id] = line
        
        self.active_legend = self.ax.legend(loc='upper right')
        title_suffix = f"(Teeth: {', '.join(map(str, tooth_ids_to_display))})" if tooth_ids_to_display else "(No tooth selected)"
        self.ax.set_title(f"Average Bite Force Over Time {title_suffix}")
        
        logging.info("Graph lines plotted for teeth: %s", tooth_ids_to_display)
        if self.figure: self.figure.canvas.draw_idle()

    def update_graph_to_timestamp(self, current_timestamp, tooth_ids_currently_plotted): # Parameter is tooth_ids_currently_plotted
        """Updates lines to show data up to current_timestamp for currently plotted teeth."""
        if self.figure is None or self.ax is None: return

        for tooth_id in tooth_ids_currently_plotted: 
            if tooth_id in self.lines and tooth_id in self.full_data_cache:
                full_times, full_forces = self.full_data_cache[tooth_id]
                if full_times is not None and len(full_times) > 0:
                    idx_up_to_time = np.searchsorted(full_times, current_timestamp, side='right')
                    times_to_plot = full_times[:idx_up_to_time]
                    forces_to_plot = full_forces[:idx_up_to_time]
                    self.lines[tooth_id].set_data(times_to_plot, forces_to_plot)
                else:
                    self.lines[tooth_id].set_data([], []) 
        if self.figure: self.figure.canvas.draw_idle()
        
        # --- CORRECTED LOGGING STATEMENT ---
        logging.debug("Graph data updated up to timestamp: %.2f for teeth: %s", 
                      current_timestamp, tooth_ids_currently_plotted) 
        # --- END CORRECTION ---
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
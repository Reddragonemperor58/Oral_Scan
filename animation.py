# --- START OF FILE animation.py ---
import matplotlib
try:
    matplotlib.use('Qt5Agg') 
    print("INFO: Using Matplotlib backend: Qt5Agg")
except ImportError:
    print("WARNING: PyQt5 not found, Matplotlib will use its default backend (likely TkAgg).")
    pass # Fallback to default if PyQt5 is not available

import time # Not strictly needed by controller anymore, but good for general use
import logging
import matplotlib.pyplot as plt 

from data_acquisition import SensorDataReader
from data_processing import DataProcessor
from graph_visualization import GraphVisualizer
from dental_arch_visualization import DentalArchVisualizer # This should be your T-Scan grid style visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnimationController:
    def __init__(self, graph_visualizer, arch_visualizer, fps=10):
        self.graph_visualizer = graph_visualizer
        self.arch_visualizer = arch_visualizer # This is the T-Scan grid style one
        self.fps = fps
        self.graph_time_indicator = None # To show current time on the Matplotlib graph

    def start_animation_integrated(self, graph_tooth_ids_to_display):
        logging.info("Starting integrated animation (T-Scan Grid Style Arch + Line Graph)...")

        # 1. Prepare Matplotlib graph (will be initially empty for gradual build-up)
        if self.graph_visualizer.figure is None:
            if not graph_tooth_ids_to_display:
                logging.warning("No teeth selected for graph. Matplotlib window will not be shown.")
            else:
                # create_graph now sets up empty lines with fixed axes
                self.graph_visualizer.create_graph(graph_tooth_ids_to_display)
        
        if self.graph_visualizer.figure: # Only proceed if graph figure was created
            plt.ion() # Interactive mode ON for Matplotlib
            self.graph_visualizer.figure.show() # Show it once
            # Ensure the window is drawn and responsive before animation starts
            self.graph_visualizer.figure.canvas.flush_events() 
            self.graph_visualizer.figure.canvas.draw_idle()

        # 2. Define the combined animation step (this function will be called by Vedo's timer)
        def integrated_animation_step(event=None):
            # First, advance the arch visualizer's animation state (which renders the arch grid)
            self.arch_visualizer.animate(event) 

            # Get the current timestamp from the arch visualizer
            current_timestamp = self.arch_visualizer.last_animated_timestamp
            
            if current_timestamp is not None:
                # Update the Matplotlib graph to show data up to the current timestamp
                if self.graph_visualizer.figure and self.graph_visualizer.ax: 
                    self.graph_visualizer.update_graph_to_timestamp(
                        current_timestamp, graph_tooth_ids_to_display
                    )

                    # Update or create the vertical time indicator line on the graph
                    if self.graph_time_indicator:
                        try:
                            self.graph_time_indicator.remove()
                        except (ValueError, AttributeError): # Handle if already removed or None
                            pass 
                    
                    self.graph_time_indicator = self.graph_visualizer.ax.axvline(
                        current_timestamp, color='red', linestyle='--', lw=0.8, gid="time_indicator"
                    )
                    # Request Matplotlib to redraw its canvas if it hasn't already
                    self.graph_visualizer.figure.canvas.draw_idle() 
                
                logging.debug(f"Integrated frame: Arch and Graph updated for time {current_timestamp:.1f}s")
            
            # No plt.pause() here - relies on plt.ion() and draw_idle()

        # 3. Configure Vedo arch visualizer to use this integrated animation step
        self.arch_visualizer.animate_callback_ref = integrated_animation_step # Store reference
        
        # Ensure any old timer/callback is cleared before setting a new one
        self.arch_visualizer.plotter.timer_callback('destroy')
        self.arch_visualizer.plotter.remove_callback('timer') 
        
        self.arch_visualizer.plotter.add_callback('timer', self.arch_visualizer.animate_callback_ref)
        
        dt = int(1000 / self.fps) # Calculate delay in milliseconds
        self.arch_visualizer.plotter.timer_callback('create', dt=dt)
        logging.info(f"Vedo animation timer started with dt={dt}ms for integrated step.")

        # 4. Start Vedo's event loop (this is a blocking call)
        self.arch_visualizer.plotter.show(title="Dental Arch (Grid) & Force Graph Animation")
        
        # After Vedo window closes (show() returns):
        if self.graph_visualizer.figure:
            plt.ioff() # Turn off Matplotlib interactive mode
        logging.info("Animation stopped and Vedo window closed.")

    def stop_animation(self): # For potential external calls to stop
        if self.arch_visualizer and self.arch_visualizer.plotter:
            self.arch_visualizer.plotter.timer_callback('destroy') # Stop Vedo's timer
            # Consider closing plotter if this is meant to fully halt:
            # self.arch_visualizer.plotter.close() 
        logging.info("Animation stopping sequence initiated.")


if __name__ == '__main__':
    reader = SensorDataReader()
    # Simulate data with multiple sensor points per tooth for heatmap visualization
    data = reader.simulate_data(duration=5, num_teeth=16, num_sensor_points_per_tooth=4) 
    
    processor = DataProcessor(data.copy()) # Use a copy to avoid modifying original data
    
    # Ensure data is processed and force matrix is created
    if processor.cleaned_data is None: 
        processor.create_force_matrix()

    if not processor.timestamps:
        logging.error("Data processing failed to produce timestamps. Cannot animate.")
    else:
        graph_viz = GraphVisualizer(processor)
        
        # Select teeth to display on the Matplotlib graph (e.g., first two)
        teeth_for_graph = []
        if processor.tooth_ids: # Ensure tooth_ids are populated
            if len(processor.tooth_ids) >= 2:
                teeth_for_graph = [processor.tooth_ids[0], processor.tooth_ids[1]]
            elif len(processor.tooth_ids) == 1:
                teeth_for_graph = [processor.tooth_ids[0]] # Handle case of single tooth
        
        if not teeth_for_graph:
             logging.warning("No specific teeth selected for graph. Graph display might be skipped or empty if not handled in create_graph.")

        # DentalArchVisualizer is now the T-Scan grid style one by default from your latest version
        arch_viz = DentalArchVisualizer(processor) 
        
        animator = AnimationController(graph_viz, arch_viz, fps=10) # Adjust FPS as needed
        animator.start_animation_integrated(teeth_for_graph)
# --- END OF FILE animation.py ---
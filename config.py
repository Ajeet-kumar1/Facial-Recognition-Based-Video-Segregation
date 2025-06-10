""" Write the important configuration and hyperparameters."""

# Folder directory
matched_dir = "matched/"
unmatched_dir = "unmatched/"
log_file = "log.csv"

frame_interval_sec = 1                             # Time interval between frames to check      
model_name = "Facenet512"                          # Select the model
detector_backend = 'mtcnn'                         # Face detection model        
threshold = 0.4                                    # Decide the threshold
align = True
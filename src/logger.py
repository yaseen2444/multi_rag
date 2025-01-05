import logging
import os
from datetime import datetime

# creating a custom logger with the basic config
LOG_FILE=f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
log_dir=os.path.join("LOG",LOG_FILE)
os.makedirs(log_dir,exist_ok=True)

log_file_path=os.path.join(log_dir,LOG_FILE)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO

)
logging.info("logging has started")
import os
import sys

# Point TF to cuDNN libs installed via pip
site_packages = os.path.join(sys.prefix, "lib", "python3.8", "site-packages")
cudnn_lib = os.path.join(site_packages, "nvidia", "cudnn", "lib")

if os.path.exists(cudnn_lib):
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if cudnn_lib not in current_ld:
        os.environ["LD_LIBRARY_PATH"] = cudnn_lib + ":" + current_ld
        # Relaunch process with updated LD_LIBRARY_PATH
        os.execv(sys.executable, [sys.executable] + sys.argv)

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
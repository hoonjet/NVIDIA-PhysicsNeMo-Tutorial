import warp as wp
print(f"Warp version: {wp.config.version}")
wp.init()
print("Warp initialized OK!")

from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
print(f"Darcy2D imported: {Darcy2D}")
import inspect
print(f"Constructor: {inspect.signature(Darcy2D.__init__)}")
print("All imports OK!")

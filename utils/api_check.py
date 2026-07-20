import inspect
from physicsnemo.models.meshgraphnet import MeshGraphNet
print("=== MeshGraphNet ===")
print(inspect.signature(MeshGraphNet.__init__))
print()
print(MeshGraphNet.__doc__)

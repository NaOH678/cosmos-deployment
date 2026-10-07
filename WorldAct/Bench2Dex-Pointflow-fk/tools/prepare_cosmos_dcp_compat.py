"""Re-encode only pathlib metadata for older Python; symlink unchanged DCP tensors."""
import argparse
import hashlib
import json
import pathlib
import pickle
from torch.distributed.checkpoint import FileSystemReader

class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in {'pathlib._local', 'pathlib._abc'} and hasattr(pathlib, name):
            return getattr(pathlib, name)
        return super().find_class(module, name)

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=pathlib.Path,required=True)
p.add_argument('--destination',type=pathlib.Path,required=True)
a=p.parse_args()
source=a.source.resolve();dest=a.destination.resolve()
if source==dest: raise ValueError('Source must remain untouched')
with (source/'.metadata').open('rb') as f: metadata=MetadataUnpickler(f).load()
for extent in metadata.storage_data.values():
    shard=source/extent.relative_path
    if shard.stat().st_size < extent.offset+extent.length:
        raise ValueError(f'Truncated shard: {shard}')
dest.mkdir(parents=True,exist_ok=True)
for shard in source.glob('*.distcp'):
    link=dest/shard.name
    if not link.exists(): link.symlink_to(shard)
    elif link.resolve()!=shard: raise ValueError(f'Unexpected existing path {link}')
(dest/'.metadata').write_bytes(pickle.dumps(metadata,protocol=4))
read=FileSystemReader(dest).read_metadata()
assert read.state_dict_metadata==metadata.state_dict_metadata
assert read.storage_data==metadata.storage_data
report=dict(source=str(source),destination=str(dest),tensor_count=len(read.state_dict_metadata),
    tensor_files='symlinks to originals; no weight copy or change',
    original_metadata_sha256=hashlib.sha256((source/'.metadata').read_bytes()).hexdigest(),
    local_metadata_sha256=hashlib.sha256((dest/'.metadata').read_bytes()).hexdigest())
(dest/'compatibility.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

"""Restore only omitted input images from the source Ready ZIP; no downloads."""
import argparse,json,zipfile
from pathlib import Path,PurePosixPath

def main():
    ap=argparse.ArgumentParser();ap.add_argument('archive',type=Path);a=ap.parse_args()
    root=Path(__file__).resolve().parent
    info=json.loads((root/'PROVENANCE.json').read_text(encoding='utf-8'))
    wanted={x['path']:x for x in info['omitted_inputs']}
    with zipfile.ZipFile(a.archive) as z:
        found=set()
        for member in z.infolist():
            parts=PurePosixPath(member.filename).parts
            rel='/'.join(parts[1:]) if info['strip_root'] else member.filename
            if rel not in wanted:continue
            expected=wanted[rel]
            if member.file_size!=expected['bytes'] or member.CRC!=expected['crc32']:
                raise ValueError('Source archive member differs: '+rel)
            dest=(root/rel).resolve()
            if not dest.is_relative_to(root):raise ValueError('Unsafe member: '+rel)
            if dest.exists():raise FileExistsError(dest)
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(member));found.add(rel)
    if found!=set(wanted):raise RuntimeError('Missing members: '+str(sorted(set(wanted)-found)))
    print('Restored',len(found),'input images')
if __name__=='__main__':main()

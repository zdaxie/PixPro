import sys
import torch
import subprocess

if __name__ == '__main__':
    in_file = sys.argv[1]
    ckpt = torch.load(in_file, map_location='cpu')

    for key in ['opt', 'optimizer', 'scheduler', 'epoch', 'amp']:
        if ckpt.get(key):
            del ckpt[key]

    out_file = sys.argv[2]
    torch.save(ckpt, out_file)

    md5sum = subprocess.check_output(['md5sum', out_file]).decode()
    out_file_with_md5sum = out_file.split('.')[0] + f'_md5_{md5sum[:8]}.pth'
    subprocess.Popen(['mv', out_file, out_file_with_md5sum])

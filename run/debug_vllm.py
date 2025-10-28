# -*- coding: utf-8 -*-
import sys
from vllm.entrypoints.cli.main import main
from dots_ocr import modeling_dots_ocr_vllm

# from batch_invariant_ops import enable_batch_invariant_mode
# enable_batch_invariant_mode()
# from minicpmv4 import modeling_minicpmv4_vllm
if __name__ == "__main__":
    main()
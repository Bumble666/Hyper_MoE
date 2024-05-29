# HyperMoE
The official implementation of the paper "HyperMoE: Towards Better Mixture of Experts via Transferring Among Experts"

## Installation 

Install pytorch (2.0.0 recommended). Install required packages, preferably in a virtualenv: `pip install -r requirements.txt`.

## Usage

`gpt-2-moe` is the code we used to modify GPT-2 into an MoE structure by duplicating weights and then conducted experiments on Wikitext.

`switch_transformer` is the code we used to conduct experiments on Switch Transformer.

You can choose a task from `./tasks` and run the corresponding sh script to conduct experiments,

The scripts with the suffix `hmoe` represent the method we propose.

For text classification:

    # Run Switch Transformer MoE for text classification:

    sh switch_transformer/tasks/text-classification/run_glue.sh

    Run Switch Transformer HyperMoE for text classification:
    
    sh switch_transformer/tasks/text-classification/run_glue_hmoe.sh
    
For summarization:

    # Run Switch Transformer MoE for summarization:
    
    sh switch_transformer/tasks/summarization/run_summarization.sh
    
    # Run Switch Transformer HyperMoE for summarization:
    
    sh switch_transformer/tasks/summarization/run_summarization_hmoe.sh

For question-answering:
    
    # Run Switch Transformer MoE for question-answering:
    
    sh switch_transformer/tasks/question-answering/run_seq2seq_qa.sh
    
    # Run Switch Transformer HyperMoE for question-answering:
    
    sh switch_transformer/tasks/question-answering/run_seq2seq_qa_hmoe.sh

For language-modeling:
    
    # Run GPT-2 MoE for language-modeling:
    
    sh gpt-2-moe/tasks/language-modeling/run_clm.sh
    
    # Run GPT-2 HyperMoE for language-modeling:
    
    sh gpt-2-moe/tasks/language-modeling/run_clm_hmoe.sh


We modified `gpt-2-moe/transformers/models/gpt2/modeling_gpt2.py` and `switch_transformer/transformers/models/switch_transformers/modeling_switch_transformers.py` to accommodate the method we proposed.

## Acknowledgments

The implementations of the codebase are from the [Merging Experts into One](https://github.com/Shwai-He/MEO) repository. Huge thanks to the contributors of the amazing repository!

## Citation

if you find this repository useful, please cite our paper:

    @inproceedings{hao2024hmoe,
      title={HyperMoE: Towards Better Mixture of Experts via Transferring Among Experts},
      author={Hao Zhao, Zihan Qiu, Huijia Wu, Zili Wang, Zhaofeng He, Jie Fu},
      booktitle={Proceedings of ACL},
      year={2024}
    }

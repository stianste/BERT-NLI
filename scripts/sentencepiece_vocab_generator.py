import sentencepiece as spm

base_path = './data/NLI-shared-task-2017'
model_prefix = base_path + '/toefl'
vocab_size = 3600

arguments = f'--input={base_path}/all_lower.txt \
             --model_prefix={model_prefix} \
             --vocab_size={vocab_size} \
             --model_type=bpe \
             --character_coverage=1.0'

def create_wordpiece_file_from_sentencepiece_file(sentencepiece_file_path):
    with open(f'{model_prefix}_wordpiece_size_{vocab_size}.vocab', 'w') as wordpiece_file:
        with open(sentencepiece_file_path, 'r') as sentencepiece_file:
            for line in sentencepiece_file.readlines():
                token = line.split()[0]
                if token[0] == '▁': # Fat underline
                    new_token = token[1:]
                else:
                    new_token = '##' + token

                wordpiece_file.write(new_token + '\n')

        special_tokens = '[CLS]\n[UNK]\n[SEP]\n[PAD]\n[MASK]'
        wordpiece_file.write(special_tokens)

spm.SentencePieceTrainer.Train(arguments)
create_wordpiece_file_from_sentencepiece_file(f'{model_prefix}.vocab')

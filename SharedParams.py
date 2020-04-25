def get_context_size():
    return 10

def get_amino_3grams_dict():
    amino_chars_single = 'ACDEFGHIKLMNPQRSTVWY'
    amino_3grams = ['{}{}{}'.format(amino_chars_single[i], amino_chars_single[j], amino_chars_single[k]) for i in
                    range(len(amino_chars_single)) for j in range(len(amino_chars_single)) for k in
                    range(len(amino_chars_single))]
    return {three_gram: i for i, three_gram in enumerate(amino_3grams)}